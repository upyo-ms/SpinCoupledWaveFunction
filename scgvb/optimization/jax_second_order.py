"""Opt-in FP64 autodifferentiated SCGVB Rayleigh curvature.

Only orbital and spin-coupling coefficients are variables. Polynomial minors
avoid QR rank branches and inverse-minor derivatives. The full Rayleigh Hessian
is differentiated on the selected JAX device; metric projection and safeguarded
steps remain shared CPU code. No independent determinant optimization is used.
"""
import numpy as np
from math import comb
from pyscf import ao2mo

from scgvb.optimization.core_response import AmplitudeDerivatives, project_curvature


def curvature_model(system):
    """Reuse compiled derivatives across Hamiltonian changes in a core solve."""
    import jax
    device = getattr(system, 'device', None)
    if device is None:
        device = jax.devices('cpu')[0]
    key = (system.Ne, system.Sm, np.asarray(system.S).tobytes(), str(device))
    previous = getattr(system, '_jax_second_order_curvature', None)
    if previous is None or previous[0] != key:
        previous = key, JAXCurvature(system, device)
        system._jax_second_order_curvature = previous
    return previous[1]


class JAXCurvature:
    def __init__(self, system, device):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        from scgvb.energy.compound_minors import make_compounds
        from scgvb.energy.stable_jax import make_hamiltonian

        # This constructor builds index tables only, never derivative jets.
        metadata = AmplitudeDerivatives(system)
        self.n, self.ne, self.p = metadata.n, metadata.ne, metadata.p
        self.ci_shape = (comb(self.n, metadata.na), comb(self.n, metadata.nb))
        self.device = device
        self.overlap = np.asarray(system.S).copy()
        w, v = np.linalg.eigh(self.overlap)
        self.inverse = np.einsum('ik,k,jk->ij', v, 1/np.sqrt(w), v, optimize=False)
        compounds, _ = make_compounds(self.n, self.ne, (metadata.na, metadata.nb), device)
        root, spin_basis, ai, bi, signs = [jax.device_put(a, device) for a in
            (metadata.root, metadata.spin_basis, np.asarray(metadata.alpha),
             np.asarray(metadata.beta), metadata.signs)]
        apply_h = make_hamiltonian(np.zeros((self.n, self.n)),
                                  np.zeros((self.n,)*4), metadata.na, metadata.nb, device)

        def amplitudes(x):
            C = x[:self.n*self.ne].reshape(self.n, self.ne)
            Cs = x[self.n*self.ne:]
            alpha, beta = compounds(root @ C)
            weights = (spin_basis @ Cs)*signs
            a, b = alpha[:, ai], beta[:, bi]
            ci = (a*weights[None, :]) @ b.T
            # Norm of each primitive contribution, without a 3D temporary.
            absolute = jnp.sum(jnp.abs(weights)*jnp.linalg.norm(a, axis=0)
                               * jnp.linalg.norm(b, axis=0))
            return ci, absolute

        def normalized(x):
            ci, _ = amplitudes(x)
            return ci.ravel()/jnp.linalg.norm(ci)

        def rayleigh(x, parameters, reference):
            ci, _ = amplitudes(x)
            hc = apply_h(ci, *parameters)
            # The reference is constant with respect to x. Shifting before
            # contraction reduces subtraction of large Rayleigh derivatives.
            return jnp.sum(ci*(hc-reference*ci))/jnp.sum(ci*ci)

        def evaluate(x, parameters, reference):
            value, gradient = jax.value_and_grad(rayleigh)(x, parameters, reference)
            hessian = jax.hessian(rayleigh, argnums=0)(x, parameters, reference)
            ci, absolute = amplitudes(x)
            jacobian = jax.jacfwd(normalized)(x)
            return value+reference, gradient, hessian, jacobian, normalized(x), absolute/jnp.linalg.norm(ci)

        self._reference = jax.jit(lambda x, params: rayleigh(x, params, 0.))
        self._evaluate = jax.jit(evaluate)

    def evaluate(self, C, Cs, one, two):
        import jax
        from scgvb.energy.stable_jax import hamiltonian_factors

        C, Cs = np.asarray(C, dtype=float), np.asarray(Cs, dtype=float)
        if C.shape != (self.n, self.ne) or Cs.shape != (self.p-self.n*self.ne,):
            raise ValueError('Invalid SCGVB coefficient shape for second_order')
        if not np.array_equal(self.overlap, np.eye(self.n)):
            intermediate = np.einsum('ip,ij->pj', self.inverse, one, optimize=False)
            one = np.einsum('pi,iq->pq', intermediate, self.inverse, optimize=False)
            two = ao2mo.incore.full(two, self.inverse, compact=False).reshape((self.n,)*4)
        parameters = tuple(jax.device_put(a, self.device) for a in hamiltonian_factors(one, two))
        x = jax.device_put(np.r_[C.ravel(), Cs], self.device)
        reference = self._reference(x, parameters)
        energy, gradient, hessian, jacobian, ci, cancellation = (
            np.asarray(a) for a in self._evaluate(x, parameters, reference))
        cancellation = float(cancellation)
        if not np.isfinite(cancellation) or cancellation > 1e8:
            raise FloatingPointError('SCGVB second-order wavefunction is too strongly cancelling for FP64 derivatives')
        return project_curvature(float(energy), gradient, hessian, jacobian,
                                 ci.reshape(self.ci_shape), cancellation)
