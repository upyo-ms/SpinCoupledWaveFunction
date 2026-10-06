"""Optional float64 JAX backend; the NumPy augmented-Lagrangian solver is retained."""
import math

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from NonorthgonalEnergy import SCWF_Energy as NumPyEnergy


@jax.jit
def _energy(x, overlap, hcore, eri, representations, left, right):
    nao, ne = overlap.shape[0], left.shape[1]
    C = x[:nao * ne].reshape(nao, ne)
    cs = x[nao * ne:]
    cs = cs / jnp.linalg.norm(cs)
    s = C.T @ overlap @ C
    h = C.T @ hcore @ C
    g = eri
    for _ in range(4):
        g = jnp.tensordot(g, C, axes=(0, 0))
    weights = jnp.einsum('i,pij,j->p', cs, representations, cs)
    factors = s[left, right]
    count = math.factorial(ne)
    norm = jnp.sum(weights[:count] * jnp.prod(factors[:count], axis=1))
    one = jnp.sum(weights * h[left[:, 0], right[:, 0]]
                  * jnp.prod(factors[:, 1:], axis=1)) / math.factorial(ne - 1)
    two = jnp.sum(weights * g[left[:, 0], right[:, 0], left[:, 1], right[:, 1]]
                  * jnp.prod(factors[:, 2:], axis=1)) / (2 * math.factorial(ne - 2))
    return one + two, norm


def _target(x, mu, lam, *data):
    energy, norm = _energy(x, *data)
    error = norm - 1.
    return energy + mu * error ** 2 + lam * error


def _rayleigh(x, *data):
    energy, norm = _energy(x, *data)
    return energy / norm


_target_vg = jax.jit(jax.value_and_grad(_target))
_rayleigh_vg = jax.jit(jax.value_and_grad(_rayleigh))


class SCWF_Energy(NumPyEnergy):
    """Same public optimizer and result tuple, with JIT/autodiff on a device.

    Integral and permutation data are copied to the chosen device at creation.
    Construct a new instance when changing the molecule or its integrals.
    SciPy optimization and PySCF initialization still run on the CPU.
    """
    def __init__(self, mol, device=None, compute_fci=True, energy_evaluation='permutation', active_space=None, determinant_size_profile='standard'):
        if energy_evaluation not in ('permutation', 'determinant'):
            raise ValueError('energy_evaluation must be permutation or determinant')
        super().__init__(mol, compute_fci=compute_fci, energy_evaluation=energy_evaluation,
                         active_space=active_space, determinant_size_profile=determinant_size_profile)
        self._initialize_jax(device, energy_evaluation)

    def _initialize_jax(self, device=None, energy_evaluation='permutation'):
        if device not in (None, 'cpu', 'gpu'):
            raise ValueError("device must be None, 'cpu', or 'gpu'")
        devices = jax.devices(device) if device else jax.devices()
        self.device = devices[0]
        if energy_evaluation == 'permutation':
            indices = np.asarray(self.IndexElectron, dtype=np.int32)
            arrays = (self.S, self.Hcore, self.JK, np.asarray(self.DN),
                      indices[:, 0], indices[:, 1])
            self._jax_data = tuple(jax.device_put(
                np.asarray(a, dtype=np.float64 if i < 4 else np.int32), self.device)
                for i, a in enumerate(arrays))
        self._last_evaluation = None
        self.energy_evaluation = 'permutation'
        self._stable_energy = None
        if energy_evaluation == 'determinant':
            from StableEnergyJAX import make_evaluator
            self._stable_energy = make_evaluator(self)
            def target(x, mu, lam):
                e, norm = self._stable_energy(x)
                return e + mu * (norm - 1.) ** 2 + lam * (norm - 1.)
            def rayleigh(x):
                e, norm = self._stable_energy(x)
                return e / norm
            self._stable_target_vg = jax.jit(jax.value_and_grad(target))
            self._stable_rayleigh_vg = jax.jit(jax.value_and_grad(rayleigh))
            self.energy_evaluation = 'determinant'

    def _parameters(self, x):
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise ValueError('JAX backend expects real parameters; gradients use autodiff')
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(self.S) * self.Ne + self.fs,):
            raise ValueError('Invalid orbital/spin coefficient shape')
        if not np.isfinite(x).all() or np.linalg.norm(x[-self.fs:]) == 0:
            raise FloatingPointError('Coefficients must be finite with nonzero spin norm')
        return x

    def calcEnergy(self, C, Cs):
        x = self._parameters(np.r_[np.asarray(C).ravel(), np.asarray(Cs).ravel()])
        device_x = jax.device_put(x, self.device)
        result = np.asarray(self._stable_energy(device_x) if self._stable_energy is not None
                            else _energy(device_x, *self._jax_data))
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite JAX energy or norm')
        return float(result[0]), float(result[1])

    def _wavefunction_call(self, method, *arrays):
        if self._stable_energy is None:
            raise ValueError('Wavefunction geometry requires determinant energy evaluation')
        function = getattr(self._stable_energy, method)
        result = function(*(jax.device_put(np.asarray(a, dtype=np.float64), self.device) for a in arrays))
        if isinstance(result, tuple):
            return tuple(np.asarray(a) for a in result)
        return np.asarray(result)

    def orbital_wavefunction_metric(self, C, Cs):
        return self._wavefunction_call('orbital_metric', C, Cs)

    def joint_wavefunction_metric(self, C, Cs, basis):
        return self._wavefunction_call('joint_metric', C, Cs, basis)

    def wavefunction_overlap(self, C, Cs, other_C, other_Cs):
        return float(self._wavefunction_call('wavefunction_overlap', C, Cs, other_C, other_Cs))

    def spin_overlap_matrix(self, C):
        return self._wavefunction_call('overlap_matrix', C)

    def projected_spin_matrices(self, C, basis, reference):
        return self._wavefunction_call('projected_matrices', C, basis, reference)

    def single_orbital_overlap(self, C, Cs, index):
        function = self._stable_energy.orbital_overlap if self._stable_energy is not None else None
        if function is None:
            raise ValueError('Single-orbital optimization requires determinant energy evaluation')
        return np.asarray(function(C, Cs, np.int32(index)))

    def projected_orbital_matrices(self, C, Cs, index, basis, reference):
        if self._stable_energy is None:
            raise ValueError('Single-orbital optimization requires determinant energy evaluation')
        return tuple(np.asarray(a) for a in self._stable_energy.orbital_projected(
            C, Cs, np.int32(index), basis, reference))

    def _value_gradient(self, x, kind, mu=0., lam=0.):
        x = self._parameters(x)
        key = (kind, float(mu), float(lam))
        cached = self._last_evaluation
        if cached is not None and cached[0] == key and np.array_equal(cached[1], x):
            return cached[2], cached[3].copy()
        device_x = jax.device_put(x, self.device)
        if self._stable_energy is not None:
            if kind == 'target':
                value, gradient = self._stable_target_vg(
                    device_x, jax.device_put(np.float64(mu), self.device),
                    jax.device_put(np.float64(lam), self.device))
            else:
                value, gradient = self._stable_rayleigh_vg(device_x)
        elif kind == 'target':
            value, gradient = _target_vg(device_x, jax.device_put(np.float64(mu), self.device),
                                         jax.device_put(np.float64(lam), self.device), *self._jax_data)
        else:
            value, gradient = _rayleigh_vg(device_x, *self._jax_data)
        value, gradient = float(value), np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Non-finite JAX objective or gradient')
        self._last_evaluation = (key, x.copy(), value, gradient.copy())
        return value, gradient

    def gen_target(self, mu, lam):
        return lambda x: self._value_gradient(x, 'target', mu, lam)[0]

    def target_gradient(self, x, mu, lam):
        return self._value_gradient(x, 'target', mu, lam)[1]

    def rayleigh_gradient(self, x):
        return self._value_gradient(x, 'rayleigh')[1]
