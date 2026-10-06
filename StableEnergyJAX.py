"""Memory-bounded FP64 SCGVB energy and gradients on one JAX device.

CI amplitudes are exterior products, not independent optimization variables.
The Hamiltonian is an exact sum of symmetric one-body-operator squares. Its
custom VJP saves only c and Hc, not an orbital-pair-by-determinant tensor.
"""
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import ao2mo
from pyscf.fci import cistring
from Spin_Function.PrimitiveSpinFunc import PrSpinFunc_gen
from DeterminantEnergy import determinant_space_size, check_determinant_size
from CompoundMinors import make_compounds


def make_hamiltonian(h, eri, na, nb, device, block_size=32):
    """Return Hc using bounded gathers. No ERI eigenvalues are discarded."""
    n = len(h)
    pair_basis = []
    for p in range(n):
        for q in range(p+1):
            a = np.zeros((n, n))
            if p == q:
                a[p, q] = 1.
            else:
                a[p, q] = a[q, p] = 1 / np.sqrt(2.)
            pair_basis.append(a)
    pairs = np.asarray(pair_basis)
    pair_eri = np.einsum('apq,pqrs,brs->ab', pairs, eri, pairs, optimize=True)
    weights, vectors = np.linalg.eigh(pair_eri)
    factors = np.einsum('ak,apq->kpq', vectors, pairs)
    hshift = h - .5 * np.einsum('pqqs->ps', eri)
    weights, factors, hshift = [jax.device_put(a, device) for a in (weights, factors, hshift)]

    def links(count):
        # Source -> destination links. Transpose the orbital coefficient below
        # to evaluate the operator acting on each output determinant.
        table = cistring.gen_linkstr_index(range(n), count)
        ndet = len(table)
        if count == 0:
            table = np.zeros((ndet, 1, 4), dtype=np.int32)
        size = min(block_size, ndet)
        padded = np.pad(table, ((0, (-ndet) % size), (0, 0), (0, 0)))
        return jax.device_put(padded.reshape(-1, size, table.shape[1], 4), device), ndet

    alinks, adim = links(na)
    blinks, bdim = links(nb)

    def sector(ci, matrix, table, dimension):
        def tile(link):
            values = matrix[link[:, :, 1], link[:, :, 0]] * link[:, :, 3]
            gathered = ci[link[:, :, 2], :]
            return jnp.einsum('bl,blj->bj', values, gathered)
        return jax.lax.map(tile, table).reshape(-1, ci.shape[1])[:dimension]

    def one_body(ci, matrix):
        return sector(ci, matrix, alinks, adim) + sector(ci.T, matrix, blinks, bdim).T

    def apply(ci):
        def add_factor(k, total):
            intermediate = one_body(ci, factors[k])
            return total + .5 * weights[k] * one_body(intermediate, factors[k])
        return jax.lax.fori_loop(0, len(weights), add_factor, one_body(ci, hshift))

    return apply


def quadratic_energy(apply_h):
    """Custom reverse rule for fixed, real symmetric H: (c.Hc, c.c)."""
    def evaluate(c):
        hc = apply_h(c)
        return (jnp.sum(c * hc), jnp.sum(c * c)), (c, hc)

    @jax.custom_vjp
    def energy(c):
        return evaluate(c)[0]

    def backward(residual, cotangents):
        c, hc = residual
        ge, gn = cotangents
        return (2 * (ge * hc + gn * c),)

    energy.defvjp(evaluate, backward)
    return energy


def make_evaluator(system):
    n, ne = len(system.S), system.Ne
    check_determinant_size(n, ne, system.Sm - 1, profile=getattr(system, 'determinant_size_profile', 'standard'))
    na = (ne + system.Sm - 1) // 2
    nb = ne - na
    _, patterns, spin_basis = PrSpinFunc_gen(na, nb)
    w, v = np.linalg.eigh(system.S)
    if np.any(w <= 0):
        raise ValueError('AO overlap must be positive definite')
    root = np.einsum('ik,k,jk->ij', v, np.sqrt(w), v)
    invroot = np.einsum('ik,k,jk->ij', v, 1 / np.sqrt(w), v)
    h = np.einsum('ip,ij,jq->pq', invroot, system.Hcore, invroot)
    eri = ao2mo.incore.full(system.JK, invroot, compact=False).reshape((n,) * 4)
    apply_h = make_hamiltonian(h, eri, na, nb, system.device)
    expectation = quadratic_energy(apply_h)
    compounds, maps = make_compounds(n, ne, (na, nb), system.device)
    alpha = [tuple(np.flatnonzero(p == 1)) for p in patterns]
    beta = [tuple(np.flatnonzero(p == -1)) for p in patterns]
    ai = np.asarray([maps[na][a] for a in alpha], dtype=np.int32)
    bi = np.asarray([maps[nb][b] for b in beta], dtype=np.int32)
    signs = np.asarray([(-1.) ** sum(i < j for i in b for j in a) for a, b in zip(alpha, beta)])
    root, spin_basis, ai, bi, signs = [jax.device_put(a, system.device)
                                      for a in (root, spin_basis, ai, bi, signs)]

    def ci_from_parameters(x):
        C = x[:n * ne].reshape(n, ne)
        cs = x[n * ne:]
        cs = cs / jnp.linalg.norm(cs)
        a, b = compounds(root @ C)
        amplitudes = (spin_basis @ cs) * signs
        # Matrix multiplication avoids a primitive-spin x det-alpha x det-beta tensor.
        return (a[:, ai] * amplitudes[None, :]) @ b[:, bi].T

    def energy(x):
        return expectation(ci_from_parameters(x))

    def normalized_ci(C, cs):
        ci = ci_from_parameters(jnp.concatenate((C.ravel(), cs)))
        return ci / jnp.sqrt(jnp.sum(ci * ci))

    def orbital_metric(C, cs):
        # Sequential JVP/VJP columns avoid a parameter x determinant Jacobian.
        f = lambda c: normalized_ci(c, cs)
        _, tangent = jax.linearize(f, C)
        _, transpose = jax.vjp(f, C)
        def column(v):
            return transpose(tangent(v.reshape(C.shape)))[0].ravel()
        metric = jax.lax.map(column, jnp.eye(C.size))
        return .5 * (metric + metric.T)

    def wavefunction_overlap(C, cs, other_C, other_cs):
        return jnp.sum(normalized_ci(C, cs) * normalized_ci(other_C, other_cs))

    def joint_metric(C, cs, basis):
        def normalized(x):
            ci = ci_from_parameters(x)
            return ci/jnp.sqrt(jnp.sum(ci*ci))
        x = jnp.concatenate((C.ravel(), cs))
        # At most six determinant tangents; no full parameter Jacobian/Hessian.
        states = jax.lax.map(lambda v: jax.jvp(normalized, (x,), (v,))[1], basis.T)
        metric = jnp.einsum('iab,jab->ij', states, states)
        return .5*(metric+metric.T)

    def overlap_matrix(C):
        # No Hamiltonian applications are needed to resolve metric support.
        a, b = compounds(root @ C)
        def forward(cs):
            amplitudes = (spin_basis @ cs) * signs
            return (a[:, ai] * amplitudes[None, :]) @ b[:, bi].T
        def column(cs):
            ci, transpose = jax.vjp(forward, cs)
            return transpose(ci)[0]
        metric = jax.lax.map(column, jnp.eye(system.fs))
        return .5 * (metric + metric.T)

    def projected_matrices(C, basis, reference):
        # First form physical CI vectors. Their Gram matrix avoids subtracting
        # large entries of the ill-conditioned original metric after whitening.
        a, b = compounds(root @ C)
        def column(cs):
            amplitudes = (spin_basis @ cs) * signs
            return (a[:, ai] * amplitudes[None, :]) @ b[:, bi].T
        states = jax.lax.map(column, basis.T)
        shifted = jax.lax.map(lambda ci: apply_h(ci) - reference * ci, states)
        metric = jnp.einsum('iab,jab->ij', states, states)
        h = jnp.einsum('iab,jab->ij', states, shifted)
        return .5 * (h+h.T), .5 * (metric+metric.T)

    def spin_matrices(C):
        a, b = compounds(root @ C)
        def linear(cs):
            amplitudes = (spin_basis @ cs) * signs
            return (a[:, ai] * amplitudes[None, :]) @ b[:, bi].T

        def column(cs):
            ci, transpose = jax.vjp(linear, cs)
            return transpose(apply_h(ci))[0], transpose(ci)[0]

        # Sequential mapping bounds determinant-sized intermediates to one spin state.
        hspin, sspin = jax.lax.map(column, jnp.eye(system.fs))
        return (hspin + hspin.T) * .5, (sspin + sspin.T) * .5

    def orbital_states(C, cs, index, basis):
        def column(coefficients):
            trial = C.at[:, index].set(coefficients)
            return ci_from_parameters(jnp.concatenate((trial.ravel(), cs)))
        # Only one orbital's trial vectors are materialized, never all Ne*n.
        return jax.lax.map(column, basis.T)

    def orbital_overlap(C, cs, index):
        states = orbital_states(C, cs, index, jnp.eye(n))
        metric = jnp.einsum('iab,jab->ij', states, states)
        return .5*(metric+metric.T)

    def orbital_projected(C, cs, index, basis, reference):
        states = orbital_states(C, cs, index, basis)
        shifted = jax.lax.map(lambda ci: jax.lax.cond(
            jnp.any(ci != 0), lambda a: apply_h(a)-reference*a,
            lambda a: jnp.zeros_like(a), ci), states)
        metric = jnp.einsum('iab,jab->ij', states, states)
        h = jnp.einsum('iab,jab->ij', states, shifted)
        return .5*(h+h.T), .5*(metric+metric.T)

    if ne <= 4 and n > ne:
        evaluator = make_small_evaluator(system, energy)
    else:
        evaluator = jax.jit(energy)
    evaluator.spin_matrices = jax.jit(spin_matrices)
    evaluator.orbital_metric = jax.jit(orbital_metric)
    evaluator.wavefunction_overlap = jax.jit(wavefunction_overlap)
    evaluator.joint_metric = jax.jit(joint_metric)
    evaluator.overlap_matrix = jax.jit(overlap_matrix)
    evaluator.projected_matrices = jax.jit(projected_matrices)
    evaluator.orbital_overlap = jax.jit(orbital_overlap)
    evaluator.orbital_projected = jax.jit(orbital_projected)
    return evaluator


def make_small_evaluator(system, fallback):
    """Exact moving orbital-span CI for <=4 electrons; no orbital truncation.

    QR only changes the representation of the current SCGVB orbitals. At a
    rank-deficient C, use the polynomial AO evaluator (QR derivatives would
    be singular). The two branches describe exactly the same wavefunction.
    """
    n, ne = len(system.S), system.Ne
    na = (ne + system.Sm - 1) // 2
    nb = ne - na
    _, patterns, spin_basis = PrSpinFunc_gen(na, nb)
    w, v = np.linalg.eigh(system.S)
    root = np.einsum('ik,k,jk->ij', v, np.sqrt(w), v)
    invroot = np.einsum('ik,k,jk->ij', v, 1 / np.sqrt(w), v)
    compounds, maps = make_compounds(ne, ne, (na, nb), system.device)
    alpha = [tuple(np.flatnonzero(p == 1)) for p in patterns]
    beta = [tuple(np.flatnonzero(p == -1)) for p in patterns]
    ai = np.asarray([maps[na][a] for a in alpha])
    bi = np.asarray([maps[nb][b] for b in beta])
    signs = np.asarray([(-1.) ** sum(i < j for i in b for j in a)
                        for a, b in zip(alpha, beta)])

    def operators(count):
        dim = len(cistring.make_strings(range(ne), count))
        ops = np.zeros((ne, ne, dim, dim))
        if count:
            for source, links in enumerate(cistring.gen_linkstr_index(range(ne), count)):
                for p, q, target, sign in links:
                    ops[p, q, target, source] = sign
        return ops

    root, invroot, h, eri, spin_basis, signs, opa, opb, ai, bi = [
        jax.device_put(a, system.device) for a in
        (root, invroot, system.Hcore, system.JK, spin_basis, signs,
         operators(na), operators(nb), ai, bi)]

    def compressed(x):
        C = x[:n * ne].reshape(n, ne)
        Q, R = jnp.linalg.qr(root @ C, mode='reduced')
        basis = invroot @ Q
        hmo = basis.T @ h @ basis
        gmo = eri
        for _ in range(4):
            gmo = jnp.tensordot(gmo, basis, axes=(0, 0))
        cs = x[n * ne:]
        cs = cs / jnp.linalg.norm(cs)
        a, b = compounds(R)
        ci = (a[:, ai] * ((spin_basis @ cs) * signs)[None, :]) @ b[:, bi].T
        # E_pq |CI>, in the orthonormal, moving orbital basis.
        t = jnp.einsum('pqik,kj->pqij', opa, ci) + jnp.einsum('pqjk,ik->pqij', opb, ci)
        hshift = hmo - .5 * jnp.einsum('pqqs->ps', gmo)
        one = jnp.einsum('ij,pq,pqij->', ci, hshift, t)
        two = .5 * jnp.einsum('pqrs,qpij,rsij->', gmo, t, t)
        return one + two, jnp.sum(ci * ci)

    def energy(x):
        C = jax.lax.stop_gradient(x[:n * ne].reshape(n, ne))
        singular_values = jnp.linalg.svd(root @ C, compute_uv=False)
        full_rank = singular_values[-1] > 1e-7 * singular_values[0]
        return jax.lax.cond(full_rank, compressed, fallback, x)

    return jax.jit(energy)
