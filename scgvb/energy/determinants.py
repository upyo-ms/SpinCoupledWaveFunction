"""SCGVB exterior-product evaluation, without electron-permutation tables.

CI amplitudes are derived from the SCGVB orbitals and spin coefficients, not
variational CI parameters. PySCF is used only to apply the Hamiltonian.
"""
import math
import numpy as np
from pyscf import ao2mo, fci
from scgvb.spin.primitive import PrSpinFunc_gen


def determinant_space_size(n, ne, spin):
    na = (ne + spin) // 2
    return math.comb(n, na) * math.comb(n, ne - na)


SIZE_PROFILES = {
    'standard': dict(max_electrons_per_spin=6, max_entries=2_000_000),
    'large14': dict(max_electrons_per_spin=7, max_entries=12_000_000,
                    max_electrons=14, max_orbitals=14),
}


def determinant_size_profile(profile):
    if not isinstance(profile, str) or profile not in SIZE_PROFILES:
        raise ValueError('determinant_size_profile must be standard or large14')
    return SIZE_PROFILES[profile].copy()


def check_determinant_size(n, ne, spin, limit=None, *, profile='standard'):
    bounds = determinant_size_profile(profile)
    limit = bounds['max_entries'] if limit is None else min(limit, bounds['max_entries'])
    na, nb = (ne + spin) // 2, (ne - spin) // 2
    if min(na, nb) < 0 or max(na, nb) > n or (ne - spin) % 2:
        raise ValueError('Invalid electron count/spin for the orbital space')
    # Also bound primitive spin coefficients and polynomial minor evaluation.
    count = determinant_space_size(n, ne, spin)
    from scgvb.spin.dimensions import spin_function_dim
    spin_entries = math.comb(ne, nb) * spin_function_dim(ne, spin + 1, 0)
    minor_entries = max(math.comb(n, k) * math.comb(ne, k)
                        for k in range(max(na, nb) + 1))
    if (max(na, nb) > bounds['max_electrons_per_spin']
            or ne > bounds.get('max_electrons', float('inf'))
            or n > bounds.get('max_orbitals', float('inf'))
            or count > limit or spin_entries > limit
            or minor_entries > limit or n**4 > limit):
        raise ValueError('Determinant evaluation exceeds its size limit: '
                         f"profile={profile}, bounds={bounds}, entry_limit={limit}; "
                         'choose a smaller active space')
    return count


def make_ci_builder(system):
    n, ne = len(system.S), system.Ne
    spin = system.Sm - 1
    check_determinant_size(n, ne, spin, profile=getattr(system, 'determinant_size_profile', 'standard'))
    na, nb = (ne + spin) // 2, (ne - spin) // 2
    _, patterns, spin_basis = PrSpinFunc_gen(na, nb)
    w, v = np.linalg.eigh(system.S)
    if np.any(w <= 0):
        raise ValueError('Orbital overlap must be positive definite')
    root = np.einsum('ik,k,jk->ij', v, np.sqrt(w), v)
    alpha = np.asarray([np.flatnonzero(p == 1) for p in patterns])
    beta = np.asarray([np.flatnonzero(p == -1) for p in patterns])
    signs = np.asarray([(-1.) ** sum(i < j for i in b for j in a)
                        for a, b in zip(alpha, beta)])

    def rows(count):
        strings = fci.cistring.make_strings(range(n), count)
        return np.asarray([[i for i in range(n) if int(s) & (1 << i)]
                           for s in strings], dtype=int).reshape(len(strings), count)

    ra, rb = rows(na), rows(nb)

    def minors(D, r, cols):
        # Bound the temporary (patterns, rows, count, count) determinant blocks.
        result = np.empty((len(cols), len(r)), dtype=D.dtype)
        for start in range(0, len(r), 32):
            rows_block = r[start:start+32]
            result[:, start:start+32] = np.linalg.det(
                D[rows_block[None, :, :, None], cols[:, None, None, :]])
        return result

    def fixed_orbitals(C):
        """Linear spin-to-CI map and its transpose, without spin normalization."""
        C = np.asarray(C)
        if C.shape != (n, ne) or not np.isfinite(C).all():
            raise ValueError('Invalid orbital coefficient shape or values')
        D = np.einsum('ij,ja->ia', root, C)
        a, b = minors(D, ra, alpha), minors(D, rb, beta)
        weights = spin_basis * signs[:, None]

        def forward(cs):
            amplitudes = np.einsum('tk,k->t', weights, cs)
            return np.einsum('ta,tb->ab', amplitudes[:, None] * a, b, optimize=True)

        def transpose(ci):
            amplitudes = np.einsum('ta,ab,tb->t', a, ci, b, optimize=True)
            return np.einsum('tk,t->k', weights, amplitudes)

        return forward, transpose

    def build_ci(C, cs):
        C, cs = np.asarray(C), np.asarray(cs)
        if C.shape != (n, ne) or cs.shape != (system.fs,):
            raise ValueError('Invalid orbital/spin coefficient shape')
        if not np.isfinite(C).all() or not np.isfinite(cs).all() or np.linalg.norm(cs) == 0:
            raise FloatingPointError('Coefficients must be finite with nonzero spin norm')
        # Analytic continuation of the real quadratic form for complex-step.
        cs = cs / np.sqrt(np.dot(cs, cs))
        # Retain the historical evaluation path for the existing optimizer/RDMs.
        D = np.einsum('ij,ja->ia', root, C)
        a, b = minors(D, ra, alpha), minors(D, rb, beta)
        amplitudes = np.einsum('tk,k->t', spin_basis, cs) * signs
        return np.einsum('ta,tb->ab', amplitudes[:, None] * a, b, optimize=True)

    build_ci.fixed_orbitals = fixed_orbitals
    return build_ci


def make_evaluator(system):
    n, ne = len(system.S), system.Ne
    na, nb = (ne + system.Sm - 1) // 2, (ne - system.Sm + 1) // 2
    build_ci = make_ci_builder(system)
    w, v = np.linalg.eigh(system.S)
    invroot = np.einsum('ik,k,jk->ij', v, 1 / np.sqrt(w), v)
    h = np.einsum('ip,ij,jq->pq', invroot, system.Hcore, invroot)
    eri = ao2mo.incore.full(system.JK, invroot, compact=False).reshape((n,) * 4)
    h2 = fci.direct_spin1.absorb_h1e(h, eri, n, (na, nb), .5)

    def energy(C, cs):
        ci = build_ci(C, cs)
        def contract(values):
            return fci.direct_spin1.contract_2e(h2, np.ascontiguousarray(values), n, (na, nb))
        hci = (contract(ci.real) + 1j * contract(ci.imag)
               if np.iscomplexobj(ci) else contract(ci))
        return np.sum(ci * hci), np.sum(ci * ci)

    def spin_matrices(C):
        forward, transpose = build_ci.fixed_orbitals(C)
        hspin = np.empty((system.fs, system.fs))
        sspin = np.empty_like(hspin)
        # One determinant vector at a time: no fs x determinant tensor.
        for k, cs in enumerate(np.eye(system.fs)):
            ci = forward(cs)
            hci = fci.direct_spin1.contract_2e(h2, np.ascontiguousarray(ci), n, (na, nb))
            hspin[:, k], sspin[:, k] = transpose(hci), transpose(ci)
        return (hspin + hspin.T) * .5, (sspin + sspin.T) * .5

    energy.spin_matrices = spin_matrices
    def overlap_matrix(C):
        forward, transpose = build_ci.fixed_orbitals(C)
        metric = np.column_stack([transpose(forward(cs)) for cs in np.eye(system.fs)])
        return .5 * (metric + metric.T)

    def projected_matrices(C, basis, reference):
        forward, _ = build_ci.fixed_orbitals(C)
        states = [forward(cs) for cs in basis.T]
        shifted = [fci.direct_spin1.contract_2e(h2, np.ascontiguousarray(ci), n, (na, nb))
                   - reference * ci for ci in states]
        metric = np.array([[np.sum(a*b) for b in states] for a in states])
        projected = np.array([[np.sum(a*b) for b in shifted] for a in states])
        return .5 * (projected+projected.T), .5 * (metric+metric.T)

    def normalized_ci(C, cs):
        ci = build_ci(C, cs)
        return ci/np.sqrt(np.sum(ci*ci))

    def orbital_metric(C, cs):
        if C.size > 36:
            raise ValueError('Large wavefunction orbital metrics require the JAX determinant backend')
        derivatives = []
        for index in range(C.size):
            trial = np.asarray(C, dtype=complex).copy()
            trial.flat[index] += 1e-20j
            derivatives.append(normalized_ci(trial, cs).imag.ravel()/1e-20)
        jacobian = np.array(derivatives)
        return np.einsum('ik,jk->ij', jacobian, jacobian)

    energy.overlap_matrix = overlap_matrix
    energy.projected_matrices = projected_matrices
    def orbital_states(C, cs, index, basis):
        # Every SCGVB orbital occurs exactly once in each exterior product.
        # This map is linear in this one column, even though the same spatial
        # orbital occurs in different spin assignments across determinants.
        states = []
        for column in basis.T:
            trial = np.asarray(C).copy()
            trial[:, index] = column
            states.append(build_ci(trial, cs))
        return np.asarray(states)

    def orbital_overlap(C, cs, index):
        states = orbital_states(C, cs, index, np.eye(n)).reshape(n, -1)
        return states @ states.T

    def orbital_projected(C, cs, index, basis, reference):
        states = orbital_states(C, cs, index, basis)
        shifted = np.asarray([fci.direct_spin1.contract_2e(
            h2, np.ascontiguousarray(ci), n, (na, nb)) - reference*ci
            if np.any(ci) else np.zeros_like(ci) for ci in states])
        flat, hflat = states.reshape(len(states), -1), shifted.reshape(len(states), -1)
        metric, h = flat @ flat.T, flat @ hflat.T
        return .5*(h+h.T), .5*(metric+metric.T)

    energy.orbital_overlap = orbital_overlap
    energy.orbital_projected = orbital_projected
    energy.orbital_metric = orbital_metric
    energy.wavefunction_overlap = lambda C, cs, other_C, other_cs: float(np.sum(
        normalized_ci(C, cs)*normalized_ci(other_C, other_cs)))
    return energy
