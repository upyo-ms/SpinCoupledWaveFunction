"""Experimental SCGVB internal response for inactive/active/external rotations.

Only C and Cs are varied. Polynomial exterior products supply exact first and
second amplitude derivatives, including manifold curvature. PySCF supplies Hc
and orbital integral contractions, never an FCI optimization. This deliberately
bounded CPU implementation is shared by second_order internal corrections and
the outer response when inactive orbitals are optimized.
"""
from itertools import combinations
from math import comb
import time

import numpy as np
from pyscf import fci
from pyscf.lib import logger

from scgvb.energy.determinants import check_determinant_size
from scgvb.spin.dimensions import spin_function_dim
from scgvb.spin.primitive import PrSpinFunc_gen


MAX_JET_ENTRIES = 2_000_000
METRIC_CUTOFF = 1e-10
CURVATURE_FLOOR = 1e-6


def check_response_size(n, ne, spin):
    """Bound derivative tensors independently of the energy evaluator's limit."""
    check_determinant_size(n, ne, spin)
    na, nb = (ne + spin) // 2, (ne - spin) // 2
    parameters = n * ne + spin_function_dim(ne, spin + 1, 0)
    blocks = [comb(n, na) * comb(n, nb)]
    blocks.extend(comb(n, k) * comb(ne, k) for k in range(max(na, nb) + 1))
    entries = max(blocks) * (1 + parameters + parameters**2)
    if entries > MAX_JET_ENTRIES:
        raise ValueError('second_order derivative tensors exceed the experimental size limit '
                         f'({entries} > {MAX_JET_ENTRIES} entries); use a smaller active space')
    return dict(parameters=parameters, largest_derivative_block_entries=entries)


def _product(a, b):
    """Product rule for value, gradient and Hessian jets (trailing axes)."""
    av, aj, ah = a
    bv, bj, bh = b
    return (av * bv,
            aj * bv[..., None] + av[..., None] * bj,
            ah * bv[..., None, None] + av[..., None, None] * bh
            + aj[..., :, None] * bj[..., None, :]
            + bj[..., :, None] * aj[..., None, :])


class AmplitudeDerivatives:
    """Polynomial amplitude derivatives, valid even at zero individual minors."""
    def __init__(self, system):
        n, ne, spin = len(system.S), system.Ne, system.Sm - 1
        self.size = check_response_size(n, ne, spin)
        self.n, self.ne, self.p = n, ne, self.size['parameters']
        self.na, self.nb = (ne + spin) // 2, (ne - spin) // 2
        _, patterns, basis = PrSpinFunc_gen(self.na, self.nb)
        self.patterns, self.spin_basis = patterns, basis
        w, v = np.linalg.eigh(system.S)
        if w[0] <= 0:
            raise ValueError('second_order requires positive-definite orbital overlap')
        self.root = np.einsum('ik,jk->ij', v*np.sqrt(w), v, optimize=False)
        self.stages, maps = [], {0: {(): 0}}
        previous_rows = {(): 0}
        for k in range(1, max(self.na, self.nb) + 1):
            strings = fci.cistring.make_strings(range(n), k)
            rows = [tuple(i for i in range(n) if int(s) & (1 << i)) for s in strings]
            columns = list(combinations(range(ne), k))
            prefix = np.array([previous_rows[r[:-1]] for r in rows])
            remove = np.array([[maps[k-1][c[:j] + c[j+1:]] for c in columns]
                               for j in range(k)])
            self.stages.append((prefix, np.array([r[-1] for r in rows]),
                                remove, np.array(columns).T))
            previous_rows = {r: i for i, r in enumerate(rows)}
            maps[k] = {c: i for i, c in enumerate(columns)}
        self.alpha = [maps[self.na][tuple(np.flatnonzero(p == 1))] for p in patterns]
        self.beta = [maps[self.nb][tuple(np.flatnonzero(p == -1))] for p in patterns]
        self.signs = np.array([(-1.) ** sum(i < j for i in np.flatnonzero(p == -1)
                                          for j in np.flatnonzero(p == 1)) for p in patterns])

    def __call__(self, C, Cs):
        n, ne, p = self.n, self.ne, self.p
        C, Cs = np.asarray(C), np.asarray(Cs)
        if C.shape != (n, ne) or Cs.shape != (p - n*ne,):
            raise ValueError('Invalid SCGVB coefficient shape for second_order')
        dtype = np.result_type(C, Cs, float)
        orbital = np.einsum('ij,jk->ik', self.root, C, optimize=False)
        derivative = np.zeros((n, ne, p), dtype=dtype)
        for j in range(ne):
            derivative[:, j, np.arange(n)*ne+j] = self.root
        previous = (np.ones((1, 1), dtype=dtype), np.zeros((1, 1, p), dtype=dtype),
                    np.zeros((1, 1, p, p), dtype=dtype))
        compounds = {0: previous}
        for k, (prefix, last, remove, columns) in enumerate(self.stages, 1):
            shape = (len(prefix), columns.shape[1])
            result = [np.zeros(shape + tail, dtype=dtype) for tail in ((), (p,), (p, p))]
            for j in range(k):
                ij = (last[:, None], columns[j][None, :])
                dv = orbital[ij]
                dj = derivative[ij]
                minor = tuple(a[prefix[:, None], remove[j][None, :]] for a in previous)
                # The orbital is linear, so its second derivative vanishes.
                term = (minor[0]*dv,
                        minor[1]*dv[..., None] + minor[0][..., None]*dj,
                        minor[2]*dv[..., None, None]
                        + minor[1][..., :, None]*dj[..., None, :]
                        + dj[..., :, None]*minor[1][..., None, :])
                sign = (-1.)**(k-1+j)
                for target, source in zip(result, term):
                    target += sign*source
            previous = tuple(result)
            compounds[k] = previous
        shape = (comb(n, self.na), comb(n, self.nb))
        total = [np.zeros(shape + tail, dtype=dtype) for tail in ((), (p,), (p, p))]
        absolute_sum = 0.
        for t, (ai, bi, sign) in enumerate(zip(self.alpha, self.beta, self.signs)):
            a = tuple(v[:, ai][:, None] for v in compounds[self.na])
            b = tuple(v[:, bi][None, :] for v in compounds[self.nb])
            weight = np.asarray(sign * (self.spin_basis[t] @ Cs))
            gradient = np.r_[np.zeros(n*ne), sign*self.spin_basis[t]].astype(dtype)
            term = _product(_product(a, b), (weight, gradient, np.zeros((p, p), dtype=dtype)))
            absolute_sum += float(np.linalg.norm(term[0]))
            for target, source in zip(total, term):
                target += source
        norm = np.linalg.norm(total[0])
        cancellation = absolute_sum / max(float(norm), np.finfo(float).tiny)
        return (*total, cancellation)


def internal_curvature(amplitudes, C, Cs, apply_h):
    """Physical Rayleigh Hessian, including second amplitude derivatives."""
    a, jacobian, second, cancellation = amplitudes(C, Cs)
    shape, p = a.shape, amplitudes.p
    a, j, d2 = a.ravel(), jacobian.reshape(-1, p), second.reshape(-1, p, p)
    norm = float(a @ a)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny or cancellation > 1e8:
        raise FloatingPointError('SCGVB second-order wavefunction is too strongly cancelling for FP64 derivatives')
    ha = np.asarray(apply_h(a.reshape(shape))).ravel()
    energy = float(a @ ha / norm)
    residual = ha - energy*a
    gradient = 2*np.einsum('di,d->i', j, residual, optimize=False)/norm
    hj = np.column_stack([np.asarray(apply_h(j[:, i].reshape(shape))).ravel() for i in range(p)])
    hessian = 2*(np.einsum('di,dj->ij', j, hj-energy*j, optimize=False)
                 + np.einsum('dij,d->ij', d2, residual))/norm
    norm_gradient = 2*np.einsum('di,d->i', j, a, optimize=False)
    hessian -= (np.outer(gradient, norm_gradient) + np.outer(norm_gradient, gradient))/norm
    normalized_jacobian = (j - np.outer(a, norm_gradient/(2*norm)))/np.sqrt(norm)
    u, singular, vt = np.linalg.svd(normalized_jacobian, full_matrices=False)
    keep = singular > max(singular[0]*np.sqrt(METRIC_CUTOFF), 1e-12)
    tangent = vt[keep].T / singular[keep]
    ht = np.einsum('ij,jb->ib', hessian, tangent, optimize=False)
    reduced = np.einsum('ia,ib->ab', tangent, ht, optimize=False)
    if not np.isfinite(reduced).all():
        raise FloatingPointError('Nonfinite SCGVB response curvature')
    return dict(energy=energy, gradient=gradient, hessian=.5*(hessian+hessian.T),
                tangent=tangent, reduced_hessian=.5*(reduced+reduced.T),
                ci=(a/np.sqrt(norm)).reshape(shape),
                states=u[:, keep], cancellation_ratio=cancellation,
                metric_singular_values=singular)


def _one_electron_mo(mc, mo):
    # Two contractions avoid both a four-index temporary and spurious macOS
    # Accelerate floating-point status warnings from chained small matmuls.
    intermediate = np.einsum('ip,ij->pj', mo, mc.get_hcore(), optimize=False)
    return np.einsum('pi,iq->pq', intermediate, mo, optimize=False)


def orbital_gradient_response(mc, mo, dm1, dm2, eris, *, hcore_mo=None):
    """Full dE/dkappa derivative from differential active RDMs.

    This is the RDM-linear part of mc1step.gen_g_hop's gradient. Inactive-only
    constants are omitted analytically, avoiding subtraction of core energies.
    PySCF's packed gradient is half the full physical derivative.
    """
    nc, na = mc.ncore, mc.ncas
    no, nm = nc+na, mo.shape[1]
    h = _one_electron_mo(mc, mo) if hcore_mo is None else hcore_mo
    g = np.zeros((nm, nm))
    for i in range(nm):
        jbuf, kbuf = eris.ppaa[i], eris.papa[i]
        vhf = np.einsum('quv,uv->q', jbuf, dm1) - .5*np.einsum('uqv,uv->q', kbuf, dm1)
        g[i, :nc] = 2*vhf[:nc]
        jtmp = np.einsum('ix,xy->iy', jbuf.reshape(nm, -1), dm2.reshape(na*na, -1),
                         optimize=False).reshape(nm, na, na)
        g[i, nc:no] = np.einsum('uuv->v', jtmp[nc:no])
    g[:, nc:no] += np.einsum('ip,pq->iq', h[:, nc:no] + eris.vhf_c[:, nc:no], dm1, optimize=False)
    return 2*mc.pack_uniq_var(g-g.T)


def install_core_response(mc, solver):
    """Add -B.T A^-1 B to orbital curvature; keep the existing AH safeguards."""
    amplitudes = AmplitudeDerivatives(solver.template)
    original = mc.gen_g_hop
    history = []

    def gen_g_hop(mo, u, dm1, dm2, eris):
        gradient, update, hop, diagonal = original(mo, u, dm1, dm2, eris)
        started = time.perf_counter()
        nc, na = mc.ncore, mc.ncas
        hmo = _one_electron_mo(mc, mo)
        h = hmo
        h = h[nc:nc+na, nc:nc+na] + eris.vhf_c[nc:nc+na, nc:nc+na]
        eri = np.asarray(eris.ppaa[nc:nc+na])[:, nc:nc+na]
        h2 = fci.direct_spin1.absorb_h1e(h, eri, na, mc.nelecas, .5)

        def apply_h(ci):
            return fci.direct_spin1.contract_2e(h2, np.ascontiguousarray(ci), na, mc.nelecas)

        # A change of coefficient scale leaves the normalized SCGVB state intact.
        C = solver.C / np.linalg.norm(solver.C, axis=0)
        Cs = solver.Cs / np.linalg.norm(solver.Cs)
        data = internal_curvature(amplitudes, C, Cs, apply_h)
        ci, states = data['ci'], data['states']
        actual = solver.build_ci(solver.C, solver.Cs)
        actual /= np.linalg.norm(actual)
        if abs(np.sum(actual*ci)) < 1-1e-10:
            raise FloatingPointError('core_response amplitude evaluation disagrees with the active solver')
        # Compare physical RDMs too: response must be based at the current state.
        check1, check2 = solver.make_rdm12(ci, na, mc.nelecas)
        if not (np.allclose(dm1, check1, atol=1e-8, rtol=1e-8)
                and np.allclose(dm2, check2, atol=1e-8, rtol=1e-8)):
            raise RuntimeError('core_response requires the current SCGVB stationary state')
        mixed = np.empty((states.shape[1], gradient.size))
        for i in range(states.shape[1]):
            state = states[:, i].reshape(ci.shape)
            plus = solver.make_rdm12(ci+state, na, mc.nelecas)
            minus = solver.make_rdm12(ci-state, na, mc.nelecas)
            d1, d2 = ((a-b)*.5 for a, b in zip(plus, minus))
            mixed[i] = orbital_gradient_response(mc, mo, d1, d2, eris, hcore_mo=hmo)
        values, vectors = np.linalg.eigh(data['reduced_hessian'])
        shift = max(0., CURVATURE_FLOOR - float(values[0])) if len(values) else 0.
        rotated = np.einsum('ji,jk->ik', vectors, mixed, optimize=False)/(values+shift)[:, None]
        inverse_mixed = np.einsum('ij,jk->ik', vectors, rotated, optimize=False)
        correction_diagonal = .5*np.sum(mixed*inverse_mixed, axis=0)
        record = dict(internal_tangent_rank=len(values), parameters=amplitudes.p,
                      metric_relative_cutoff=METRIC_CUTOFF,
                      internal_curvature_min=None if not len(values) else float(values[0]),
                      curvature_shift_hartree=shift, curvature_floor_hartree=CURVATURE_FLOOR,
                      mixed_derivative_norm=float(np.linalg.norm(mixed)),
                      correction_diagonal_max_abs=float(np.max(np.abs(correction_diagonal), initial=0.)),
                      cancellation_ratio=data['cancellation_ratio'],
                      internal_gradient_max_abs=float(np.max(np.abs(data['gradient']), initial=0.)),
                      internal_tangent_gradient_norm=float(np.linalg.norm(np.einsum(
                          'ia,i->a', data['tangent'], data['gradient'], optimize=False))),
                      elapsed_seconds=time.perf_counter()-started)
        history.append(record)
        logger.info(mc, 'SCGVB core response: tangent rank %d, |B|=%.6g, shift=%.6g',
                    len(values), record['mixed_derivative_norm'], shift)

        def relaxed_hop(direction):
            # PySCF uses half of the physical orbital Hessian, just as its gradient.
            response = np.einsum('ij,j->i', inverse_mixed, direction, optimize=False)
            return hop(direction) - .5*np.einsum('ij,i->j', mixed, response, optimize=False)

        return gradient, update, relaxed_hop, diagonal-correction_diagonal

    mc.gen_g_hop = gen_g_hop
    return history
