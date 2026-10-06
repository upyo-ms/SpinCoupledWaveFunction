"""Relax doubly occupied inactive orbitals with the SCGVB active-state solver.

PySCF supplies orbital rotations and RDM contractions, never an FCI optimization.
The active CI vector is derived from C and Cs and is not a variational parameter.
"""
from copy import copy, deepcopy

import numpy as np
from scipy.linalg import null_space
from pyscf import ao2mo, fci, mcscf
from pyscf.mcscf import mc2step

from DeterminantEnergy import make_ci_builder, check_determinant_size


def complete_orbitals(space, overlap):
    """Keep the given inactive/active orbitals and add their S-orthogonal complement."""
    w, v = np.linalg.eigh(overlap)
    if np.any(w <= 0):
        raise ValueError('Core optimization requires positive-definite AO overlap')
    root = np.einsum('ik,k,jk->ij', v, np.sqrt(w), v)
    invroot = np.einsum('ik,k,jk->ij', v, 1 / np.sqrt(w), v)
    occupied = np.column_stack((space.inactive, space.basis))
    transformed = np.einsum('ij,jk->ik', root, occupied)
    # Avoid spurious floating-point status warnings from Accelerate matmul
    # on macOS; a direct contraction also checks the same Gram matrix.
    gram = np.einsum('pi,pj->ij', transformed, transformed, optimize=False)
    if not np.allclose(gram, np.eye(occupied.shape[1]), atol=1e-9, rtol=0.):
        raise ValueError('Inactive and active bases must be jointly S-orthonormal')
    external = np.einsum('ij,jk->ik', invroot, null_space(transformed.T))
    return np.column_stack((occupied, external))


class SCGVBSolver(fci.direct_spin1.FCISolver):
    """Adapter for two-step MCSCF: solve only the SCGVB C/Cs parameters."""
    def __init__(self, system, C, Cs, optimizer_options):
        super().__init__(system.mol)
        self.template = system
        self.C, self.Cs = C.copy(), Cs.copy()
        self.optimizer_options = optimizer_options
        self.build_ci = make_ci_builder(system)
        self.history = []
        self.converged = False

    def kernel(self, h1, eri, norb, nelec, ci0=None, ecore=0., **kwargs):
        s = copy(self.template)
        s.Hcore = np.asarray(h1)
        s.JK = ao2mo.restore(1, eri, norb)
        s.energy_nuc = s.inactive_energy = 0.
        s._determinant_evaluator = None
        if hasattr(s, '_initialize_jax'):
            s._initialize_jax(self.template.device.platform, self.template.energy_evaluation)
        if hasattr(s, 'diagnostics'):
            s.diagnostics.ecore = ecore
        energy, C, Cs = s.optimize(self.C.copy(), self.Cs.copy(), **self.optimizer_options)
        gradient = float(np.max(np.abs(s.rayleigh_gradient(np.r_[C.ravel(), Cs]))))
        norm = s.calcEnergy(C, Cs)[1]
        if (not np.isfinite(gradient) or not np.isfinite(norm) or gradient > 1e-6
                or abs(norm - 1.) >= self.optimizer_options.get('epsilon', 1e-8)):
            raise RuntimeError('SCGVB inner solve did not satisfy physical stationarity and normalization '
                               f'(gradient={gradient:.6g}, norm_error={norm-1:.6g})')
        ci = self.build_ci(C, Cs)
        ci_norm = np.linalg.norm(ci)
        if not np.isfinite(ci_norm) or ci_norm <= 0:
            raise RuntimeError('Invalid SCGVB state for orbital density matrices')
        ci /= ci_norm
        self.C, self.Cs = C, Cs
        self.optimization_info = getattr(s, 'optimization_info', None)
        self.history.append({'energy_hartree': float(energy + ecore),
                             'active_gradient_max_abs': gradient})
        self.converged = True
        return energy + ecore, ci


def optimize_core(system, C, Cs, *, max_iterations=50, gradient_tolerance=1e-6,
                  optimizer_options=None, verbose=False):
    """Return (energy, C, Cs, rebuilt system, relaxed space, diagnostics).

    The active electron count and SCGVB spin space stay fixed. Rotations between
    inactive, active and external spaces are allowed; inactive occupations stay 2.
    """
    space = system.active_space
    if space is None or space.inactive.shape[1] == 0:
        raise ValueError('--optimize-core requires a nonempty inactive/core space')
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError('core_max_iterations must be a positive integer')
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('core_gradient_tolerance must be finite and positive')
    nact, ncore = space.basis.shape[1], space.inactive.shape[1]
    check_determinant_size(nact, space.electrons, system.mol.spin, profile=getattr(system, 'determinant_size_profile', 'standard'))
    mo = complete_orbitals(space, system.mol.intor('int1e_ovlp'))
    na = (space.electrons + system.mol.spin) // 2
    mc = mcscf.CASSCF(system.mol, nact, (na, space.electrons - na), ncore=ncore)
    # AO F targets occupy the leading inactive columns and remain fixed.
    n_frozen = int(space.metadata.get('frozen_inactive_orbitals', 0))
    mc.frozen = list(range(n_frozen)) if n_frozen else None
    mc.mo_coeff = mo
    mc.verbose = 4 if verbose else 0
    mc.canonicalization = False  # Do not change the basis of the returned C/Cs.
    mc.internal_rotation = False
    mc.max_cycle_macro = max_iterations
    mc.conv_tol = 1e-9
    # PySCF's packed orbital gradient is half dE/d(kappa) for its
    # antisymmetric rotation coordinates. Report/check the full derivative.
    mc.conv_tol_grad = gradient_tolerance / 2
    # Resolve the small final rotations instead of stopping the AH solve at
    # its default tolerance before the requested orbital gradient is met.
    mc.ah_conv_tol = min(mc.ah_conv_tol, gradient_tolerance**2 * 1e-3)
    mc.ah_lindep = min(mc.ah_lindep, gradient_tolerance**2 * 1e-4)
    solver = SCGVBSolver(system, C, Cs, optimizer_options or {})
    mc.fcisolver = solver
    diagnostics = getattr(system, 'diagnostics', None)
    def diagnostic_callback(env):
        if diagnostics is not None and int(env['imacro']) != diagnostics.macro:
            diagnostics.basis = env['mo'][:, ncore:ncore+nact].copy()
            diagnostics.inactive = env['mo'][:, :ncore].copy()
            diagnostics.macro = int(env['imacro'])
            diagnostics.snapshot(system, 'orbital_update',
                                 orbital_gradient_norm=float(2 * env['norm_gorb']),
                                 rotation_norm=float(env['norm_t']))
    converged, energy, _, ci, mo, _ = mc2step.kernel(
        mc, mo, tol=mc.conv_tol, conv_tol_grad=mc.conv_tol_grad, dump_chk=False,
        callback=diagnostic_callback if diagnostics is not None else None)
    dm1, dm2 = solver.make_rdm12(ci, nact, mc.nelecas)
    gradient = mc.gen_g_hop(mo, 1, dm1, dm2, mc.ao2mo(mo))[0]
    gradient_norm = float(2 * np.linalg.norm(gradient))
    if not converged or not np.isfinite(gradient_norm) or gradient_norm > gradient_tolerance:
        if diagnostics is not None:
            diagnostics.snapshot(system, 'core_failure', orbital_gradient_norm=gradient_norm,
                                 error='Core orbital optimization did not converge')
        raise RuntimeError(f'Core orbital optimization did not converge in {max_iterations} '
                           f'macro iterations (orbital gradient norm={gradient_norm:g})')
    if energy > solver.history[0]['energy_hartree'] + 1e-8:
        raise RuntimeError('Relaxed-core energy exceeds the initial fixed-core energy')
    if n_frozen and not np.allclose(mo[:, :n_frozen], space.inactive[:, :n_frozen],
                                    atol=1e-10, rtol=0.):
        raise RuntimeError('Frozen AO-selected orbitals changed during core optimization')
    relaxed = deepcopy(space)
    relaxed.inactive = mo[:, :ncore].copy()
    relaxed.basis = mo[:, ncore:ncore+nact].copy()
    relaxed.metadata.update(inactive_orbitals_optimized=True,
                            orbital_optimization='two_step_scgvb',
                            active_subspace_optimized=True)
    options = dict(compute_fci=False, energy_evaluation=system.energy_evaluation, active_space=relaxed, determinant_size_profile=getattr(system, 'determinant_size_profile', 'standard'))
    if hasattr(system, 'device'):
        options['device'] = system.device.platform
    final = type(system)(system.mol, **options)
    if diagnostics is not None:
        final.diagnostics = diagnostics
    if getattr(solver, 'optimization_info', None) is not None:
        final.optimization_info = solver.optimization_info
    e, norm = final.calcEnergy(solver.C, solver.Cs)
    total = e / norm + final.energy_nuc + final.inactive_energy
    if abs(total - energy) > 1e-8:
        raise RuntimeError('Relaxed-core energy failed independent integral reconstruction')
    info = {'converged': True, 'orbital_gradient_norm': gradient_norm,
            'orbital_gradient_tolerance': gradient_tolerance,
            'max_macro_iterations': max_iterations,
            'energy_tolerance_hartree': mc.conv_tol,
            'initial_fixed_core_energy_hartree': solver.history[0]['energy_hartree'],
            'macro_iterations': len(solver.history) - 1, 'history': solver.history,
            'inactive_occupations': [2.] * ncore,
            'frozen_inactive_orbitals': n_frozen,
            'initial_inactive_orbital_coefficients': space.inactive.tolist(),
            'orbital_optimizer_device': 'cpu'}
    info['active_optimization_method'] = solver.optimizer_options.get('optimization_method', 'simultaneous')
    return total, solver.C, solver.Cs, final, relaxed, info
