"""Relax doubly occupied inactive orbitals with the SCGVB active-state solver.

PySCF supplies orbital rotations and RDM contractions, never an FCI optimization.
The active CI vector is derived from C and Cs and is not a variational parameter.
"""
from copy import copy, deepcopy

import numpy as np
from scipy.linalg import null_space
from pyscf import ao2mo, fci, mcscf
from pyscf.mcscf import mc2step
from pyscf.lib import logger

from scgvb.energy.determinants import make_ci_builder, check_determinant_size
from scgvb.optimization.convergence import ElectronicConvergenceError, restart_candidate


def _unit_ah_seed(seed):
    """Normalize a Davidson basis vector without squaring its original scale."""
    if seed is None:
        return None
    vector = np.asarray(seed, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        return None
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return None
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _normalize_ah_seeds(mc):
    """Condition this CASSCF instance's AH basis, leaving physical steps intact.

    PySCF reuses the previous correction as a Davidson seed. Its squared norm
    can fall below ah_lindep near convergence, discarding a useful direction
    and producing an identity orbital update. Only the basis vector is scaled;
    Davidson still determines the actual rotation and applies its usual limits.
    """
    rotate = mc.rotate_orb_cc
    history = []

    def rotate_with_unit_seed(mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                              conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
        seed = _unit_ah_seed(x0_guess)
        record = dict(source='previous_correction', input_max_abs=None)
        if seed is None:
            gradient = mc.gen_g_hop(mo, 1, fcasdm1(), fcasdm2(), eris)[0]
            seed = _unit_ah_seed(gradient)
            record['source'] = 'current_gradient'
            record['fallback'] = 'missing' if x0_guess is None else 'zero_or_nonfinite'
            original = np.asarray(gradient)
            if seed is None:
                # A zero gradient is handled by PySCF's existing early exit.
                if not np.isfinite(original).all():
                    raise RuntimeError('Nonfinite orbital gradient in AH initialization')
                seed = np.zeros_like(original)
        else:
            original = np.asarray(x0_guess)
        record['input_max_abs'] = float(np.max(np.abs(original))) if original.size else 0.
        record['seed_norm'] = float(np.linalg.norm(seed))
        history.append(record)
        logger.info(mc, 'AH seed: %s, input max|x|=%.6g, normalized norm=%.6g',
                    record['source'], record['input_max_abs'], record['seed_norm'])
        return rotate(mo, fcivec, fcasdm1, fcasdm2, eris, seed,
                      conv_tol_grad, max_stepsize, verbose)

    mc.rotate_orb_cc = rotate_with_unit_seed
    return history


def _gradient_aware_step_control(mc):
    """Avoid interpreting an energy plateau as repeated rejected AH steps.

    Keep PySCF's reduction for a resolved energy increase and its geometric
    recovery after a resolved decrease. Within the energy tolerance, recover
    the cap if the orbital gradient is still unconverged; otherwise hold it.
    The AH solver still chooses the rotation inside this cap and its own trust
    checks. Neither the physical gradient definition nor its tolerance changes.
    """
    history = []

    def schedule(env):
        maximum = float(mc.max_stepsize)
        previous = float(env.get('max_stepsize', maximum))
        if not np.isfinite([maximum, previous]).all() or min(maximum, previous) <= 0:
            raise RuntimeError('Invalid AH orbital step limit')
        previous = min(previous, maximum)
        delta = float(env['de'])
        packed_gradient = env.get('norm_gorb')
        if not np.isfinite(delta) or (packed_gradient is not None
                                     and not np.isfinite(packed_gradient)):
            raise RuntimeError('Nonfinite energy change or orbital gradient in AH step control')
        if packed_gradient is None:
            # The first macro step has no measured energy change or gradient.
            limit, action = maximum, 'initial'
        elif delta > mc.conv_tol:
            limit, action = previous * .3, 'reduce_energy_increase'
        elif delta < -mc.conv_tol:
            limit, action = np.sqrt(maximum) * np.sqrt(previous), 'recover_energy_decrease'
        elif packed_gradient > mc.conv_tol_grad:
            limit, action = np.sqrt(maximum) * np.sqrt(previous), 'recover_unconverged_plateau'
        else:
            limit, action = previous, 'hold_converged_plateau'
        limit = min(float(limit), maximum)
        record = dict(macro_iteration=int(env.get('imacro', 0)), action=action,
                      energy_change_hartree=delta,
                      orbital_gradient_norm=None if packed_gradient is None else float(2 * packed_gradient),
                      previous_max_stepsize=previous, max_stepsize=limit)
        history.append(record)
        mc._max_stepsize = limit
        logger.info(mc, 'AH step control: %s, dE=%.6g, max_stepsize=%.6g', action, delta, limit)
        return limit

    mc.max_stepsize_scheduler = schedule
    return history


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
        self.second_order_history = []
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
        if self.optimizer_options.get('second_order', False):
            self.second_order_history.append(deepcopy(s.second_order_info))
        gradient = float(np.max(np.abs(s.rayleigh_gradient(np.r_[C.ravel(), Cs]))))
        norm = s.calcEnergy(C, Cs)[1]
        if (not np.isfinite(gradient) or not np.isfinite(norm) or gradient > 1e-6
                or abs(norm - 1.) >= self.optimizer_options.get('epsilon', 1e-8)):
            error = ElectronicConvergenceError('SCGVB inner solve did not satisfy physical stationarity and normalization '
                                              f'(gradient={gradient:.6g}, norm_error={norm-1:.6g})')
            error.retained_optimization_states = dict(last_iterate=dict(x=np.r_[C.ravel(), Cs]))
            raise error
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
                  optimizer_options=None, verbose=False, core_response=None, second_order=None,
                  capture_failure_state=False):
    """Return (energy, C, Cs, rebuilt system, relaxed space, diagnostics).

    The active electron count and SCGVB spin space stay fixed. Rotations between
    inactive, active and external spaces are allowed; inactive occupations stay 2.
    """
    space = system.active_space
    from scgvb.optimization.second_order import resolve_option
    optimizer_options = dict(optimizer_options or {})
    if second_order is None and core_response is None:
        second_order = optimizer_options.get('second_order')
    second_order = resolve_option(second_order, core_response)
    core_response = second_order
    if 'second_order' in optimizer_options and optimizer_options['second_order'] != second_order:
        raise ValueError('Inner and outer second_order options disagree')
    optimizer_options['second_order'] = second_order
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
    ah_seed_history = _normalize_ah_seeds(mc)
    ah_step_history = _gradient_aware_step_control(mc)
    solver = SCGVBSolver(system, C, Cs, optimizer_options or {})
    mc.fcisolver = solver
    response_history = None
    if core_response:
        from scgvb.optimization.core_response import install_core_response
        response_history = install_core_response(mc, solver)
    diagnostics = getattr(system, 'diagnostics', None)
    current_mo = mo
    def attach_restart(error, orbitals, source):
        # The callback sees each rotated MO basis before the next inner solve.
        # Rebuild AO integrals in that basis before interpreting retained C/Cs.
        try:
            restart_space = deepcopy(space)
            restart_space.inactive = orbitals[:, :ncore].copy()
            restart_space.basis = orbitals[:, ncore:ncore+nact].copy()
            options = dict(compute_fci=False, energy_evaluation=system.energy_evaluation,
                           active_space=restart_space,
                           determinant_size_profile=getattr(system, 'determinant_size_profile', 'standard'))
            if hasattr(system, 'device'):
                options['device'] = system.device.platform
            rebuilt = type(system)(system.mol, **options)
            if diagnostics is not None:
                rebuilt.diagnostics = diagnostics
            c, cs, seed = restart_candidate(rebuilt, solver.C, solver.Cs, error)
            error.core_restart_state = (rebuilt, c, cs, source + ':' + seed)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as unavailable:
            error.core_restart_unavailable = str(unavailable)

    def diagnostic_callback(env):
        nonlocal current_mo
        if capture_failure_state:
            current_mo = env['mo'].copy()
        if diagnostics is not None and int(env['imacro']) != diagnostics.macro:
            diagnostics.basis = env['mo'][:, ncore:ncore+nact].copy()
            diagnostics.inactive = env['mo'][:, :ncore].copy()
            diagnostics.macro = int(env['imacro'])
            diagnostics.snapshot(system, 'orbital_update',
                                 orbital_gradient_norm=float(2 * env['norm_gorb']),
                                 rotation_norm=float(env['norm_t']))
    try:
        converged, energy, _, ci, mo, _ = mc2step.kernel(
            mc, mo, tol=mc.conv_tol, conv_tol_grad=mc.conv_tol_grad, dump_chk=False,
            callback=diagnostic_callback if diagnostics is not None or capture_failure_state else None)
    except Exception as exc:
        if core_response:
            exc.core_response_history = deepcopy(response_history)
        if capture_failure_state and isinstance(exc, ElectronicConvergenceError):
            attach_restart(exc, current_mo, 'core_inner_failure')
        raise
    dm1, dm2 = solver.make_rdm12(ci, nact, mc.nelecas)
    gradient = mc.gen_g_hop(mo, 1, dm1, dm2, mc.ao2mo(mo))[0]
    gradient_norm = float(2 * np.linalg.norm(gradient))
    if not converged or not np.isfinite(gradient_norm) or gradient_norm > gradient_tolerance:
        if diagnostics is not None:
            diagnostics.snapshot(system, 'core_failure', orbital_gradient_norm=gradient_norm,
                                 error='Core orbital optimization did not converge')
        error = ElectronicConvergenceError(f'Core orbital optimization did not converge in {max_iterations} '
                                          f'macro iterations (orbital gradient norm={gradient_norm:g})')
        if capture_failure_state:
            attach_restart(error, mo, 'core_outer_failure')
        raise error
    if energy > solver.history[0]['energy_hartree'] + 1e-8:
        raise RuntimeError('Relaxed-core energy exceeds the initial fixed-core energy')
    if n_frozen and not np.allclose(mo[:, :n_frozen], space.inactive[:, :n_frozen],
                                    atol=1e-10, rtol=0.):
        raise RuntimeError('Frozen AO-selected orbitals changed during core optimization')
    relaxed = deepcopy(space)
    relaxed.inactive = mo[:, :ncore].copy()
    relaxed.basis = mo[:, ncore:ncore+nact].copy()
    relaxed.metadata.update(inactive_orbitals_optimized=True,
                            orbital_optimization='two_step_scgvb_response' if core_response else 'two_step_scgvb',
                            active_subspace_optimized=True)
    options = dict(compute_fci=False, energy_evaluation=system.energy_evaluation, active_space=relaxed, determinant_size_profile=getattr(system, 'determinant_size_profile', 'standard'))
    if hasattr(system, 'device'):
        options['device'] = system.device.platform
    final = type(system)(system.mol, **options)
    if diagnostics is not None:
        final.diagnostics = diagnostics
    if getattr(solver, 'optimization_info', None) is not None:
        final.optimization_info = solver.optimization_info
    if second_order:
        final.second_order_info = dict(enabled=True, method='analytic_safeguarded_newton',
            inner_solves=len(solver.second_order_history),
            accepted_steps=sum(h['accepted_steps'] for h in solver.second_order_history),
            history=solver.second_order_history)
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
            'ah_seed_normalization': True, 'ah_seed_history': ah_seed_history,
            'ah_step_control': 'gradient_aware', 'ah_step_history': ah_step_history,
            'inactive_occupations': [2.] * ncore,
            'frozen_inactive_orbitals': n_frozen,
            'initial_inactive_orbital_coefficients': space.inactive.tolist(),
            'orbital_optimizer_device': 'cpu'}
    info['active_optimization_method'] = solver.optimizer_options.get('optimization_method', 'simultaneous')
    info['core_response'] = core_response
    info['second_order'] = second_order
    if core_response:
        info['response_method'] = 'scgvb_relaxed_orbital_hessian'
        info['response_history'] = response_history
    return total, solver.C, solver.Cs, final, relaxed, info
