"""Optional SCGVB block optimization, retaining nonorthogonal spatial orbitals.

Only the SCGVB spin coefficients are solved linearly. Determinant amplitudes
remain derived quantities, never independent CI optimization variables.
"""
import numpy as np
from dataclasses import dataclass
from scipy.linalg import eigh


def validate_options(optimization_method, spin_overlap_cutoff, orbital_micro_iterations=0,
                     orbital_update='bfgs', spin_solver='canonical', orbital_overlap_cutoff=1e-10,
                     joint_correction=False, joint_correction_start=1):
    if optimization_method not in ('simultaneous', 'alternating'):
        raise ValueError('optimization_method must be simultaneous or alternating')
    if not np.isfinite(spin_overlap_cutoff) or not 0 < spin_overlap_cutoff < 1:
        raise ValueError('spin_overlap_cutoff must be finite and between 0 and 1')
    if type(orbital_micro_iterations) is not int or orbital_micro_iterations < 0:
        raise ValueError('orbital_micro_iterations must be a nonnegative integer')
    if orbital_micro_iterations and optimization_method != 'alternating':
        raise ValueError('orbital_micro_iterations requires alternating optimization')
    if orbital_update not in ('bfgs', 'wavefunction', 'single_orbital'):
        raise ValueError('orbital_update must be bfgs, wavefunction or single_orbital')
    if not np.isfinite(orbital_overlap_cutoff) or not 0 < orbital_overlap_cutoff < 1:
        raise ValueError('orbital_overlap_cutoff must be finite and between 0 and 1')
    if spin_solver not in ('canonical', 'projected'):
        raise ValueError('spin_solver must be canonical or projected')
    if optimization_method != 'alternating' and (orbital_update != 'bfgs' or spin_solver != 'canonical'):
        raise ValueError('Wavefunction geometry options require alternating optimization')
    if type(joint_correction) is not bool:
        raise ValueError('joint_correction must be a boolean')
    if joint_correction and (optimization_method != 'alternating' or spin_solver != 'projected'):
        raise ValueError('joint_correction requires alternating optimization and projected spin solves')
    if type(joint_correction_start) is not int or joint_correction_start < 1:
        raise ValueError('joint_correction_start must be a positive integer')
    if joint_correction_start != 1 and not joint_correction:
        raise ValueError('joint_correction_start requires joint_correction')


def solve_spin(h, overlap, cutoff=1e-10, *, incoming=None):
    """Lowest spin state in the numerically resolved support of its metric.

    The relative cutoff applies to eigenvalues of the spin-function overlap,
    not to orbital overlaps. Full physical gradients are still required
    for acceptance by the optimizer. An optional valid incoming state is
    retained as one coherent direction, rather than resolving every discarded
    small metric eigenvalue. This preserves the variational comparison after
    an orbital update changes the retained metric support.
    """
    validate_options('alternating', cutoff)
    h, overlap = np.asarray(h), np.asarray(overlap)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape != overlap.shape
            or not np.isfinite(h).all() or not np.isfinite(overlap).all()):
        raise ValueError('Invalid spin Hamiltonian or overlap matrices')
    h, overlap = .5 * (h + h.T), .5 * (overlap + overlap.T)
    values, vectors = eigh(overlap)
    scale = float(values[-1])
    if scale <= 0 or values[0] < -1e-8 * scale:
        raise FloatingPointError('Spin overlap has no positive support or is indefinite')
    keep = values > cutoff * scale
    basis = vectors[:, keep] / np.sqrt(values[keep])
    reference = 0.
    if incoming is not None:
        incoming = np.asarray(incoming, dtype=float)
        if incoming.shape != (len(values),) or not np.isfinite(incoming).all():
            raise ValueError('Invalid incoming spin coefficients')
        norm = float(np.einsum('i,ij,j->', incoming, overlap, incoming))
        if norm <= 0 or not np.isfinite(norm):
            raise FloatingPointError('Incoming spin state has no positive norm')
        reference = float(np.einsum('i,ij,j->', incoming, h, incoming) / norm)
        basis = np.column_stack((basis, incoming / np.sqrt(norm)))
        # All input columns now have unit metric norm. Remove a duplicate
        # incoming direction in this preconditioned metric, without applying
        # the original cutoff to its coherent discarded-space component.
        small_metric = np.einsum('ia,ij,jb->ab', basis, overlap, basis, optimize=True)
        small_values, small_vectors = eigh(.5 * (small_metric + small_metric.T))
        resolved = small_values > max(1e-12, 100 * np.finfo(float).eps * len(small_values)) * small_values[-1]
        basis = np.einsum('ia,ab->ib', basis,
                          small_vectors[:, resolved] / np.sqrt(small_values[resolved]), optimize=True)
    reduced = np.einsum('ia,ij,jb->ab', basis, h - reference * overlap, basis, optimize=True)
    energies, states = eigh(.5 * (reduced + reduced.T))
    cs = np.einsum('ij,j->i', basis, states[:, 0])
    cs /= np.linalg.norm(cs)
    spin_energy = float(energies[0] + reference)
    info = dict(spin_energy_hartree=spin_energy,
                spin_overlap_rank=int(keep.sum()), spin_functions=len(values),
                spin_subspace_dimension=basis.shape[1], spin_incoming_preserved=incoming is not None,
                spin_overlap_cutoff=float(cutoff),
                spin_overlap_eigenvalues=values.tolist(),
                spin_metric_residual_max_abs=float(np.max(np.abs(
                    np.einsum('ij,j->i', h, cs) - spin_energy * np.einsum('ij,j->i', overlap, cs)))))
    return cs, info


def normalized_columns(C, overlap):
    norms = np.einsum('ia,ij,ja->a', C, overlap, C)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise FloatingPointError('Invalid orbital column norms')
    return C / np.sqrt(norms), np.sqrt(norms)


def norm_floor_for_cycle(norm):
    """Keep both blocks inside half the actual incoming unit-column norm.

    The metric eigencutoff resolves a spin basis; it is not a wavefunction
    norm threshold. Rebasing on the accepted state makes every cycle start
    strictly feasible, including a preserved incoming spin direction.
    """
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Cannot start an update cycle with a non-positive norm')
    floor = .5 * norm
    if floor == 0:
        raise FloatingPointError('Wavefunction norm is too small for a safe update cycle')
    return floor


@dataclass
class OrbitalResult:
    C: np.ndarray
    nit: int
    status: int
    norm: float
    gradient_max_abs: float
    norm_floor: float
    rejected_trials: int = 0
    max_accepted_step: float = 0.
    rejection_counts: dict = None
    geometry_info: dict = None
    max_wavefunction_distance: float = 0.


def orbital_block(system, C, Cs, *, norm_floor, max_iterations=1000,
                  max_step=.1, progress=None, orbital_update='bfgs',
                  orbital_overlap_cutoff=1e-10, start_orbital=0):
    """Bounded BFGS with feasible Armijo backtracking on unit column spheres.

    Only individual orbital norms are fixed. Inter-orbital overlaps remain
    unrestricted. Rejected trials have no artificial value or zero gradient;
    they are skipped by the line search. An infeasible start is an error.
    """
    if orbital_update == 'single_orbital':
        from scgvb.optimization.single_orbital import orbital_block as sequential
        return sequential(system, C, Cs, norm_floor=norm_floor, max_iterations=max_iterations,
                          max_step=max_step, progress=progress, cutoff=orbital_overlap_cutoff,
                          start_orbital=start_orbital)
    if not np.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('Orbital norm floor must be positive and finite')
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('Orbital step bound must be positive and finite')
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError('Orbital iteration limit must be a positive integer')
    values, vectors = eigh(system.S)
    if np.any(values <= 0):
        raise FloatingPointError('Orbital basis overlap is not positive definite')
    root = np.einsum('ik,k,jk->ij', vectors, np.sqrt(values), vectors)
    invroot = np.einsum('ik,k,jk->ij', vectors, 1/np.sqrt(values), vectors)
    C, _ = normalized_columns(np.asarray(C, dtype=float), system.S)
    shape = C.shape
    failed_reason = 'nonfinite'
    rejection_counts = dict(norm=0, nonfinite=0, energy=0, wavefunction=0)

    def evaluate(flat, with_gradient):
        nonlocal failed_reason
        failed_reason = 'nonfinite'
        try:
            raw = flat.reshape(shape)
            scales = np.sqrt(np.sum(raw * raw, axis=0))
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                return None
            d = raw / scales
            c = np.einsum('ij,ja->ia', invroot, d)
            numerator, norm = system.calcEnergy(c, Cs)
            if not np.isfinite(norm):
                return None
            if norm < norm_floor:
                failed_reason = 'norm'
                return None
            value = float(numerator / norm)
            if not np.isfinite(value):
                return None
            gradient = None
            if with_gradient:
                gc = system.rayleigh_gradient(np.r_[c.ravel(), Cs])[:c.size].reshape(shape)
                gd = np.einsum('ij,ja->ia', invroot, gc)
                gradient = ((gd - d * np.sum(d * gd, axis=0)) / scales).ravel()
                if not np.isfinite(gradient).all():
                    return None
            return value, float(norm), d.ravel(), c, gradient
        except (FloatingPointError, OverflowError):
            return None

    x = np.einsum('ij,ja->ia', root, C).ravel()
    state = evaluate(x, True)
    if state is None:
        raise FloatingPointError('Orbital block starts with an infeasible or non-finite state; '
                                 'no optimization was performed')
    value, norm, x, c, gradient = state
    if orbital_update == 'wavefunction':
        from scgvb.optimization.wavefunction import orbital_geometry
        initial_inverse, wavefunction_metric, geometry_info = orbital_geometry(system, c, Cs, root, invroot)
    elif orbital_update == 'bfgs':
        initial_inverse = np.eye(len(x)) / max(1., np.max(np.abs(gradient)))
        wavefunction_metric, geometry_info = None, {}
    else:
        raise ValueError('orbital_update must be bfgs or wavefunction')
    inverse = initial_inverse.copy()
    largest_wavefunction_distance = 0.
    rejected, accepted, largest_step, status = 0, 0, 0., 1

    def report():
        if progress is not None:
            progress(dict(orbital_iteration=accepted, orbital_energy_hartree=value,
                          orbital_unit_norm=norm, orbital_norm_floor=norm_floor,
                          orbital_gradient_max_abs=float(np.max(np.abs(gradient))),
                          orbital_rejected_trials=rejected,
                          orbital_rejection_counts=rejection_counts.copy(),
                          orbital_update=orbital_update,
                          orbital_max_wavefunction_distance=largest_wavefunction_distance,
                          **geometry_info))

    report()
    for _ in range(max_iterations):
        if np.max(np.abs(gradient)) <= 1e-8:
            status = 0
            break
        direction = -np.einsum('ij,j->i', inverse, gradient)
        d = x.reshape(shape)
        direction = direction.reshape(shape)
        direction -= d * np.sum(d * direction, axis=0)
        direction = direction.ravel()
        slope = float(np.dot(gradient, direction))
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
            inverse = (initial_inverse.copy() if wavefunction_metric is not None else
                       np.eye(len(x))/max(1., np.max(np.abs(gradient))))
            direction = -np.einsum('ij,j->i', inverse, gradient)
            direction = direction.reshape(shape)
            direction -= d*np.sum(d*direction, axis=0)
            direction = direction.ravel()
        size = float(np.max(np.linalg.norm(direction.reshape(shape), axis=0)))
        if size > max_step:
            direction *= max_step / size
        if wavefunction_metric is not None:
            distance_squared = float(np.einsum('i,ij,j->', direction, wavefunction_metric, direction))
            if distance_squared > .05**2:
                direction *= .05/np.sqrt(distance_squared)
        slope = float(np.dot(gradient, direction))
        alpha, next_state, hit_guard = 1., None, False
        for _ in range(40):
            rejection_reason = 'energy'
            trial_distance = 0.
            trial = x + alpha * direction
            trial_state = evaluate(trial, False)
            if trial_state is None:
                hit_guard = True
                rejection_reason = failed_reason
            elif trial_state[0] < value and trial_state[0] <= value + 1e-4 * alpha * slope:
                if wavefunction_metric is not None:
                    overlap = system.wavefunction_overlap(c, Cs, trial_state[3], Cs)
                    trial_distance = float(np.sqrt(max(0., 1.-overlap**2)))
                    if not np.isfinite(overlap) or trial_distance > .05:
                        rejected += 1
                        rejection_counts['wavefunction'] += 1
                        alpha *= .5
                        continue
                # Evaluate the derivative at the retracted, unit-column point.
                trial_state = evaluate(trial_state[2], True)
                if (trial_state is not None and trial_state[0] < value
                        and trial_state[0] <= value + 1e-4 * alpha * slope):
                    next_state = trial_state
                    break
                if trial_state is None:
                    rejection_reason = failed_reason
            rejected += 1
            rejection_counts[rejection_reason] += 1
            alpha *= .5
        if next_state is None:
            # A line-search failure with a nonzero gradient is never convergence.
            status = 3 if hit_guard else 2
            break
        new_value, new_norm, new_x, new_c, new_gradient = next_state
        step, change = new_x - x, new_gradient - gradient
        curvature = float(np.dot(step, change))
        if curvature > 1e-10 * np.linalg.norm(step) * np.linalg.norm(change):
            hy = np.einsum('ij,j->i', inverse, change)
            inverse += ((curvature + np.dot(change, hy)) / curvature**2 * np.outer(step, step)
                        - (np.outer(hy, step) + np.outer(step, hy)) / curvature)
            inverse = .5 * (inverse + inverse.T)
            if not np.isfinite(inverse).all():
                inverse = (initial_inverse.copy() if wavefunction_metric is not None else
                           np.eye(len(x))/max(1., np.max(np.abs(new_gradient))))
        largest_step = max(largest_step, float(np.max(np.linalg.norm(step.reshape(shape), axis=0))))
        largest_wavefunction_distance = max(largest_wavefunction_distance, trial_distance)
        value, norm, x, c, gradient = new_value, new_norm, new_x, new_c, new_gradient
        accepted += 1
        if accepted == 1 or accepted % 25 == 0:
            report()
    report()
    return OrbitalResult(c, accepted, status, norm, float(np.max(np.abs(gradient))),
                         norm_floor, rejected, largest_step, rejection_counts, geometry_info,
                         largest_wavefunction_distance)


def guarded_spin_update(system, C, incoming, proposed, *, norm_floor=None):
    """Accept a physical spin descent in the same feasible domain as orbitals."""
    incoming, proposed = np.asarray(incoming, dtype=float), np.asarray(proposed, dtype=float)
    if (incoming.shape != proposed.shape or incoming.ndim != 1
            or not np.isfinite(incoming).all() or not np.isfinite(proposed).all()
            or np.linalg.norm(incoming) == 0 or np.linalg.norm(proposed) == 0):
        raise ValueError('Spin update requires finite nonzero coefficient vectors of matching shape')
    if norm_floor is not None and (not np.isfinite(norm_floor) or norm_floor <= 0):
        raise ValueError('Spin norm floor must be positive and finite')
    incoming = incoming / np.linalg.norm(incoming)
    proposed = proposed / np.linalg.norm(proposed)
    if np.dot(proposed, incoming) < 0:
        proposed = -proposed
    numerator, norm = system.calcEnergy(C, incoming)
    if not np.isfinite(norm) or norm <= 0 or (norm_floor is not None and norm < norm_floor):
        raise FloatingPointError('Spin block starts with an infeasible wavefunction norm')
    before = float(numerator / norm)
    if not np.isfinite(before):
        raise FloatingPointError('Spin block starts with a non-finite energy')
    fraction, rejected = 1., 0
    for _ in range(40):
        cs = incoming + fraction * (proposed - incoming)
        cs /= np.linalg.norm(cs)
        try:
            numerator, trial_norm = system.calcEnergy(C, cs)
            value = float(numerator / trial_norm) if trial_norm > 0 else np.inf
        except (FloatingPointError, OverflowError):
            value, trial_norm = np.inf, 0.
        if (np.isfinite(value) and np.isfinite(trial_norm) and trial_norm > 0
                and (norm_floor is None or trial_norm >= norm_floor)
                and value <= before):
            return cs, dict(spin_step_fraction=fraction, spin_rejected_trials=rejected,
                            spin_update_status='accepted', spin_energy_hartree=value)
        rejected += 1
        fraction *= .5
    return incoming.copy(), dict(spin_step_fraction=0., spin_rejected_trials=rejected,
                                 spin_update_status='retained_incoming', spin_energy_hartree=before)


def optimize(system, C, Cs, *, verbose=False, epsilon=1e-8,
             max_outer_iterations=50, max_inner_iterations=1000,
             spin_overlap_cutoff=1e-10, orbital_micro_iterations=0,
             orbital_update='bfgs', spin_solver='canonical', orbital_overlap_cutoff=1e-10,
             joint_correction=False, joint_correction_start=1):
    validate_options('alternating', spin_overlap_cutoff, orbital_micro_iterations,
                     orbital_update, spin_solver, orbital_overlap_cutoff, joint_correction, joint_correction_start)
    if joint_correction and getattr(system, 'energy_evaluation', None) != 'determinant':
        raise ValueError('joint_correction requires determinant energy evaluation')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    for limit in (max_outer_iterations, max_inner_iterations):
        if not isinstance(limit, (int, np.integer)) or limit < 1:
            raise ValueError('Iteration limits must be positive integers')
    C, Cs = np.asarray(C, dtype=float).copy(), np.asarray(Cs, dtype=float).copy()
    # Reuse the public shape/finite/nonzero-spin validation.
    C, Cs = system.split_x(np.r_[C.ravel(), Cs])
    if not np.isfinite(C).all() or not np.isfinite(Cs).all():
        raise FloatingPointError('Initial coefficients must be finite')
    C, _ = normalized_columns(C, system.S)
    diagnostics = getattr(system, 'diagnostics', None)
    info = dict(optimization_method='alternating', spin_overlap_cutoff=spin_overlap_cutoff,
                physical_gradient_tolerance=1e-6, norm_retention=.5, orbital_step_limit=.1,
                orbital_micro_iterations=orbital_micro_iterations,
                orbital_update=orbital_update, spin_solver=spin_solver,
                orbital_overlap_cutoff=orbital_overlap_cutoff,
                joint_correction=joint_correction,
                joint_correction_start=joint_correction_start,
                converged=False, history=[])
    system.optimization_info = info
    saved = np.r_[C.ravel(), Cs]
    if diagnostics is not None:
        diagnostics.start(system)
        diagnostics.snapshot(system, 'start', saved, optimizer_options=dict(
            optimization_method='alternating', spin_overlap_cutoff=spin_overlap_cutoff,
            norm_retention=.5, orbital_step_limit=.1,
            joint_correction=joint_correction,
            joint_correction_start=joint_correction_start,
            spin_initialization=getattr(system, 'spin_initialization_info', None),
            epsilon=epsilon, max_outer_iterations=max_outer_iterations,
            max_inner_iterations=max_inner_iterations,
            orbital_micro_iterations=orbital_micro_iterations,
            orbital_update=orbital_update, spin_solver=spin_solver,
            orbital_overlap_cutoff=orbital_overlap_cutoff))

    def energy(c, cs):
        numerator, norm = system.calcEnergy(c, cs)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError('Non-positive wavefunction norm')
        return float(numerator / norm), float(norm)

    def spin_step(c, old_cs, norm_floor=None):
        nonlocal saved
        # The orbital block may have moved since the last accepted spin solve.
        # Preserve its actual incoming state if this spin solve fails.
        saved = np.r_[c.ravel(), old_cs]
        if spin_solver == 'projected':
            from scgvb.optimization.wavefunction import projected_spin_solve
            cs, spin_info = projected_spin_solve(system, c, old_cs, spin_overlap_cutoff)
            eigenvalue = spin_info['spin_energy_hartree']
            cs, guard_info = guarded_spin_update(system, c, old_cs, cs, norm_floor=norm_floor)
            spin_gradient = system.rayleigh_gradient(np.r_[c.ravel(), cs])[c.size:]
            physical_norm = energy(c, cs)[1]
            spin_info.update(guard_info, spin_unrestricted_energy_hartree=eigenvalue,
                             spin_metric_residual_max_abs=float(.5*physical_norm*np.max(np.abs(spin_gradient))),
                             spin_physical_gradient_max_abs=float(np.max(np.abs(spin_gradient))))
            return cs, spin_info
        h, metric = system.spin_matrices(c)
        cs, spin_info = solve_spin(h, metric, spin_overlap_cutoff)
        if np.dot(cs, old_cs) < 0:
            cs = -cs
        before = energy(c, old_cs)[0]
        after = energy(c, cs)[0]
        if after > before + 1e-9:
            # The previous orbital block minimized a physical state that need
            # not lie in the newly truncated metric support. Keep that state
            # in the linear spin solve instead of raising the cutoff blindly.
            cs, spin_info = solve_spin(h, metric, spin_overlap_cutoff, incoming=old_cs)
            if np.dot(cs, old_cs) < 0:
                cs = -cs
            after = energy(c, cs)[0]
        eigenvalue = spin_info['spin_energy_hartree']
        cs, guard_info = guarded_spin_update(system, c, old_cs, cs, norm_floor=norm_floor)
        spin_info.update(guard_info, spin_unrestricted_energy_hartree=eigenvalue,
                         spin_metric_residual_max_abs=float(np.max(np.abs(
                             np.einsum('ij,j->i', h, cs) - guard_info['spin_energy_hartree'] *
                             np.einsum('ij,j->i', metric, cs)))))
        return cs, spin_info

    try:
        Cs, spin_info = spin_step(C, Cs)
        previous_energy = energy(C, Cs)[0]
        stalled = 0
        orbital_cursor = 0
        joint_previous = None
        for iteration in range(1, max_outer_iterations + 1):
            reference_norm = energy(C, Cs)[1]
            floor = norm_floor_for_cycle(reference_norm)
            micro_history = []
            orbital_iterations, rejected_trials, largest_step = 0, 0, 0.
            orbital_attempts = 0
            # One fixed floor and one total orbital-step budget per outer
            # cycle. Micro updates do not rebase the floor. A new BFGS block
            # starts after each spin solve because its objective has changed.
            micro_limit = orbital_micro_iterations or max_inner_iterations
            micro_count_limit = (max_inner_iterations + micro_limit - 1) // micro_limit

            def progress(details):
                if diagnostics is not None:
                    diagnostics.snapshot(system, 'orbital_progress', iteration=iteration,
                                         micro_iteration=len(micro_history) + 1, **details)
                if verbose:
                    print(f'Orbital block {iteration}.{len(micro_history) + 1}, '
                          f'step {details["orbital_iteration"]}: '
                          f'energy={details["orbital_energy_hartree"]:.12f}, '
                          f'unit_norm={details["orbital_unit_norm"]:g}, floor={floor:g}', flush=True)

            for micro in range(1, micro_count_limit + 1):
                before_spin = Cs.copy()
                budget_used = orbital_attempts if orbital_update == 'single_orbital' else orbital_iterations
                orbital = orbital_block(system, C, Cs, norm_floor=floor,
                                        max_iterations=min(micro_limit, max_inner_iterations - budget_used),
                                        progress=progress, orbital_update=orbital_update,
                                        orbital_overlap_cutoff=orbital_overlap_cutoff,
                                        start_orbital=orbital_cursor)
                if orbital_update == 'single_orbital':
                    orbital_cursor = orbital.geometry_info['next_orbital']
                    orbital_attempts += orbital.geometry_info['orbital_attempted_solves']
                C = orbital.C
                Cs, spin_info = spin_step(C, Cs, norm_floor=floor)
                # Failure snapshots must retain the last accepted spin state.
                saved = np.r_[C.ravel(), Cs]
                orbital_iterations += orbital.nit
                rejected_trials += orbital.rejected_trials
                largest_step = max(largest_step, orbital.max_accepted_step)
                micro_energy, micro_norm = energy(C, Cs)
                micro_record = dict(micro_iteration=micro, orbital_iterations=int(orbital.nit),
                                    orbital_optimizer_status=int(orbital.status),
                                    active_energy_hartree=micro_energy, unit_orbital_norm=micro_norm,
                                    cycle_norm_floor=floor, **spin_info)
                micro_record.update(orbital_rejection_counts=orbital.rejection_counts,
                                    orbital_max_wavefunction_distance=orbital.max_wavefunction_distance,
                                    orbital_geometry=orbital.geometry_info)
                micro_history.append(micro_record)
                if orbital_micro_iterations:
                    if diagnostics is not None:
                        diagnostics.snapshot(system, 'alternating_micro', iteration=iteration, **micro_record)
                    if verbose:
                        print(f'Spin update {iteration}.{micro}: energy={micro_energy:.12f}, '
                              f'fraction={spin_info["spin_step_fraction"]:g}, '
                              f'unit_norm={micro_norm:g}, floor={floor:g}', flush=True)
                # A zero-step full-orbital block continues only if spin changed.
                # A retained single column must still advance to other columns.
                # Neither a bound-limited nor a zero-step stop is convergence.
                budget_used = orbital_attempts if orbital_update == 'single_orbital' else orbital_iterations
                if (budget_used >= max_inner_iterations
                        or (orbital_update != 'single_orbital' and orbital.nit == 0
                            and np.linalg.norm(Cs - before_spin) <= 1e-10)):
                    break
            joint_info = None
            joint_active = joint_correction and iteration >= joint_correction_start
            if joint_active:
                if iteration == joint_correction_start:
                    # Warmup stagnation must not immediately stop the new stage.
                    stalled = 0
                from scgvb.optimization.coupled import correction
                before_joint = system.normalized_candidate(np.r_[C.ravel(), Cs])
                if diagnostics is not None:
                    diagnostics.snapshot(system, 'joint_before', before_joint, iteration=iteration)
                C, Cs, joint_info, joint_step = correction(system, C, Cs, norm_floor=floor,
                    spin_cutoff=spin_overlap_cutoff, previous=joint_previous)
                if joint_info['joint_accepted']:
                    joint_previous = joint_step
                saved = np.r_[C.ravel(), Cs]
                if diagnostics is not None:
                    diagnostics.snapshot(system, 'joint_after', system.normalized_candidate(saved),
                                         iteration=iteration, **joint_info)
            current_energy, unit_norm = energy(C, Cs)
            saved = system.normalized_candidate(np.r_[C.ravel(), Cs])
            gradient = system.rayleigh_gradient(saved)
            norm_error = system.calcEnergy(*system.split_x(saved))[1] - 1.
            gradient_max = float(np.max(np.abs(gradient)))
            record = dict(iteration=iteration, energy_hartree=current_energy + system.energy_nuc
                          + getattr(system, 'inactive_energy', 0.),
                          physical_gradient_max_abs=gradient_max, normalization_error=float(norm_error),
                          normalized_orbital_gradient_max_abs=float(np.max(np.abs(gradient[:C.size]))),
                          normalized_spin_gradient_max_abs=float(np.max(np.abs(gradient[C.size:]))),
                          physical_gradient_l2=float(np.linalg.norm(gradient)),
                          unit_orbital_norm=unit_norm, cycle_reference_norm=reference_norm,
                          cycle_norm_floor=floor, orbital_iterations=int(orbital_iterations),
                          orbital_gradient_max_abs=orbital.gradient_max_abs,
                          orbital_rejected_trials=rejected_trials,
                          orbital_max_accepted_step=largest_step,
                          orbital_optimizer_status=int(orbital.status), **spin_info)
            if orbital_update == 'single_orbital':
                record['orbital_attempted_solves'] = int(orbital_attempts)
            if joint_correction:
                record['joint_correction'] = joint_info
                record['joint_correction_active'] = joint_active
                record['optimization_phase'] = 'joint_correction' if joint_active else 'alternating'
            record.update(orbital_rejection_counts={key:sum((m['orbital_rejection_counts'] or {}).get(key, 0)
                                                           for m in micro_history)
                                                    for key in ('norm', 'nonfinite', 'energy', 'wavefunction',
                                                                'orbital_step')},
                          orbital_max_wavefunction_distance=max(m['orbital_max_wavefunction_distance']
                                                                for m in micro_history))
            if orbital_micro_iterations:
                record.update(spin_updates=len(micro_history), micro_history=micro_history)
            info['history'].append(record)
            if diagnostics is not None:
                diagnostics.snapshot(system, 'alternating', saved, **record)
            if verbose:
                print(f'Alternating iteration {iteration}: energy={record["energy_hartree"]:.12f}, '
                      f'gradient={gradient_max:g}, spin_rank={spin_info["spin_overlap_rank"]}/{system.fs}, '
                      f'unit_orbital_norm={unit_norm:g}')
            if (np.isfinite(gradient).all() and gradient_max <= 1e-6
                    and abs(norm_error) < epsilon):
                final_C, final_Cs = system.split_x(saved)
                info.update(converged=True, iterations=iteration)
                if diagnostics is not None:
                    diagnostics.snapshot(system, 'accepted', saved)
                return energy(final_C, final_Cs)[0] + system.energy_nuc + getattr(system, 'inactive_energy', 0.), final_C, final_Cs
            if current_energy > previous_energy + 1e-9:
                raise RuntimeError('Alternating optimization increased the physical energy')
            stalled = stalled + 1 if previous_energy - current_energy < 1e-10 else 0
            # Bounded warmup may reach a plateau before the coupled stage.
            # Energy increase and full convergence checks still apply throughout.
            if stalled >= 5 and (not joint_correction or joint_active):
                raise RuntimeError(f'Alternating optimization stalled (physical_gradient={gradient_max:g}). '
                                   'No converged energy is available.')
            previous_energy = current_energy
        raise RuntimeError(f'Alternating optimization did not converge in {max_outer_iterations} iterations '
                           f'(physical_gradient={gradient_max:g}). No converged energy is available.')
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.snapshot(system, 'failure', saved, error=str(exc),
                                 spin_rejection=info.get('spin_rejection'))
        raise
