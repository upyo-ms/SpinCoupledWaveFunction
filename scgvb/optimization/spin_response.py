"""Experimental joint correction with damping confined to orbital variables.

Spin response is eliminated from the local Hessian before the orbital trust
step. Retraction, the fixed norm floor and physical residual guards still
apply to the resulting *joint* state. Existing correction APIs are unchanged.
"""
import time
import numpy as np
from scipy.linalg import eigh
from scgvb.optimization.alternating import normalized_columns
from scgvb.optimization.coupled import (direction_basis, reduced_gradient, reduced_hessian,
                                 retract, append_spin_residual, _physical_residual)


class UnresolvedResponse(FloatingPointError):
    """The small model cannot resolve an undamped, stable spin response."""


def split_joint_basis(basis, orbital_size, cutoff=1e-10):
    """Separate the orbital and spin spans, including mixed previous steps.

    Splitting can enlarge a mixed span. We retain both of its components,
    rather than treating a mixed direction as orbital or spin by its label.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2 or not 0 < orbital_size < basis.shape[0] or not np.isfinite(basis).all():
        raise ValueError('Incompatible joint basis')
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError('Block direction cutoff must be between zero and one')
    columns, labels, orbital_count = [], [], 0
    for lo, hi, label in [(0, orbital_size, 'orbital'),
                          (orbital_size, basis.shape[0], 'spin')]:
        u, singular, _ = np.linalg.svd(basis[lo:hi], full_matrices=False)
        keep = singular > max(cutoff*(singular[0] if len(singular) else 0), cutoff)
        for j in np.flatnonzero(keep):
            vector = np.zeros(basis.shape[0]); vector[lo:hi] = u[:, j]
            columns.append(vector); labels.append(f'{label}_subspace_{j+1}')
        if label == 'orbital': orbital_count = int(keep.sum())
    return (np.column_stack(columns) if columns else np.zeros((basis.shape[0], 0)),
            labels, orbital_count)


def _metric_transform(metric):
    if not np.isfinite(metric).all():
        raise UnresolvedResponse('nonfinite_metric')
    values, vectors = eigh(.5*(metric+metric.T))
    if not np.isfinite(values).all() or values[-1] <= 0 or values[0] < -1e-8*values[-1]:
        raise UnresolvedResponse('unresolved_metric')
    keep = values > max(1e-12*values[-1], np.finfo(float).tiny)
    return vectors[:, keep]/np.sqrt(values[keep]), values


def spin_response_step(g, hessian, metric, orbital_count, trust_radius):
    """Schur solve: Hss ds = -gs-Hso do; only do receives a level shift.

    The orbital trust metric includes the spin change induced by do, including
    orbital/spin cross terms. The affine spin correction at do=0 is undamped.
    A singular or nonpositive resolved spin Hessian is reported, not shifted.
    """
    g, hessian, metric = map(lambda a: np.asarray(a, dtype=float), (g, hessian, metric))
    count = len(g); no = orbital_count
    if (g.ndim != 1 or hessian.shape != (count, count) or metric.shape != (count, count)
            or type(no) is not int or not 0 < no < count
            or not np.isfinite(g).all() or not np.isfinite(hessian).all()
            or not np.isfinite(metric).all() or not np.isfinite(trust_radius)
            or not 0 < trust_radius <= .05):
        raise ValueError('Incompatible spin-response model or trust radius')
    _metric_transform(metric)  # Check the full physical metric as well.
    spin_transform, spin_metric_values = _metric_transform(metric[no:, no:])
    if spin_transform.shape[1] != count-no:
        raise UnresolvedResponse('unresolved_spin_metric')
    hss = spin_transform.T @ hessian[no:, no:] @ spin_transform
    curvature, rotation = eigh(.5*(hss+hss.T))
    if not np.isfinite(curvature).all() or curvature[0] <= 1e-12*max(1., np.max(np.abs(curvature))):
        raise UnresolvedResponse('unresolved_spin_curvature')
    inverse = (rotation/curvature) @ rotation.T
    kernel = spin_transform @ inverse @ spin_transform.T
    spin_offset = -kernel @ g[no:]
    response = -kernel @ hessian[no:, :no]
    lift = np.vstack((np.eye(no), response))
    orbital_metric = lift.T @ metric @ lift
    transform, orbital_metric_values = _metric_transform(orbital_metric)
    effective_h = hessian[:no, :no]+hessian[:no, no:] @ response
    effective_g = g[:no]+hessian[:no, no:] @ spin_offset
    reduced = transform.T @ effective_h @ transform
    orbital_curvature, orbital_rotation = eigh(.5*(reduced+reduced.T))
    hg = transform.T @ effective_g
    shift = max(0., -orbital_curvature[0])+1e-6*max(1., np.max(np.abs(orbital_curvature)))
    for _ in range(60):
        orbital_step = -orbital_rotation @ ((orbital_rotation.T @ hg)/(orbital_curvature+shift))
        if np.linalg.norm(orbital_step) <= trust_radius: break
        shift = max(2*shift, 1e-6)
    else:
        raise UnresolvedResponse('unresolved_orbital_trust_step')
    do = transform @ orbital_step
    step = np.r_[do, spin_offset+response @ do]
    predicted = g+hessian @ step
    if not np.isfinite(step).all() or not np.isfinite(predicted).all():
        raise UnresolvedResponse('nonfinite_response')
    info = dict(joint_orbital_basis_dimension=no, joint_spin_basis_dimension=count-no,
                joint_spin_metric_eigenvalues=spin_metric_values.tolist(),
                joint_spin_metric_rank=spin_transform.shape[1],
                joint_spin_curvature_eigenvalues=curvature.tolist(),
                joint_response_metric_eigenvalues=orbital_metric_values.tolist(),
                joint_response_metric_rank=transform.shape[1],
                joint_curvature_eigenvalues=orbital_curvature.tolist(),
                joint_level_shift=float(shift), joint_spin_level_shift=0.,
                joint_spin_offset=spin_offset.tolist(), joint_spin_response_matrix=response.tolist(),
                joint_orbital_coordinate_step=do.tolist(),
                joint_predicted_spin_residual=predicted[no:].tolist(),
                joint_predicted_spin_residual_max_abs=float(np.max(np.abs(predicted[no:]))),
                joint_predicted_slope=float(g @ step),
                joint_predicted_orbital_response_distance=float(np.linalg.norm(orbital_step)),
                joint_predicted_wavefunction_distance=float(np.sqrt(max(0., step @ metric @ step))))
    return step, info


def _response_in_basis(system, C, Cs, basis, labels, orbital_count, *, norm_floor,
                       trust_radius, residual_guard=None, trial_recorder=None):
    """One original-base model, with orbital-only backtracking and spin recovery."""
    num, norm = system.calcEnergy(C, Cs); before = float(num/norm)
    count = basis.shape[1]
    info = dict(joint_status='retained', joint_accepted=False, joint_spin_response=True,
                joint_basis_labels=labels.copy(), joint_basis_dimension=count,
                joint_reduced_basis=basis.tolist(),
                joint_norm_floor=norm_floor, joint_trust_radius=trust_radius,
                joint_energy_before_hartree=before, joint_gradient_evaluations=0,
                joint_rejected_trials=0, joint_rejection_counts=dict(
                    nonfinite=0, norm=0, orbital_step=0, energy=0, wavefunction=0, residual=0))
    zero = np.zeros(C.size+len(Cs))
    if not 0 < orbital_count < count <= 10:
        info['joint_status'] = 'missing_response_block'; return C, Cs, info, zero
    if not np.isfinite(before) or norm < norm_floor:
        raise FloatingPointError('Spin-response correction starts from an infeasible state')
    metric = system.joint_wavefunction_metric(C, Cs, np.pad(basis, ((0, 0), (0, 10-count))))[:count, :count]
    metric = .5*(metric+metric.T)
    g = reduced_gradient(system, C, Cs, basis, np.zeros(count))
    hessian, asymmetry, steps = reduced_hessian(system, C, Cs, basis, metric)
    info.update(joint_gradient_evaluations=1+2*count, joint_hessian_asymmetry=asymmetry,
                joint_finite_difference_steps=steps.tolist(), joint_reduced_gradient=g.tolist(),
                joint_reduced_hessian=hessian.tolist(), joint_reduced_metric=metric.tolist())
    if not np.isfinite(hessian).all() or not np.isfinite(g).all() or asymmetry > .05:
        info['joint_status'] = 'unreliable_curvature'; return C, Cs, info, zero
    try:
        step, model = spin_response_step(g, hessian, metric, orbital_count, trust_radius)
    except UnresolvedResponse as exc:
        info['joint_status'] = str(exc); return C, Cs, info, zero
    info.update(model)
    offset = np.asarray(model['joint_spin_offset']); response = np.asarray(model['joint_spin_response_matrix'])
    fraction = 1.
    for _ in range(30):
        do = fraction*step[:orbital_count]
        # Recompute spin response at every orbital fraction. Scaling the full
        # joint step would reintroduce (1-fraction)*gs into its model residual.
        z = np.r_[do, offset+response @ do]
        slope = float(g @ z); reason = 'nonfinite'
        try:
            c, cs, _, _ = retract(C, Cs, basis, z, system.S)
            delta = c-C
            orbital_step = float(np.max(np.sqrt(np.maximum(0., np.einsum('ia,ij,ja->a',delta,system.S,delta)))))
            n, den = system.calcEnergy(c, cs); energy = float(n/den) if den > 0 else np.inf
            if not np.isfinite(energy) or not np.isfinite(den) or not np.isfinite(slope): reason = 'nonfinite'
            elif den < norm_floor: reason = 'norm'
            elif orbital_step > .1+1e-12: reason = 'orbital_step'
            elif slope >= 0 or energy >= before or energy > before+1e-4*slope: reason = 'energy'
            else:
                overlap = system.wavefunction_overlap(C, Cs, c, cs)
                distance = float(np.sqrt(max(0., 1-overlap**2)))
                if not np.isfinite(distance): reason = 'nonfinite'
                elif distance > .05+1e-12: reason = 'wavefunction'
                else:
                    if residual_guard is None:
                        passed, residual = True, None
                    else:
                        residual, spin = _physical_residual(system, c, cs)
                        info['joint_gradient_evaluations'] += 1
                        passed = residual_guard(residual)
                        if trial_recorder is not None:
                            trial_recorder(c, cs, dict(fraction=fraction, passed=bool(passed),
                                residual=residual, predicted_spin_residual=(g+hessian @ z)[orbital_count:].tolist()))
                    if passed:
                        info.update(joint_status='response_candidate', joint_accepted=True,
                            joint_step_fraction=fraction, joint_energy_after_hartree=energy,
                            joint_unit_orbital_norm=float(den), joint_orbital_step=orbital_step,
                            joint_wavefunction_distance=distance, joint_final_residual=residual,
                            joint_accepted_coordinate_step=z.tolist(), joint_accepted_predicted_slope=slope,
                            joint_accepted_predicted_spin_residual=(g+hessian @ z)[orbital_count:].tolist())
                        return c, cs, info, np.r_[(c-C).ravel(), cs-Cs]
                    reason = 'residual'
        except (FloatingPointError, OverflowError):
            reason = 'nonfinite'
        info['joint_rejection_counts'][reason] += 1
        info['joint_rejected_trials'] += 1; fraction *= .5
    info.update(joint_status='response_line_search_retained', joint_step_fraction=0.)
    return C, Cs, info, zero


def response_correction(system, C, Cs, *, norm_floor, spin_cutoff=1e-12,
                        previous=None, max_directions=6, trust_radius=.02,
                        direction_cutoff=1e-12, max_added_directions=4,
                        residual_direction_cutoff=1e-10, spin_growth_limit=2.,
                        trial_recorder=None, initial_basis=None):
    """Resolve spin response, enrich rejected candidates, then shrink orbitals.

    All models use the original base. At most ten block-separated directions
    are retained. Full-gradient/spin guards are the same as adaptive_correction.
    No separate spin state or orbital state is committed inside this procedure.
    """
    if type(max_directions) is not int or not 2 <= max_directions <= 6:
        raise ValueError('Response correction starts with 2 to 6 directions')
    if type(max_added_directions) is not int or not 1 <= max_added_directions <= 4:
        raise ValueError('Response correction adds 1 to 4 directions')
    for value in (spin_cutoff, direction_cutoff, residual_direction_cutoff):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError('Response cutoffs must be between zero and one')
    if not np.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('Response correction requires a positive fixed norm floor')
    if not np.isfinite(trust_radius) or not 0 < trust_radius <= .05:
        raise ValueError('Response trust radius must be positive and at most 0.05')
    if not np.isfinite(spin_growth_limit) or spin_growth_limit < 1:
        raise ValueError('Response spin growth limit must be at least one')
    if getattr(system, 'energy_evaluation', None) != 'determinant':
        raise ValueError('Response correction requires determinant energy evaluation')
    started = time.perf_counter()
    C, _ = normalized_columns(np.asarray(C, dtype=float), system.S)
    Cs = np.asarray(Cs, dtype=float).copy(); Cs /= np.linalg.norm(Cs)
    num, norm = system.calcEnergy(C, Cs)
    if not np.isfinite(num/norm) or norm < norm_floor:
        raise FloatingPointError('Response correction starts from an infeasible state')
    incoming, _ = _physical_residual(system, C, Cs)
    spin_bound = max(1e-6, spin_growth_limit*incoming['spin_l2'])
    def guard(row):
        return row['full_l2'] <= incoming['full_l2']+1e-8 and row['spin_l2'] <= spin_bound
    if initial_basis is None:
        raw, raw_labels = direction_basis(system, C, Cs, spin_cutoff=spin_cutoff,
            previous=previous, max_directions=max_directions, direction_cutoff=direction_cutoff)
    else:
        raw, raw_labels = np.asarray(initial_basis[0], dtype=float).copy(), list(initial_basis[1])
        if (raw.ndim != 2 or raw.shape[0] != C.size+len(Cs) or raw.shape[1] != len(raw_labels)
                or not 0 < raw.shape[1] <= max_directions or not np.isfinite(raw).all()
                or not np.allclose(raw.T @ raw, np.eye(raw.shape[1]), atol=1e-10, rtol=1e-10)):
            raise ValueError('Initial response basis must be finite, orthonormal and compatible')
        dc = raw[:C.size].reshape(*C.shape, raw.shape[1])
        if (np.any(np.abs(np.einsum('ia,ij,jak->ak', C, system.S, dc))>1e-10)
                or np.any(np.abs(Cs @ raw[C.size:])>1e-10)):
            raise ValueError('Initial response basis must be tangent to the supplied state')
    basis, labels, no = split_joint_basis(raw, C.size)
    if basis.shape[1] > 10:
        raise ValueError('Separated response basis exceeds ten directions')
    capacity = min(max_added_directions, 10-basis.shape[1])
    rounds, searches = [], []
    evaluations = 1+int(initial_basis is None); rejected = 0; reason = 'addition_limit'
    counts = dict(nonfinite=0, norm=0, orbital_step=0, energy=0, wavefunction=0, residual=0)
    def solve(guarded=False):
        nonlocal evaluations, rejected
        def record(a,b,row):
            details = dict(stage='orbital_backtracking', round=len(rounds), **row)
            searches.append(details)
            if trial_recorder is not None: trial_recorder(a,b,details)
        result = _response_in_basis(system, C, Cs, basis, labels, no,
            norm_floor=norm_floor, trust_radius=trust_radius,
            residual_guard=guard if guarded else None, trial_recorder=record if guarded else None)
        model = result[2]; evaluations += model['joint_gradient_evaluations']
        rejected += model['joint_rejected_trials']
        for k,v in model['joint_rejection_counts'].items(): counts[k] += v
        return result
    accepted = False; result = None
    for index in range(capacity+1):
        result = solve()
        c, cs, model, displacement = result
        residual, spin = _physical_residual(system, c, cs); evaluations += 1
        accepted = bool(model['joint_accepted'] and guard(residual))
        row = dict(stage='enrichment', round=index, model=model, physical_residual=residual,
                   residual_guard_passed=accepted, added_directions=index)
        rounds.append(row)
        if trial_recorder is not None: trial_recorder(c, cs, row)
        if accepted: break
        if not model['joint_accepted']:
            reason = model['joint_status']; break
        if index == capacity: break
        expanded, addition = append_spin_residual(basis, C, Cs, spin, cutoff=residual_direction_cutoff)
        row['direction_addition'] = addition
        if not addition['added']:
            reason = 'dependent_spin_residual'; break
        basis = expanded; labels.append(f'post_trial_spin_residual_{index+1}')
    if not accepted and result[2]['joint_accepted']:
        # Enrich before shrinking so a missing response direction cannot be
        # hidden by accepting an almost vanishing orbital move.
        result = solve(guarded=True)
        c, cs, model, displacement = result
        accepted = bool(model['joint_accepted'])
        if accepted:
            residual = model['joint_final_residual']
        else: reason = model['joint_status']
    if accepted:
        c, cs, model, displacement = result
        info = dict(model, joint_status='response_accepted', joint_accepted=True,
                    joint_final_residual=residual)
    else:
        c, cs, displacement = C, Cs, np.zeros(C.size+len(Cs))
        info = dict(joint_status='response_retained', joint_accepted=False,
            joint_response_stop=reason, joint_basis_labels=labels.copy(),
            joint_basis_dimension=basis.shape[1], joint_energy_before_hartree=float(num/norm),
            joint_energy_after_hartree=float(num/norm), joint_unit_orbital_norm=float(norm),
            joint_step_fraction=0., joint_orbital_step=0., joint_wavefunction_distance=0.,
            joint_final_residual=incoming)
    info.update(joint_spin_response=True, joint_response_rounds=rounds,
                joint_response_backtracking=searches, joint_original_basis_labels=raw_labels,
                joint_block_basis_initial_dimension=rounds[0]['model']['joint_basis_dimension'],
                joint_added_directions=len(rounds)-1, joint_direction_limit=10,
                joint_direction_cutoff=direction_cutoff, joint_block_direction_cutoff=1e-10,
                joint_norm_floor=norm_floor, joint_trust_radius=trust_radius,
                joint_incoming_residual=incoming, joint_spin_l2_bound=spin_bound,
                joint_gradient_evaluations=evaluations, joint_rejected_trials=rejected,
                joint_rejection_counts=counts, joint_seconds=time.perf_counter()-started)
    return c, cs, info, displacement
