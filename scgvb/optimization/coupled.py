"""Opt-in, bounded orbital/spin Newton corrections in small tangent subspaces.

The retracted Rayleigh energy is differentiated by central differences of its
physical gradient. Only a small Hessian is formed; determinant amplitudes remain
derived SCGVB quantities. The physical tangent metric includes cross terms.
"""
import time
import numpy as np
from scipy.linalg import eigh
from scgvb.optimization.alternating import normalized_columns


def retract(C, Cs, basis, coordinates, overlap):
    """Unit-column and unit-spin retraction and its gradient pullback data."""
    raw = np.r_[C.ravel(), Cs] + basis @ coordinates
    orbital = raw[:C.size].reshape(C.shape)
    c, scales = normalized_columns(orbital, overlap)
    spin_scale = np.linalg.norm(raw[C.size:])
    if not np.isfinite(spin_scale) or spin_scale <= 0:
        raise FloatingPointError('Invalid coupled spin retraction')
    return c, raw[C.size:]/spin_scale, scales, spin_scale


def reduced_gradient(system, C, Cs, basis, coordinates):
    c, cs, scales, spin_scale = retract(C, Cs, basis, coordinates, system.S)
    gradient = system.rayleigh_gradient(np.r_[c.ravel(), cs])
    gc = gradient[:C.size].reshape(C.shape)
    gc = (gc - (system.S @ c)*np.sum(c*gc, axis=0)) / scales
    gs = gradient[C.size:]
    gs = (gs-cs*np.dot(cs, gs))/spin_scale
    return basis.T @ np.r_[gc.ravel(), gs]


def reduced_hessian(system, C, Cs, basis, metric, step=1e-5):
    """Two full gradient evaluations per direction, with small physical probes."""
    count = basis.shape[1]
    matrix = np.empty((count, count))
    steps = np.minimum(step, 1e-4/np.sqrt(np.maximum(np.diag(metric), 1e-30)))
    for j, h in enumerate(steps):
        z = np.zeros(count); z[j] = h
        matrix[:, j] = (reduced_gradient(system, C, Cs, basis, z)
                        - reduced_gradient(system, C, Cs, basis, -z))/(2*h)
    asymmetry = np.linalg.norm(matrix-matrix.T)/max(np.linalg.norm(matrix), 1e-30)
    return .5*(matrix+matrix.T), float(asymmetry), steps


def direction_basis(system, C, Cs, *, spin_cutoff, previous=None, max_directions=6,
                    direction_cutoff=1e-12):
    """Pure orbital/spin directions plus a previous accepted joint displacement."""
    if not np.isfinite(direction_cutoff) or not 0 < direction_cutoff < 1:
        raise ValueError('Joint direction cutoff must be finite and between 0 and 1')
    from scgvb.optimization.wavefunction import projected_spin_solve
    w, v = eigh(system.S)
    if w[0] <= 0:
        raise FloatingPointError('Invalid coupled orbital overlap')
    root, invroot = (v*np.sqrt(w)) @ v.T, (v/np.sqrt(w)) @ v.T
    d = root @ C
    gradient = system.rayleigh_gradient(np.r_[C.ravel(), Cs])
    orbital = invroot @ gradient[:C.size].reshape(C.shape)
    orbital -= d*np.sum(d*orbital, axis=0)
    spin = gradient[C.size:]-Cs*np.dot(Cs, gradient[C.size:])
    directions, labels = [], []

    def add(dc, ds, label):
        # Tangent projection removes radial orbital and spin scale freedoms.
        dc = root @ dc; dc -= d*np.sum(d*dc, axis=0)
        ds = ds-Cs*np.dot(Cs, ds)
        vector = np.r_[(invroot @ dc).ravel(), ds]
        for old in directions:
            vector -= old*np.dot(old, vector)
        size = np.linalg.norm(vector)
        if np.isfinite(size) and size > direction_cutoff and len(directions) < max_directions:
            directions.append(vector/size); labels.append(label)

    add(-invroot @ orbital, np.zeros_like(Cs), 'orbital_gradient')
    proposed, _ = projected_spin_solve(system, C, Cs, spin_cutoff)
    if np.dot(proposed, Cs) < 0: proposed = -proposed
    add(np.zeros_like(C), proposed-Cs, 'spin_response')
    add(np.zeros_like(C), -spin, 'spin_gradient')
    if previous is not None:
        add(previous[:C.size].reshape(C.shape), previous[C.size:], 'previous_joint_step')
    for j in np.argsort(np.linalg.norm(orbital, axis=0))[::-1]:
        delta = np.zeros_like(C); delta[:, j] = -(invroot @ orbital)[:, j]
        add(delta, np.zeros_like(Cs), f'orbital_{j+1}_gradient')
        if len(directions) == max_directions: break
    return np.column_stack(directions) if directions else np.zeros((C.size+len(Cs), 0)), labels


def correction(system, C, Cs, *, norm_floor, spin_cutoff=1e-12,
               previous=None, max_directions=6, trust_radius=.02,
               direction_cutoff=1e-12):
    """Return a safe joint state or retain the incoming state, never convergence."""
    if not np.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('Joint correction requires a positive fixed norm floor')
    if type(max_directions) is not int or not 2 <= max_directions <= 6:
        raise ValueError('Joint correction supports 2 to 6 directions')
    if not np.isfinite(trust_radius) or not 0 < trust_radius <= .05:
        raise ValueError('Joint trust radius must be positive and at most 0.05')
    if not np.isfinite(direction_cutoff) or not 0 < direction_cutoff < 1:
        raise ValueError('Joint direction cutoff must be finite and between 0 and 1')
    if getattr(system, 'energy_evaluation', None) != 'determinant':
        raise ValueError('Joint correction requires determinant energy evaluation')
    started = time.perf_counter()
    C, _ = normalized_columns(np.asarray(C, dtype=float), system.S)
    Cs = np.asarray(Cs, dtype=float).copy(); Cs /= np.linalg.norm(Cs)
    num, norm = system.calcEnergy(C, Cs)
    if not np.isfinite(num/norm) or norm < norm_floor:
        raise FloatingPointError('Joint correction starts from an infeasible state')
    basis, labels = direction_basis(system, C, Cs, spin_cutoff=spin_cutoff,
                                    previous=previous, max_directions=max_directions,
                                    direction_cutoff=direction_cutoff)
    return _correction_in_basis(system, C, Cs, basis, labels, norm_floor=norm_floor,
                                max_directions=max_directions, trust_radius=trust_radius,
                                direction_cutoff=direction_cutoff, started_at=started)


def _correction_in_basis(system, C, Cs, basis, labels, *, norm_floor,
                         max_directions, trust_radius, direction_cutoff,
                         padding=6, basis_gradient_evaluations=1, started_at=None):
    """Solve at a fixed unit-column base; added directions never rebase the step."""
    start = time.perf_counter() if started_at is None else started_at
    num, norm = system.calcEnergy(C, Cs); before = float(num/norm)
    if not np.isfinite(before) or norm < norm_floor:
        raise FloatingPointError('Joint correction starts from an infeasible state')
    info = dict(joint_status='retained', joint_accepted=False, joint_direction_limit=max_directions,
                joint_direction_cutoff=float(direction_cutoff), joint_trust_radius=float(trust_radius),
                joint_norm_floor=float(norm_floor), joint_energy_before_hartree=before,
                joint_gradient_evaluations=basis_gradient_evaluations, joint_rejected_trials=0)
    displacement = np.zeros(C.size+len(Cs))

    def finish(c=C, cs=Cs):
        info['joint_seconds'] = time.perf_counter()-start
        return c, cs, info, displacement

    count = basis.shape[1]; info.update(joint_basis_labels=labels, joint_basis_dimension=count)
    if not count:
        info['joint_status'] = 'no_tangent_direction'; return finish()
    # Fixed padding within a method avoids compiling for every subspace size.
    if not count <= padding <= 10:
        raise ValueError("Invalid joint metric padding")
    padded = np.pad(basis, ((0, 0), (0, padding-count)))
    metric = system.joint_wavefunction_metric(C, Cs, padded)[:count, :count]
    metric = .5*(metric+metric.T)
    values, vectors = eigh(metric)
    info['joint_metric_eigenvalues'] = values.tolist()
    if not np.isfinite(values).all() or values[-1] <= 0 or values[0] < -1e-8*values[-1]:
        info['joint_status'] = 'unresolved_metric'; return finish()
    keep = values > max(1e-12*values[-1], np.finfo(float).tiny)
    transform = vectors[:, keep]/np.sqrt(values[keep])
    info['joint_metric_rank'] = int(keep.sum())
    g = reduced_gradient(system, C, Cs, basis, np.zeros(count))
    hessian, asymmetry, steps = reduced_hessian(system, C, Cs, basis, metric)
    info.update(joint_gradient_evaluations=basis_gradient_evaluations+1+2*count, joint_hessian_asymmetry=asymmetry,
                joint_finite_difference_steps=steps.tolist(), joint_reduced_gradient=g.tolist(),
                joint_reduced_hessian=hessian.tolist(), joint_reduced_metric=metric.tolist())
    if not np.isfinite(hessian).all() or not np.isfinite(g).all() or asymmetry > .05:
        info['joint_status'] = 'unreliable_curvature'; return finish()
    reduced = transform.T @ hessian @ transform
    hg = transform.T @ g
    curvature, rotation = eigh(.5*(reduced+reduced.T))
    info['joint_curvature_eigenvalues'] = curvature.tolist()
    shift = max(0., -curvature[0]) + 1e-6*max(1., np.max(np.abs(curvature)))
    for _ in range(60):
        step = -rotation @ ((rotation.T @ hg)/(curvature+shift))
        if np.linalg.norm(step) <= trust_radius: break
        shift = max(2*shift, 1e-6)
    coordinate_step = transform @ step
    slope = float(g @ coordinate_step)
    info.update(joint_level_shift=float(shift), joint_predicted_slope=slope,
                joint_predicted_wavefunction_distance=float(np.linalg.norm(step)))
    if not np.isfinite(slope) or slope >= 0:
        info['joint_status'] = 'no_resolved_descent'; return finish()
    reasons = dict(norm=0, energy=0, nonfinite=0, wavefunction=0, orbital_step=0)
    fraction = 1.
    for _ in range(30):
        reason = 'nonfinite'
        try:
            c, cs, _, _ = retract(C, Cs, basis, fraction*coordinate_step, system.S)
            delta = c-C
            orbital_step = float(np.max(np.sqrt(np.maximum(0., np.einsum('ia,ij,ja->a',delta,system.S,delta)))))
            n, den = system.calcEnergy(c, cs); energy = float(n/den) if den > 0 else np.inf
            if not np.isfinite(energy) or not np.isfinite(den): reason = 'nonfinite'
            elif den < norm_floor: reason = 'norm'
            elif orbital_step > .1+1e-12: reason = 'orbital_step'
            elif energy > before+1e-4*fraction*slope or energy >= before: reason = 'energy'
            else:
                overlap = system.wavefunction_overlap(C, Cs, c, cs)
                distance = float(np.sqrt(max(0.,1-overlap**2)))
                if not np.isfinite(distance): reason = 'nonfinite'
                elif distance > .05+1e-12: reason = 'wavefunction'
                else:
                    displacement = np.r_[(c-C).ravel(),cs-Cs]
                    info.update(joint_status='accepted', joint_accepted=True, joint_step_fraction=fraction,
                                joint_energy_after_hartree=energy, joint_unit_orbital_norm=float(den),
                                joint_orbital_step=orbital_step, joint_wavefunction_distance=distance,
                                joint_rejection_counts=reasons)
                    return finish(c, cs)
        except (FloatingPointError, OverflowError):
            reason = 'nonfinite'
        reasons[reason] += 1; info['joint_rejected_trials'] += 1; fraction *= .5
    info.update(joint_status='line_search_retained', joint_step_fraction=0., joint_rejection_counts=reasons)
    return finish()


def _physical_residual(system, C, Cs):
    """Compare derivatives only after restoring unit physical wavefunction norm."""
    x = system.normalized_candidate(np.r_[C.ravel(), Cs])
    gradient = system.rayleigh_gradient(x)
    if not np.isfinite(gradient).all():
        raise FloatingPointError('Non-finite adaptive joint residual')
    orbital, spin = gradient[:C.size], gradient[C.size:]
    return dict(orbital_max_abs=float(np.max(np.abs(orbital))),
                spin_max_abs=float(np.max(np.abs(spin))),
                spin_l2=float(np.linalg.norm(spin)),
                full_l2=float(np.linalg.norm(gradient))), spin


def append_spin_residual(basis, C, Cs, spin_residual, *, cutoff=1e-10):
    """Add a post-trial spin derivative, projected at the unchanged base.

    Normalize before orthogonalization: cutoff measures relative independence,
    not the absolute gradient magnitude or the physical convergence tolerance.
    Subtracting existing mixed directions may introduce orbital components;
    the enlarged span still contains the original pure-spin tangent vector.
    """
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError('Residual direction cutoff must be finite and between 0 and 1')
    spin = np.asarray(spin_residual, dtype=float).copy()
    spin -= Cs*np.dot(Cs, spin)
    vector = np.r_[np.zeros(C.size), -spin]
    size = np.linalg.norm(vector)
    if not np.isfinite(size) or size == 0:
        return basis, dict(added=False, relative_independence=0.)
    vector /= size
    # Reorthogonalize so near-duplicate residuals cannot fill the small space.
    for _ in range(2):
        vector -= basis @ (basis.T @ vector)
    novelty = float(np.linalg.norm(vector))
    if novelty <= cutoff:
        return basis, dict(added=False, relative_independence=novelty)
    return np.column_stack((basis, vector/novelty)), dict(
        added=True, relative_independence=novelty)


def adaptive_correction(system, C, Cs, *, norm_floor, spin_cutoff=1e-12,
                        previous=None, max_directions=6, trust_radius=.02,
                        direction_cutoff=1e-12, max_added_directions=4,
                        residual_direction_cutoff=1e-10, spin_growth_limit=2.,
                        trial_recorder=None, initial_basis=None):
    """Expand with post-trial spin residuals and re-solve at the original base.

    At most four additions (ten total directions), with a single trust radius
    and norm floor. An energy-decreasing candidate also needs nonincreasing
    full physical gradient L2 and bounded spin-gradient growth. This experimental
    stationarity guard is separate from the unchanged convergence test.
    Legacy correction remains independently callable and unchanged by default.
    """
    if type(max_directions) is not int or not 2 <= max_directions <= 6:
        raise ValueError('Adaptive joint correction starts with 2 to 6 directions')
    if type(max_added_directions) is not int or not 1 <= max_added_directions <= 4:
        raise ValueError('Adaptive joint correction adds 1 to 4 directions')
    for name, value in [('direction cutoff', direction_cutoff),
                        ('residual direction cutoff', residual_direction_cutoff),
                        ('spin cutoff', spin_cutoff)]:
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'Adaptive joint {name} must be finite and between 0 and 1')
    if not np.isfinite(spin_growth_limit) or spin_growth_limit < 1:
        raise ValueError('Adaptive joint spin growth limit must be finite and at least one')
    if not np.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('Adaptive joint correction requires a positive fixed norm floor')
    if not np.isfinite(trust_radius) or not 0 < trust_radius <= .05:
        raise ValueError('Adaptive joint trust radius must be positive and at most 0.05')
    if getattr(system, 'energy_evaluation', None) != 'determinant':
        raise ValueError('Adaptive joint correction requires determinant energy evaluation')
    started = time.perf_counter()
    C, _ = normalized_columns(np.asarray(C, dtype=float), system.S)
    Cs = np.asarray(Cs, dtype=float).copy(); Cs /= np.linalg.norm(Cs)
    num, norm = system.calcEnergy(C, Cs)
    if not np.isfinite(num/norm) or norm < norm_floor:
        raise FloatingPointError('Adaptive joint correction starts from an infeasible state')
    incoming, _ = _physical_residual(system, C, Cs)
    spin_bound = max(1e-6, spin_growth_limit*incoming['spin_l2'])
    if initial_basis is None:
        basis, labels = direction_basis(system, C, Cs, spin_cutoff=spin_cutoff,
                                        previous=previous, max_directions=max_directions,
                                        direction_cutoff=direction_cutoff)
    else:
        # A diagnostic caller can share the same initial directions across
        # methods, eliminating projected-spin-solver restart sensitivity.
        basis, labels = np.asarray(initial_basis[0], dtype=float).copy(), list(initial_basis[1])
        if (basis.ndim != 2 or basis.shape[0] != C.size+len(Cs)
                or basis.shape[1] != len(labels) or basis.shape[1] > max_directions
                or not np.isfinite(basis).all()
                or not np.allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-10, rtol=1e-10)):
            raise ValueError('Initial adaptive basis must be finite, orthonormal and compatible')
        dc = basis[:C.size].reshape(*C.shape, basis.shape[1])
        orbital_radial = np.einsum('ia,ij,jak->ak', C, system.S, dc)
        spin_radial = Cs @ basis[C.size:]
        if np.any(np.abs(orbital_radial)>1e-10) or np.any(np.abs(spin_radial)>1e-10):
            raise ValueError('Initial adaptive basis must be tangent to the supplied state')
    rounds, evaluations, rejections, reason = [], 1+(initial_basis is None), 0, 'addition_limit'
    rejection_counts = dict(norm=0, energy=0, nonfinite=0, wavefunction=0, orbital_step=0)
    last_info = {}
    for index in range(max_added_directions+1):
        c, cs, model, displacement = _correction_in_basis(
            system, C, Cs, basis, labels.copy(), norm_floor=norm_floor,
            max_directions=max_directions+max_added_directions,
            trust_radius=trust_radius, direction_cutoff=direction_cutoff,
            padding=10, basis_gradient_evaluations=0)
        residual, spin = _physical_residual(system, c, cs)
        evaluations += model['joint_gradient_evaluations']+1
        rejections += model['joint_rejected_trials']
        for key, value in model.get('joint_rejection_counts', {}).items():
            rejection_counts[key] += value
        passes = bool(model['joint_accepted']
                      and residual['spin_l2'] <= spin_bound
                      and residual['full_l2'] <= incoming['full_l2']+1e-8)
        row = dict(round=index, model=model, physical_residual=residual,
                   residual_guard_passed=passes, added_directions=index)
        rounds.append(row); last_info = model
        if trial_recorder is not None:
            trial_recorder(c, cs, row)
        if passes:
            info = dict(model, joint_status='adaptive_accepted', joint_accepted=True,
                        joint_adaptive=True, joint_adaptive_rounds=rounds,
                        joint_added_directions=index, joint_spin_l2_bound=spin_bound,
                        joint_incoming_residual=incoming, joint_final_residual=residual,
                        joint_gradient_evaluations=evaluations, joint_rejected_trials=rejections,
                        joint_rejection_counts=rejection_counts,
                        joint_residual_rejected_candidates=index,
                        joint_seconds=time.perf_counter()-started)
            return c, cs, info, displacement
        if not model['joint_accepted']:
            reason = model['joint_status']; break
        if index == max_added_directions:
            break
        expanded, addition = append_spin_residual(
            basis, C, Cs, spin, cutoff=residual_direction_cutoff)
        row['direction_addition'] = addition
        if not addition['added']:
            reason = 'dependent_spin_residual'; break
        basis = expanded; labels.append(f'post_trial_spin_residual_{index+1}')
    # None of the candidate states was committed; report the retained state,
    # rather than copying the last rejected candidate's energy/step fields.
    num, norm = system.calcEnergy(C, Cs)
    info = dict(joint_status='adaptive_retained', joint_accepted=False, joint_adaptive=True,
                joint_adaptive_stop=reason, joint_adaptive_rounds=rounds,
                joint_added_directions=len(rounds)-1,
                joint_basis_labels=labels.copy(), joint_basis_dimension=basis.shape[1],
                joint_direction_limit=max_directions+max_added_directions,
                joint_direction_cutoff=direction_cutoff, joint_trust_radius=trust_radius,
                joint_norm_floor=norm_floor, joint_energy_before_hartree=float(num/norm),
                joint_energy_after_hartree=float(num/norm), joint_unit_orbital_norm=float(norm),
                joint_spin_l2_bound=spin_bound, joint_incoming_residual=incoming,
                joint_final_residual=incoming, joint_step_fraction=0., joint_orbital_step=0.,
                joint_wavefunction_distance=0., joint_gradient_evaluations=evaluations,
                joint_rejected_trials=rejections, joint_metric_rank=last_info.get('joint_metric_rank'),
                joint_rejection_counts=rejection_counts, joint_residual_rejected_candidates=len(rounds),
                joint_seconds=time.perf_counter()-started)
    return C, Cs, info, np.zeros(C.size+len(Cs))
