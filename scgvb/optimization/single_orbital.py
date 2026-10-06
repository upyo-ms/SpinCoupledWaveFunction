"""Safeguarded exact one-column SCGVB Rayleigh solves; no independent CI variables."""
import time

import numpy as np
from scipy.linalg import eigh


def solve_orbital(system, C, Cs, index, cutoff=1e-10):
    """Solve one orbital in resolved physical support plus its incoming state.

    Whitening uses actual derived determinant vectors. Shifting H before
    projection avoids subtracting large H and E*M entries after whitening.
    The eigenvector is a proposal; the caller separately checks the original
    physical energy, norm and step bounds. All other columns and Cs are fixed.
    """
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError('Orbital overlap cutoff must be between zero and one')
    if not 0 <= index < C.shape[1]:
        raise ValueError('Invalid orbital index')
    incoming = C[:, index]
    metric = system.single_orbital_overlap(C, Cs, index)
    values, vectors = eigh(.5*(metric+metric.T))
    if (not np.isfinite(values).all() or values[-1] <= 0
            or values[0] < -1e-8*values[-1]):
        raise FloatingPointError('Invalid one-orbital multi-electron overlap')
    keep = values > cutoff*values[-1]
    num, norm = system.calcEnergy(C, Cs)
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Invalid incoming one-orbital state')
    reference = float(num/norm)
    # A fixed width avoids recompiling a GPU kernel whenever the resolved
    # rank changes. Zero columns are removed by physical whitening below.
    basis = np.zeros((len(values), len(values)+1))
    basis[:, :keep.sum()] = vectors[:, keep]/np.sqrt(values[keep])
    basis[:, -1] = incoming/np.sqrt(norm)
    h, physical_metric = system.projected_orbital_matrices(C, Cs, index, basis, reference)
    small_values, small_vectors = eigh(.5*(physical_metric+physical_metric.T))
    if (not np.isfinite(small_values).all() or small_values[-1] <= 0
            or small_values[0] < -1e-8*small_values[-1]):
        raise FloatingPointError('Invalid projected one-orbital overlap')
    resolved = small_values > max(1e-12, 100*np.finfo(float).eps*len(small_values))*small_values[-1]
    transform = small_vectors[:, resolved]/np.sqrt(small_values[resolved])
    reduced = transform.T @ h @ transform
    energies, states = eigh(.5*(reduced+reduced.T))
    proposed = basis @ transform @ states[:, 0]
    norm_s = float(proposed @ system.S @ proposed)
    if not np.isfinite(proposed).all() or not np.isfinite(norm_s) or norm_s <= 0:
        raise FloatingPointError('Invalid one-orbital solution')
    proposed /= np.sqrt(norm_s)
    # Align the physical state's phase, not the Euclidean orbital coefficients.
    trial = C.copy(); trial[:, index] = proposed
    if system.wavefunction_overlap(C, Cs, trial, Cs) < 0:
        proposed *= -1
    return proposed, dict(orbital_index=int(index), orbital_overlap_rank=int(keep.sum()),
                          orbital_subspace_dimension=int(resolved.sum()),
                          orbital_incoming_preserved=True, orbital_overlap_cutoff=float(cutoff),
                          orbital_proposed_energy_hartree=float(reference+energies[0]),
                          orbital_projected_residual_max_abs=float(np.max(np.abs(
                              reduced @ states[:, 0]-energies[0]*states[:, 0]))),
                          orbital_projected_metric_condition=float(
                              small_values[-1]/small_values[resolved][0]))


def orbital_block(system, C, Cs, *, norm_floor, max_iterations, max_step=.1,
                  progress=None, cutoff=1e-10, start_orbital=0):
    """At most max_iterations sequential proposals, with physical safeguards.

    nit counts accepted orbital updates; attempted solves are recorded
    separately. The cursor continues across micro blocks and outer cycles.
    No mutual orbital orthogonality is imposed. A bound-limited stop with a
    nonzero full gradient is never reported as convergence.
    """
    from scgvb.optimization.alternating import OrbitalResult, normalized_columns
    if not np.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError('Orbital norm floor must be positive and finite')
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError('Orbital iteration limit must be a positive integer')
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('Orbital step bound must be positive and finite')
    C, _ = normalized_columns(np.asarray(C, dtype=float), system.S)
    Cs = np.asarray(Cs, dtype=float).copy()
    num, norm = system.calcEnergy(C, Cs)
    if not np.isfinite(norm) or norm < norm_floor:
        raise FloatingPointError('Single-orbital block starts with an infeasible state')
    value = float(num/norm)
    if not np.isfinite(value):
        raise FloatingPointError('Non-finite incoming orbital energy')
    accepted, rejected, largest_step, largest_distance = 0, 0, 0., 0.
    reasons = dict(norm=0, nonfinite=0, energy=0, wavefunction=0, orbital_step=0)
    geometry = dict(orbital_solver='generalized_eigenvalue', orbital_attempted_solves=0,
                    orbital_accepted_updates=0, orbital_matrix_seconds=0.,
                    next_orbital=int(start_orbital % system.Ne), single_orbital_history=[])

    def report():
        if progress is not None:
            progress(dict(orbital_iteration=accepted, orbital_energy_hartree=value,
                          orbital_unit_norm=float(norm), orbital_norm_floor=norm_floor,
                          orbital_rejected_trials=rejected, orbital_rejection_counts=reasons.copy(),
                          orbital_update='single_orbital',
                          orbital_max_wavefunction_distance=largest_distance,
                          **geometry))

    report()
    for _ in range(max_iterations):
        index = geometry['next_orbital']
        start = time.perf_counter()
        proposed, info = solve_orbital(system, C, Cs, index, cutoff)
        geometry['orbital_matrix_seconds'] += time.perf_counter()-start
        geometry['orbital_attempted_solves'] += 1
        geometry['next_orbital'] = (index+1) % system.Ne
        old = C[:, index].copy()
        fraction, done, trials = 1., False, 0
        accepted_distance = 0.
        for _ in range(40):
            column = old+fraction*(proposed-old)
            column_norm = float(column @ system.S @ column)
            reason = 'nonfinite'
            if np.isfinite(column_norm) and column_norm > 0:
                column /= np.sqrt(column_norm)
                delta = column-old
                size = float(np.sqrt(max(0., delta @ system.S @ delta)))
                if size > max_step+1e-12:
                    reason = 'orbital_step'
                else:
                    trial = C.copy(); trial[:, index] = column
                    try:
                        n, trial_norm = system.calcEnergy(trial, Cs)
                        energy = float(n/trial_norm) if trial_norm > 0 else np.inf
                        if not np.isfinite(trial_norm) or not np.isfinite(energy):
                            reason = 'nonfinite'
                        elif trial_norm < norm_floor:
                            reason = 'norm'
                        elif energy >= value:
                            reason = 'energy'
                        else:
                            overlap = system.wavefunction_overlap(C, Cs, trial, Cs)
                            distance = float(np.sqrt(max(0., 1-overlap**2)))
                            if not np.isfinite(overlap) or distance > .05:
                                reason = 'wavefunction'
                            else:
                                C, value, norm = trial, energy, trial_norm
                                accepted += 1
                                largest_step = max(largest_step, size)
                                largest_distance = max(largest_distance, distance)
                                accepted_distance = distance
                                done = True
                                break
                    except (FloatingPointError, OverflowError):
                        reason = 'nonfinite'
            rejected += 1; trials += 1; reasons[reason] += 1
            fraction *= .5
        info.update(accepted=done, step_fraction=float(fraction if done else 0.),
                    rejected_trials=trials, active_energy_hartree=value,
                    unit_orbital_norm=float(norm), wavefunction_distance=accepted_distance)
        geometry['single_orbital_history'].append(info)
        geometry['orbital_accepted_updates'] = accepted
        report()
    gradient = system.rayleigh_gradient(np.r_[C.ravel(), Cs])[:C.size]
    if not np.isfinite(gradient).all():
        raise FloatingPointError('Non-finite final one-orbital block gradient')
    # Remove radial scale freedoms and express gradients on unit S-columns,
    # matching the coordinate convention of the existing orbital block.
    w, v = eigh(system.S)
    root = (v*np.sqrt(w)) @ v.T
    invroot = (v/np.sqrt(w)) @ v.T
    d, g = root @ C, invroot @ gradient.reshape(C.shape)
    g -= d*np.sum(d*g, axis=0)
    gmax = float(np.max(np.abs(g)))
    status = 0 if gmax <= 1e-8 else (1 if accepted else 2)
    return OrbitalResult(C, accepted, status, float(norm), gmax, norm_floor,
                         rejected, largest_step, reasons, geometry, largest_distance)
