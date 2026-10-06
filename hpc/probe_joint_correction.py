"""Compare joint corrections from one immutable saved singlet SCGVB state.

No alternating sweep, orbital eigensolve or separate spin update is performed.
The projected spin solve inside direction_basis only proposes a joint direction.
These are bounded diagnostic trials, not a converged calculation result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def experiment_plan():
    """Change one setting at a time, relative to the legacy six-direction step."""
    baseline = dict(trust_radius=.02, max_directions=6, direction_cutoff=1e-12)
    cases = [('baseline', baseline)]
    for radius in (.005, .001, .0002, .00002, .000002):
        cases.append((f'radius-{radius:g}', dict(baseline, trust_radius=radius)))
    for count in (3, 4, 5):
        cases.append((f'directions-{count}', dict(baseline, max_directions=count)))
    for cutoff in (1e-8, 1e-16):
        cases.append((f'direction-cutoff-{cutoff:g}', dict(baseline, direction_cutoff=cutoff)))
    return cases


def measure(system, C, Cs):
    """Evaluate every gradient at the same unit-wavefunction normalization."""
    x = system.normalized_candidate(np.r_[C.ravel(), Cs])
    c, cs = system.split_x(x)
    gradient = system.rayleigh_gradient(x)
    num, norm = system.calcEnergy(c, cs)
    orbital, spin = gradient[:c.size], gradient[c.size:]
    row = dict(trial_total_energy_hartree=float(num/norm+system.energy_nuc),
               physical_norm_squared=float(norm),
               unit_orbital_norm_squared=float(system.unit_orbital_wavefunction_norm(x)),
               orbital_gradient_max_abs=float(np.max(np.abs(orbital))),
               spin_gradient_max_abs=float(np.max(np.abs(spin))),
               physical_gradient_max_abs=float(np.max(np.abs(gradient))),
               physical_gradient_l2=float(np.linalg.norm(gradient)),
               spin_gradient_l2=float(np.linalg.norm(spin)),
               dominant_spin_gradient_index=int(np.argmax(np.abs(spin))))
    row['converged'] = bool(row['physical_gradient_max_abs'] <= 1e-6
                            and abs(norm-1.) < 1e-8)
    return row, (c, cs, gradient)


def run_trial(system, incoming_C, incoming_Cs, settings, *, norm_floor,
              steps, destination, spin_cutoff=1e-12, correct=None):
    """Only correction may advance the state; a single floor spans all steps."""
    if correct is None:
        from CoupledOptimization import correction
        correct = correction
    destination.mkdir(parents=True, exist_ok=True)
    C, Cs = incoming_C.copy(), incoming_Cs.copy()
    initial, state = measure(system, C, Cs)
    np.savez_compressed(destination/'initial.npz', C=state[0], Cs=state[1],
                        physical_gradient=state[2])
    previous, rows, stop = None, [], 'step_limit'
    for iteration in range(1, steps+1):
        before, _ = measure(system, C, Cs)
        if before['converged']:
            stop = 'physical_convergence'; break
        started = time.perf_counter()
        c, cs, info, displacement = correct(system, C, Cs, norm_floor=norm_floor,
                                           spin_cutoff=spin_cutoff, previous=previous,
                                           **settings)
        after, state = measure(system, c, cs)
        if (not np.isfinite(list(after.values())).all()
                or abs(after['physical_norm_squared']-1.) > 1e-8
                or after['unit_orbital_norm_squared'] < norm_floor*(1-1e-8)
                or after['trial_total_energy_hartree'] > before['trial_total_energy_hartree']+1e-8):
            raise AssertionError('Joint-only trial violated energy, normalization or its fixed floor')
        row = dict(iteration=iteration, before=before, after=after, joint=info,
                   elapsed_seconds=time.perf_counter()-started,
                   energy_change_hartree=after['trial_total_energy_hartree']-before['trial_total_energy_hartree'])
        rows.append(row)
        np.savez_compressed(destination/f'state-{iteration:03d}.npz', C=state[0], Cs=state[1],
                            physical_gradient=state[2], previous_joint_step=displacement)
        (destination/'history.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('JOINT_ONLY', json.dumps(dict(case=destination.name, **row)), flush=True)
        if not info['joint_accepted']:
            stop = 'retained'; break
        C, Cs, previous = c, cs, displacement.copy()
        if after['converged']:
            stop = 'physical_convergence'; break
    final = rows[-1]['after'] if rows else initial
    outcome = dict(settings=settings, norm_floor=norm_floor, initial=initial, final=final,
                   completed_steps=len(rows), stop=stop, converged=final['converged'],
                   alternating_sweeps=0, separate_orbital_updates=0, separate_spin_updates=0)
    (destination/'outcome.json').write_text(json.dumps(outcome, indent=2)+'\n')
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--problem', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--state-sha256', required=True)
    parser.add_argument('--problem-sha256', required=True)
    parser.add_argument('--expected-energy', type=float, required=True)
    parser.add_argument('--norm-floor', type=float, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or not np.isfinite(args.norm_floor) or args.norm_floor <= 0:
        parser.error('steps and fixed norm floor must be positive')
    if not np.isfinite(args.expected_energy):
        parser.error('expected energy must be finite')
    for path, expected in ((args.state, args.state_sha256), (args.problem, args.problem_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Input checksum mismatch: {path}')
    sys.path.insert(0, str(args.code_root))
    from NonorthgonalEnergyJAX import SCWF_Energy
    from AlternatingOptimization import normalized_columns
    from Spin_Function.SpinFuncDege import spin_function_dim
    with np.load(args.problem) as problem, np.load(args.state) as saved:
        C, Cs = saved['C'].copy(), saved['Cs'].copy()
        system = SCWF_Energy.__new__(SCWF_Energy)
        system.S, system.Hcore, system.JK = problem['S'], problem['h1'], problem['eri']
        system.Ne, system.Sm, system.fs = C.shape[1], 1, len(Cs)
        system.energy_nuc, system.inactive_energy = float(problem['ecore']), 0.
    if system.Ne % 2 or spin_function_dim(system.Ne, 1, 0) != system.fs:
        raise ValueError('This saved-state probe requires a compatible singlet spin basis')
    system._determinant_evaluator = None
    system._initialize_jax(args.device, 'determinant')
    started = time.perf_counter()
    initial, _ = measure(system, C, Cs)
    if abs(initial['trial_total_energy_hartree']-args.expected_energy) > 1e-8:
        raise AssertionError('Saved-state energy was not reproduced')
    C, _ = normalized_columns(C, system.S)
    Cs /= np.linalg.norm(Cs)
    if initial['unit_orbital_norm_squared'] < args.norm_floor:
        raise ValueError('Fixed norm floor exceeds the incoming unit-column norm')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(state=str(args.state), state_sha256=args.state_sha256,
                  problem=str(args.problem), problem_sha256=args.problem_sha256,
                  initial=initial, norm_floor=args.norm_floor,
                  spin_overlap_cutoff=1e-12, joint_metric_relative_cutoff=1e-12,
                  gradient_tolerance=1e-6, normalization_tolerance=1e-8,
                  continuation_steps=args.steps, alternating_sweeps=0,
                  one_step_plan=[dict(name=n, **s) for n, s in experiment_plan()],
                  continuation_cases=['baseline', 'radius-0.001', 'radius-2e-06'],
                  baseline_has_previous_history=False)
    (args.output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    print('INITIAL', json.dumps(config), flush=True)
    outcomes = {}
    for name, settings in experiment_plan():
        outcomes[name] = run_trial(system, C, Cs, settings, norm_floor=args.norm_floor,
                                  steps=1, destination=args.output/'one-step'/name)
        (args.output/'one-step-results.json').write_text(json.dumps(outcomes, indent=2)+'\n')
    continuations = {}
    for name, settings in experiment_plan():
        if name not in config['continuation_cases']:
            continue
        continuations[name] = run_trial(system, C, Cs, settings, norm_floor=args.norm_floor,
                                       steps=args.steps, destination=args.output/'joint-only'/name)
        (args.output/'joint-only-results.json').write_text(json.dumps(continuations, indent=2)+'\n')
    summary = dict(diagnostic_completed=True, scientific_convergence=any(
        row['converged'] for row in continuations.values()),
        elapsed_seconds=time.perf_counter()-started, one_step=outcomes, joint_only=continuations)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('DIAGNOSTIC_COMPLETED', json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
