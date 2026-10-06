"""Joint-only controlled trials of post-trial spin-residual enrichment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--problem',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--problem-sha256',required=True)
    parser.add_argument('--state-sha256',required=True)
    parser.add_argument('--expected-energy',type=float,required=True)
    parser.add_argument('--norm-floor',type=float,required=True)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--device',choices=('cpu','gpu'),default='gpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.steps<1 or not np.isfinite(args.norm_floor) or args.norm_floor<=0:
        parser.error('steps and fixed norm floor must be positive')
    if not np.isfinite(args.expected_energy):parser.error('expected energy must be finite')
    for path,expected in [(args.problem,args.problem_sha256),(args.state,args.state_sha256)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise AssertionError(f'Input checksum mismatch: {path}')
    sys.path.insert(0,str(args.code_root))
    from NonorthgonalEnergyJAX import SCWF_Energy
    from AlternatingOptimization import normalized_columns
    from Spin_Function.SpinFuncDege import spin_function_dim
    from CoupledOptimization import direction_basis,correction,adaptive_correction,_correction_in_basis
    from hpc.probe_joint_correction import measure,run_trial

    with np.load(args.problem) as problem,np.load(args.state) as saved:
        C,Cs=saved['C'].copy(),saved['Cs'].copy()
        s=SCWF_Energy.__new__(SCWF_Energy)
        s.S,s.Hcore,s.JK=problem['S'],problem['h1'],problem['eri']
        s.Ne,s.Sm,s.fs=C.shape[1],1,len(Cs)
        s.energy_nuc,s.inactive_energy=float(problem['ecore']),0.
    if s.Ne%2 or spin_function_dim(s.Ne,1,0)!=s.fs:
        raise ValueError('This probe requires a compatible saved singlet state')
    s._determinant_evaluator=None;s._initialize_jax(args.device,'determinant')
    started=time.perf_counter()
    initial,_=measure(s,C,Cs)
    if abs(initial['trial_total_energy_hartree']-args.expected_energy)>1e-8:
        raise AssertionError('Saved-state energy was not reproduced')
    C,_=normalized_columns(C,s.S);Cs/=np.linalg.norm(Cs)
    if initial['unit_orbital_norm_squared']<args.norm_floor:
        raise ValueError('Fixed norm floor exceeds initial unit-column norm')
    args.output.mkdir(parents=True,exist_ok=True)
    basis,labels=direction_basis(s,C,Cs,spin_cutoff=1e-12,max_directions=6)
    np.savez_compressed(args.output/'shared-initial-basis.npz',C=C,Cs=Cs,basis=basis,
                        labels=np.asarray(labels,dtype=str))
    cases=[]
    for radius in (.02,.001):
        for method in ['legacy','adaptive']:
            settings=dict(trust_radius=radius,max_directions=6,direction_cutoff=1e-12)
            if method=='adaptive':
                settings.update(max_added_directions=4,residual_direction_cutoff=1e-10,
                                spin_growth_limit=2.)
            cases.append((f'{method}-radius-{radius:g}',method,settings))
    config=dict(initial=initial,state=str(args.state),state_sha256=args.state_sha256,
        problem=str(args.problem),problem_sha256=args.problem_sha256,norm_floor=args.norm_floor,
        shared_initial_basis_labels=labels,shared_initial_basis_dimension=basis.shape[1],
        shared_initial_metric_padding=10,spin_overlap_cutoff=1e-12,joint_metric_relative_cutoff=1e-12,
        gradient_tolerance=1e-6,normalization_tolerance=1e-8,max_steps=args.steps,
        alternating_sweeps=0,separate_orbital_updates=0,separate_spin_updates=0,
        cases=[dict(name=n,method=m,settings=setting) for n,m,setting in cases],
        adaptive_guard='Energy decreases; full physical gradient L2 does not increase by more than 1e-8; spin gradient L2 <= max(1e-6, 2*incoming spin L2)',
        corrections_re_solved_at_same_base=True,trust_radius_shrinking=False)
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    print('CONFIG',json.dumps(config),flush=True)
    outcomes={}
    for name,method,settings in cases:
        destination=args.output/name
        iteration=0;trial_rows=[]
        def dispatch(system,c,cs,**options):
            nonlocal iteration
            iteration+=1
            if method=='legacy':
                if iteration==1:
                    # Match adaptive round zero exactly, including the JAX metric
                    # shape, without rerunning its sensitive spin-response solve.
                    base,_=normalized_columns(c,system.S);spin=cs/np.linalg.norm(cs)
                    return _correction_in_basis(system,base,spin,basis.copy(),labels.copy(),
                        norm_floor=options['norm_floor'],max_directions=6,
                        trust_radius=options['trust_radius'],direction_cutoff=options['direction_cutoff'],
                        padding=10,basis_gradient_evaluations=0)
                return correction(system,c,cs,**options)
            def record(a,b,row):
                measured,state=measure(system,a,b)
                trials=destination/'trials';trials.mkdir(parents=True,exist_ok=True)
                filename=f'step-{iteration:03d}-round-{row["round"]:02d}.npz'
                np.savez_compressed(trials/filename,C=state[0],Cs=state[1],physical_gradient=state[2])
                details=dict(step=iteration,file=str(Path('trials')/filename),after=measured,**row)
                trial_rows.append(details)
                (destination/'trial-history.json').write_text(json.dumps(trial_rows,indent=2)+'\n')
                print('ADAPTIVE_TRIAL',json.dumps(dict(case=name,step=iteration,round=row['round'],
                    basis_dimension=row['model']['joint_basis_dimension'],guard=row['residual_guard_passed'],
                    after=measured)),flush=True)
            return adaptive_correction(system,c,cs,**options,trial_recorder=record,
                initial_basis=(basis,labels) if iteration==1 else None)
        outcomes[name]=run_trial(s,C,Cs,settings,norm_floor=args.norm_floor,steps=args.steps,
            destination=destination,correct=dispatch)
        (args.output/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    comparisons=[]
    for radius in (.02,.001):
        legacy=args.output/f'legacy-radius-{radius:g}'
        adaptive=args.output/f'adaptive-radius-{radius:g}'
        with np.load(legacy/'state-001.npz') as old,np.load(adaptive/'trials/step-001-round-00.npz') as new:
            c_error=float(np.max(np.abs(old['C']-new['C'])))
            cs_error=float(np.max(np.abs(old['Cs']-new['Cs'])))
            if c_error>1e-10 or cs_error>1e-10:
                raise AssertionError('Shared six-direction control and adaptive round zero differ')
        comparisons.append(dict(radius=radius,max_C_difference=c_error,max_Cs_difference=cs_error,passed=True))
    summary=dict(diagnostic_completed=True,scientific_convergence=any(r['converged'] for r in outcomes.values()),
                 elapsed_seconds=time.perf_counter()-started,shared_prefix_checks=comparisons,outcomes=outcomes)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DIAGNOSTIC_COMPLETED',json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
