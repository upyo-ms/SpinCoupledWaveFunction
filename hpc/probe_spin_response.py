"""Joint-only comparison of common damping and orbital-damped spin response."""
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
    from scgvb.energy.jax_backend import SCWF_Energy
    from scgvb.optimization.alternating import normalized_columns
    from scgvb.spin.dimensions import spin_function_dim
    from scgvb.optimization.coupled import direction_basis,adaptive_correction
    from scgvb.optimization.spin_response import response_correction,split_joint_basis
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
    block,block_labels,no=split_joint_basis(basis,C.size)
    np.savez_compressed(args.output/'shared-block-basis.npz',basis=block,labels=np.asarray(block_labels,dtype=str),orbital_count=no)
    projection_error=float(np.max(np.abs(basis-block @ (block.T @ basis))))
    if projection_error>1e-10:raise AssertionError('Block basis lost an original direction')
    cases=[]
    for radius in (.02,.001):
        for method in ['adaptive','response']:
            settings=dict(trust_radius=radius,max_directions=6,direction_cutoff=1e-12)
            if method in ('adaptive','response'):
                settings.update(max_added_directions=4,residual_direction_cutoff=1e-10,
                                spin_growth_limit=2.)
            cases.append((f'{method}-radius-{radius:g}',method,settings))
    config=dict(initial=initial,state=str(args.state),state_sha256=args.state_sha256,
        problem=str(args.problem),problem_sha256=args.problem_sha256,norm_floor=args.norm_floor,
        shared_initial_basis_labels=labels,shared_initial_basis_dimension=basis.shape[1],
        block_basis_dimension=block.shape[1],block_orbital_dimension=no,block_span_error=projection_error,
        shared_initial_metric_padding=10,spin_overlap_cutoff=1e-12,joint_metric_relative_cutoff=1e-12,
        gradient_tolerance=1e-6,normalization_tolerance=1e-8,max_steps=args.steps,
        alternating_sweeps=0,separate_orbital_updates=0,separate_spin_updates=0,
        cases=[dict(name=n,method=m,settings=setting) for n,m,setting in cases],
        adaptive_guard='Energy decreases; full physical gradient L2 does not increase by more than 1e-8; spin gradient L2 <= max(1e-6, 2*incoming spin L2)',
        corrections_re_solved_at_same_base=True,
        response_orbital_backtracking=True,response_spin_damping=0.,
        response_trust_metric='Physical metric pulled back through orbital-induced spin response; affine spin offset excluded from orbital radius',
        response_backtracking='Enrich first; if residual guard still fails, shrink orbital coordinates and recompute undamped spin response at each fraction',
        block_direction_cutoff=1e-10,max_total_block_directions=10)
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    print('CONFIG',json.dumps(config),flush=True)
    outcomes={}
    for name,method,settings in cases:
        destination=args.output/name
        iteration=0;trial_rows=[]
        def dispatch(system,c,cs,**options):
            nonlocal iteration
            iteration+=1
            def record(a,b,row):
                measured,state=measure(system,a,b)
                trials=destination/'trials';trials.mkdir(parents=True,exist_ok=True)
                filename=f'step-{iteration:03d}-trial-{len(trial_rows):03d}.npz'
                np.savez_compressed(trials/filename,C=state[0],Cs=state[1],physical_gradient=state[2])
                details=dict(step=iteration,file=str(Path('trials')/filename),after=measured,**row)
                trial_rows.append(details)
                (destination/'trial-history.json').write_text(json.dumps(trial_rows,indent=2)+'\n')
                print('JOINT_TRIAL',json.dumps(dict(case=name,step=iteration,round=row['round'],
                    stage=row.get('stage','enrichment'),
                    basis_dimension=row.get('model',{}).get('joint_basis_dimension'),
                    guard=row.get('residual_guard_passed',row.get('passed')),
                    after=measured)),flush=True)
            corrector=adaptive_correction if method=='adaptive' else response_correction
            return corrector(system,c,cs,**options,trial_recorder=record,
                initial_basis=(basis,labels) if iteration==1 else None)
        outcomes[name]=run_trial(s,C,Cs,settings,norm_floor=args.norm_floor,steps=args.steps,
            destination=destination,correct=dispatch)
        (args.output/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    summary=dict(diagnostic_completed=True,scientific_convergence=any(r['converged'] for r in outcomes.values()),
                 elapsed_seconds=time.perf_counter()-started,block_span_error=projection_error,outcomes=outcomes)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DIAGNOSTIC_COMPLETED',json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
