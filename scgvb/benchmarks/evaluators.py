"""Compare permutation/determinant JAX evaluators in isolated processes.

Same cc-pVDZ integrals, active subspace, orbital and spin coefficients for both.
No geometry, orbital, or core optimization: this measures evaluator cost only.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

CASES = {'allyl': (3, 2), 'cyclobutadiene': (4, 1), 'benzene': (6, 1)}
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    import jax
    from scgvb.orbitals.active_space import ActiveSpace
    from scgvb.cli.molecule import build_molecule
    from scgvb.energy.jax_backend import SCWF_Energy
    f = np.load(args.fixture, allow_pickle=False)
    mol = build_molecule(str(f['atom']), str(f['basis_name']), multiplicity=int(f['multiplicity']))
    space = ActiveSpace(f['basis'], f['inactive'], int(f['electrons']), f['occupations'], {})
    start = time.perf_counter()
    system = SCWF_Energy(mol, compute_fci=False, device=args.device,
                         energy_evaluation=args.method, active_space=space)
    if hasattr(system, '_jax_data'):
        jax.block_until_ready(system._jax_data)
    setup = time.perf_counter() - start
    x, C, Cs = f['x'], f['C'], f['Cs']

    def energy():
        return system.calcEnergy(C, Cs)  # NumPy conversion synchronizes device.

    def value_gradient():
        system._last_evaluation = None  # Never benchmark a Python cache hit.
        return system._value_gradient(x, 'rayleigh')

    start = time.perf_counter()
    numerator, norm = energy()
    first_energy = time.perf_counter() - start
    start = time.perf_counter()
    value, gradient = value_gradient()
    first_vg = time.perf_counter() - start
    for _ in range(3):
        energy()
        value_gradient()
    timings = {'energy_norm': [], 'energy_gradient': []}
    for _ in range(args.repeats):
        for name, fn in [('energy_norm', energy), ('energy_gradient', value_gradient)]:
            start = time.perf_counter()
            fn()
            timings[name].append(time.perf_counter() - start)
    stats = {name: {'median_seconds': float(np.median(v)), 'min_seconds': min(v),
                    'max_seconds': max(v), 'samples_seconds': v} for name, v in timings.items()}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.output, dict(method=args.method, device=str(system.device),
        device_kind=system.device.device_kind, python=platform.python_version(), jax=jax.__version__,
        initialization_seconds=setup, first_energy_norm_seconds=first_energy,
        first_energy_gradient_seconds_after_energy_call=first_vg,
        timings=stats, electronic_numerator_hartree=float(numerator), wavefunction_norm=float(norm),
        active_energy_hartree=float(value), gradient=np.asarray(gradient).tolist(),
        peak_process_rss_bytes=int(rss*(1 if sys.platform == 'darwin' else 1024)),
        device_memory_stats=system.device.memory_stats(),
        timing_scope='synchronized public-style host-to-device evaluation; fresh process per method; first calls include JIT'))


def prepare(case, args, folder):
    from scgvb.cli.molecule import build_molecule, read_xyz
    from scgvb.orbitals.active_space import prepare_active_space
    from scgvb.spin.dimensions import spin_function_dim
    ne, multiplicity = CASES[case]
    geometry = read_xyz(ROOT/'examples'/f'{case}.xyz')
    atom = '\n'.join(symbol+' '+' '.join(map(str, xyz)) for symbol, xyz in geometry)
    mol = build_molecule(atom, args.basis, multiplicity=multiplicity)
    start = time.perf_counter()
    space = prepare_active_space(mol, active_space='pi', pi_electrons=ne)
    hf_seconds = time.perf_counter() - start
    # A perturbed HF state exercises more than a single determinant. No solver
    # normalizes the fixture, so both methods receive exactly the same arrays.
    C, _ = space.initial_orbitals('hf', 40, .08, np.zeros((len(space.occupations),)*2))
    Cs = np.random.default_rng(40).normal(size=spin_function_dim(ne, multiplicity, 0))
    Cs /= np.linalg.norm(Cs)
    fixture = folder/'fixture.npz'
    np.savez(fixture, atom=atom, basis_name=args.basis, multiplicity=multiplicity,
             basis=space.basis, inactive=space.inactive, occupations=space.occupations,
             electrons=ne, C=C, Cs=Cs, x=np.r_[C.ravel(), Cs])
    return fixture, dict(active_electrons=ne, active_orbitals=space.basis.shape[1],
                        spin_functions=len(Cs), basis=args.basis, hf_selection_seconds=hf_seconds,
                        core_optimized=False, fixture='fixture.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=[*CASES, 'all'], default='all')
    p.add_argument('--basis', default='cc-pvdz')
    p.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--output', type=Path, required=True, help='New output directory (worker: JSON path)')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--fixture', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--method', choices=['permutation', 'determinant'], help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('repeats must be positive')
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for case in CASES if args.case == 'all' else [args.case]:
        folder = args.output/case
        folder.mkdir()
        fixture, info = prepare(case, args, folder)
        methods = {}
        for method in ('permutation', 'determinant'):
            target = folder/(method+'.json')
            command = [sys.executable, '-m', 'scgvb.benchmarks.evaluators', '--worker', '--fixture', str(fixture.resolve()),
                       '--method', method, '--device', args.device, '--repeats', str(args.repeats),
                       '--output', str(target.resolve())]
            with (folder/(method+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            methods[method] = json.loads(target.read_text())
        a, b = methods['permutation'], methods['determinant']
        checks = {}
        for key in ('electronic_numerator_hartree', 'wavefunction_norm', 'active_energy_hartree', 'gradient'):
            checks[key] = bool(np.allclose(a[key], b[key], atol=1e-7, rtol=1e-8))
        info.update(checks=checks,
                    max_gradient_difference=float(np.max(np.abs(np.array(a['gradient'])-b['gradient']))),
                    determinant_speedup={key: a['timings'][key]['median_seconds']/b['timings'][key]['median_seconds']
                                         for key in ('energy_norm', 'energy_gradient')}, methods=methods)
        results[case] = info
        write_json(args.output/'summary.json', results)
        if not all(checks.values()):
            raise RuntimeError(f'{case}: evaluator mismatch; see summary.json')
        print(case, 'agreement passed; determinant speedup:', info['determinant_speedup'], flush=True)


if __name__ == '__main__':
    main()
