"""Single-device SCGVB benchmark: compile time, synchronized runtime, memory.

Default: synthetic 12e/12o Hamiltonian. --xyz selects a real planar pi molecule.
No SCGVB optimization or independently optimized CI coefficients are involved.
"""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import sys
import time
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','gpu'), default='gpu')
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument('--xyz', type=Path)
    geometry.add_argument('--synthetic-orbitals', type=int, choices=(4,8,10,12,14), default=12)
    parser.add_argument('--determinant-size-profile', choices=('standard','large14'), default='standard')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--pi-normal-mode', choices=('global','local'), default='global')
    parser.add_argument('--pi-electrons', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--verify', action='store_true', help='Compare energy and directional derivative with independent NumPy/PySCF')
    parser.add_argument('--output', type=Path, default=Path('benchmark_12pi.json'))
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    from scgvb.energy.jax_backend import SCWF_Energy
    import jax
    # Resolve explicitly before any expensive setup; GPU failure never falls back.
    device = jax.devices(args.device)[0]
    setup_start = time.perf_counter()
    if args.xyz:
        from scgvb.cli.molecule import build_molecule, read_xyz
        from scgvb.orbitals.active_space import prepare_active_space
        from scgvb.energy.numpy_backend import SCWF_Energy as NumPyEnergy
        mol = build_molecule(read_xyz(args.xyz), basis=args.basis)
        space = prepare_active_space(mol, active_space='pi', pi_electrons=args.pi_electrons, pi_normal_mode=args.pi_normal_mode)
        reference = NumPyEnergy(mol, compute_fci=False, active_space=space, energy_evaluation='determinant', determinant_size_profile=args.determinant_size_profile)
        C, _ = space.initial_orbitals('hf', 40, .02, reference.Hcore)
        x = np.r_[C.ravel(), np.ones(reference.fs)]
        description = {'xyz': str(args.xyz), 'basis': args.basis, 'active_space': space.metadata}
    else:
        from scgvb.benchmarks.problems import synthetic_system
        reference, x = synthetic_system(args.synthetic_orbitals)
        description = {'synthetic_orbitals': args.synthetic_orbitals, 'seed': 23}
    reference.determinant_size_profile = args.determinant_size_profile
    system = SCWF_Energy.__new__(SCWF_Energy)
    system.__dict__.update(reference.__dict__)
    system._initialize_jax(args.device, 'determinant')
    data = jax.device_put(x, device)
    setup_seconds = time.perf_counter()-setup_start
    start = time.perf_counter()
    compiled = system._stable_rayleigh_vg.lower(data).compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    value, gradient = jax.block_until_ready(compiled(data))
    first_seconds = time.perf_counter()-start
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(compiled(data))
        samples.append(time.perf_counter()-start)
    device_memory = device.memory_stats()
    memory = compiled.memory_analysis()
    fields = ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
              'alias_size_in_bytes','generated_code_size_in_bytes')
    compiled_memory = None if memory is None else {f: int(getattr(memory, f)) for f in fields if hasattr(memory,f)}
    verification = None
    if args.verify:
        e, norm = reference.calcEnergy(*reference.split_x(x))
        direction = np.random.default_rng(5).normal(size=x.size)
        direction /= np.linalg.norm(direction)
        z = x.astype(complex)+1e-20j*direction
        cut = len(reference.S)*reference.Ne
        ec, nc = reference.calcEnergy(z[:cut].reshape(len(reference.S),reference.Ne), z[cut:])
        numerical = (ec/nc).imag/1e-20
        directional = np.asarray(gradient) @ direction
        np.testing.assert_allclose(float(value), e/norm, atol=1e-9, rtol=1e-10)
        np.testing.assert_allclose(directional, numerical, atol=1e-8, rtol=1e-8)
        verification = {'energy_error_hartree': float(value-e/norm),
                        'directional_gradient_error': float(directional-numerical)}
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Non-finite benchmark result')
    report = {'determinant_size_profile': args.determinant_size_profile, 'problem': description, 'device': str(device), 'device_kind': device.device_kind,
              'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
              'versions': {name: importlib.metadata.version(name) for name in ('numpy','scipy','pyscf','jax','jaxlib')},
              'float64': bool(jax.config.x64_enabled), 'active_electrons': system.Ne,
              'orbital_dimension': len(system.S), 'spin_functions': system.fs,
              'setup_seconds': setup_seconds, 'compile_seconds': compile_seconds,
              'first_execution_seconds': first_seconds, 'warm_execution_seconds': samples,
              'energy_hartree_active': float(value), 'gradient_max_abs': float(np.max(np.abs(gradient))),
              'compiled_memory_analysis': compiled_memory, 'device_allocator_memory_stats': device_memory,
              'process_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
              'verification': verification}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
