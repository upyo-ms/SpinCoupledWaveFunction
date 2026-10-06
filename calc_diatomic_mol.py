import numpy as np
import argparse
import json
from pathlib import Path
import warnings
from functools import partial
import matplotlib.pyplot as plt
from pyscf import gto

verbose = False


def generateMolecule(R):
    mol = gto.Mole()
    mol.charge = 1 
    mol.spin = 1
    mol.build(atom='He 0 0 0; He 0 0 {}'.format(R), basis='cc-pvdz')
    if not verbose:
        mol.verbose = 2
    return mol

def optimize_geometry(system, C, Cs, *, verbose=False, allow_fallback=True):
    """Retry the incoming state with an exact determinant evaluator on failure.

    No failed optimizer iterate is propagated to the next geometry. Retain the
    reference FCI energy, and let the caller carry the successful evaluator on.
    """
    try:
        energy, orbitals, spin = system.augmentedLagrangian(C.copy(), Cs.copy(), verbose=verbose)
        return energy, orbitals, spin, system
    except RuntimeError as exc:
        if not allow_fallback or system.energy_evaluation != 'permutation':
            raise
        from DeterminantEnergy import check_determinant_size
        # Check before constructing a potentially large determinant space.
        check_determinant_size(len(system.S), system.Ne, system.Sm - 1, profile=getattr(system, 'determinant_size_profile', 'standard'))
        options = dict(compute_fci=False, energy_evaluation='determinant',
                       determinant_size_profile=getattr(system, 'determinant_size_profile', 'standard'))
        if hasattr(system, 'device'):
            options['device'] = system.device.platform
        warnings.warn(f'{exc} Retrying this geometry from its incoming orbitals '
                      'with determinant energy evaluation; convergence tolerances '
                      'are unchanged.', RuntimeWarning, stacklevel=2)
        replacement = type(system)(system.mol, **options)
        replacement.e_fci = system.e_fci
        energy, orbitals, spin = replacement.augmentedLagrangian(
            C.copy(), Cs.copy(), verbose=verbose)
        print('Energy evaluation switched to determinant for this and subsequent points.')
        return energy, orbitals, spin, replacement


def save_results(results, output_dir):
    """Persist only converged SCGVB points; incomplete curves end at failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'diatomic_results.json').write_text(
        json.dumps(results, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots()
    for method, marker in [('determinant', 'o'), ('permutation', 's')]:
        points = results['methods'].get(method, {}).get('points', [])
        if points:
            ax.plot([p['distance_angstrom'] for p in points],
                    [p['energy_hartree'] for p in points], marker=marker,
                    linestyle='-' if method == 'determinant' else '--', label=method,
                    markerfacecolor='none' if method == 'permutation' else None,
                    markersize=7 if method == 'permutation' else 5)
    fci = results['full_ci']
    if fci:
        ax.plot([p['distance_angstrom'] for p in fci],
                [p['energy_hartree'] for p in fci], ':x', label='Full CI')
    ax.set(xlabel='Internuclear distance / Angstrom', ylabel='Total energy / Eh')
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'plot_energy.pdf')
    fig.savefig(output_dir / 'plot_energy.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    for method, style, marker in [('determinant', '-', 'o'), ('permutation', '--', 's')]:
        points = results['methods'].get(method, {}).get('points', [])
        if not points:
            continue
        coefficients = np.asarray([p['Cs'] for p in points])
        for k in range(coefficients.shape[1]):
            ax.plot([p['distance_angstrom'] for p in points], coefficients[:, k] ** 2,
                    color=f'C{k % 10}', linestyle=style, marker=marker,
                    markerfacecolor='none' if method == 'permutation' else None,
                    markersize=7 if method == 'permutation' else 5,
                    label=f'{method}: |Cs[{k}]|²')
    ax.set(xlabel='Internuclear distance / Angstrom', ylabel='Squared spin coefficient')
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'plot_Cs.pdf')
    fig.savefig(output_dir / 'plot_Cs.png', dpi=160)
    plt.close(fig)


def run_scan(factory, methods, distances, output_dir, molecule_factory=generateMolecule):
    """Independent continuations with identical initial guesses and no switching."""
    rng = np.random.RandomState(40)
    results = {'methods': {m: {'points': [], 'failure': None} for m in methods},
               'full_ci': [], 'energy_convention': 'absolute total energy in hartree',
               'Cs_convention': 'raw normalized coefficients; plots show their squares',
               'dissociation': {'distance_angstrom': 10., 'methods': {}}}
    previous = {}
    try:
        # Keep the dissociation reference and historical random-draw order.
        for index, method in enumerate(methods):
            system = factory(molecule_factory(10.), energy_evaluation=method,
                             compute_fci=index == 0)
            if index == 0:
                initial_C = rng.rand(len(system.S), system.Ne)
                initial_Cs = np.ones(system.fs) / np.sqrt(system.fs)
                results['dissociation']['full_ci'] = float(system.e_fci)
            try:
                energy, _, _ = system.augmentedLagrangian(
                    initial_C.copy(), initial_Cs.copy(), verbose=verbose)
                results['dissociation']['methods'][method] = {'energy_hartree': float(energy)}
            except RuntimeError as exc:
                results['dissociation']['methods'][method] = {'failure': str(exc)}
                print(f'{method} dissociation reference failed: {exc}', flush=True)

        for R in distances:
            print(f'Internuclear distance: {R}', flush=True)
            mol = molecule_factory(R)
            # Full CI is computed once per geometry, including after SCGVB stops.
            reference_system = factory(mol, energy_evaluation=methods[0], compute_fci=True)
            results['full_ci'].append({'distance_angstrom': float(R),
                                       'energy_hartree': float(reference_system.e_fci)})
            if not previous and all(not data['points'] and data['failure'] is None
                                    for data in results['methods'].values()):
                initial_C = rng.rand(len(reference_system.S), reference_system.Ne)
                initial_Cs = np.ones(reference_system.fs) / np.sqrt(reference_system.fs)
            for index, method in enumerate(methods):
                data = results['methods'][method]
                if data['failure'] is not None:
                    continue
                system = reference_system if index == 0 else factory(
                    mol, energy_evaluation=method, compute_fci=False)
                C, Cs = previous.get(method, (initial_C, initial_Cs))
                try:
                    energy, C, Cs = system.augmentedLagrangian(C.copy(), Cs.copy(), verbose=verbose)
                except RuntimeError as exc:
                    data['failure'] = {'distance_angstrom': float(R), 'message': str(exc)}
                    print(f'{method} stopped at {R:g} Angstrom; retaining converged points: {exc}',
                          flush=True)
                    continue
                previous[method] = C.copy(), Cs.copy()
                data['points'].append({'distance_angstrom': float(R),
                                       'energy_hartree': float(energy),
                                       'C': C.tolist(), 'Cs': Cs.tolist()})
                print(f'{method}: {energy:.12f} Eh', flush=True)
            save_results(results, output_dir)
    finally:
        # Preserve completed points even if a later calculation is interrupted.
        save_results(results, output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description='Independent diatomic SCGVB / Full CI comparison')
    parser.add_argument('--backend', choices=('numpy', 'jax'), default='numpy')
    parser.add_argument('--energy-evaluation', choices=('both', 'permutation', 'determinant'),
                        default='both', help='Evaluators to compare (default: both; no fallback)')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=None,
                        help='JAX execution device; omitted selects the JAX default')
    parser.add_argument('--no-evaluator-fallback', action='store_true',
                        help='Compatibility option: the comparison never switches evaluators')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    if args.backend == 'jax':
        from NonorthgonalEnergyJAX import SCWF_Energy
        factory = partial(SCWF_Energy, device=args.device)
    else:
        if args.device is not None:
            parser.error('--device requires --backend jax')
        from NonorthgonalEnergy import SCWF_Energy
        factory = SCWF_Energy
    methods = ('permutation', 'determinant') if args.energy_evaluation == 'both' else (args.energy_evaluation,)
    print('Backend:', args.backend, '| Independent evaluators:', ', '.join(methods))
    # Edit the molecule above and the distance range here.
    run_scan(factory, methods, [0.1 * i for i in range(5, 25)], args.output_dir)


if __name__ == "__main__":
    main()
