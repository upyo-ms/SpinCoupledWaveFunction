"""Single-geometry nonorthogonal SCWF calculation, with optional HF initial orbitals."""
import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
from pyscf import gto
from scgvb.spin.dimensions import spin_function_dim
from scgvb.io.geometry_input import read_xyz, read_geometry


def build_molecule(atom, basis='sto-3g', charge=0, multiplicity=1, unit='Angstrom'):
    if multiplicity < 1:
        raise ValueError('Multiplicity must be at least 1')
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=multiplicity - 1,
                unit=unit, verbose=0)
    if mol.nelectron < 2:
        raise ValueError('The current SCWF energy implementation requires at least two electrons')
    if (mol.nelectron + mol.spin) // 2 > mol.nao_nr():
        raise ValueError('The basis has too few orbitals for this electron number and spin')
    return mol


def calculation_size(mol, active_space=None):
    ne = mol.nelectron if active_space is None else active_space.electrons
    n = mol.nao_nr() if active_space is None else active_space.basis.shape[1]
    fs = spin_function_dim(ne, mol.spin + 1, 0)
    pairs = math.factorial(ne) ** 2
    return {'electrons': mol.nelectron, 'atomic_orbitals': mol.nao_nr(),
            'active_electrons': ne, 'active_orbital_dimension': n,
            'determinants': math.comb(n, (ne + mol.spin)//2) * math.comb(n, (ne-mol.spin)//2),
            'spin_functions': fs, 'permutation_pairs': pairs,
            'dense_spin_matrix_bytes': pairs * fs * fs * 8}


def calculate(atom, basis='sto-3g', charge=0, multiplicity=1, unit='Angstrom',
              backend='numpy', device=None, seed=40, epsilon=1e-8,
              max_outer_iterations=50, max_inner_iterations=1000,
              max_permutation_pairs=100000, verbose=False, energy_evaluation='permutation',
              initial_guess='random', initial_perturbation=0.02,
              frozen_core=0, inactive_mos=None, active_mos=None, active_space='full', pi_electrons=None, pi_normal_mode='global',
              optimize_core=False, core_max_iterations=50, core_gradient_tolerance=1e-6,
              orbitals=None, ao_electrons=None, ao_threshold=.2, initial_json=None, diagnostics_dir=None,
              optimization_method='simultaneous', spin_overlap_cutoff=1e-10,
              orbital_micro_iterations=0, orbital_update='bfgs', spin_solver='canonical',
              orbital_overlap_cutoff=1e-10, joint_correction=False, joint_correction_start=1,
              initial_spin_guess='uniform', initial_spin_overlap_cutoff=1e-10,
              determinant_size_profile='standard', allow_geometry_change=True, core_response=None,
              second_order=None, second_order_fallback=False):
    """Return JSON-compatible results. HF runs for HF guesses or active-space selection; no FCI optimization.

    C columns correspond to the nonorthogonal orbitals in PySCF's AO order;
    Cs uses the Spin_Function branching-diagram basis, whose diagrams are saved.
    """
    from scgvb.energy.determinants import determinant_size_profile as validate_size_profile
    validate_size_profile(determinant_size_profile)
    from scgvb.optimization.second_order import resolve_option
    second_order = resolve_option(second_order, core_response)
    second_order_requested = second_order
    if type(second_order_fallback) is not bool:
        raise ValueError('second_order_fallback must be a boolean')
    core_response = bool(second_order and optimize_core)
    if verbose is None:
        verbose = 0
    from scgvb.optimization.alternating import validate_options
    validate_options(optimization_method, spin_overlap_cutoff, orbital_micro_iterations,
                     orbital_update, spin_solver, orbital_overlap_cutoff, joint_correction, joint_correction_start)
    from scgvb.optimization.initial_spin import validate_initial_spin, initial_spin_coefficients
    validate_initial_spin(initial_spin_guess, initial_spin_overlap_cutoff)
    if (orbital_update != 'bfgs' or spin_solver != 'canonical' or joint_correction) and energy_evaluation != 'determinant':
        raise ValueError('Wavefunction geometry options require determinant energy evaluation')
    if type(verbose) not in (bool, int) or not 0 <= verbose <= 2:
        raise ValueError('verbose must be false/true or an integer from 0 to 2')
    if backend not in ('numpy', 'jax'):
        raise ValueError('backend must be numpy or jax')
    if device is not None and (backend != 'jax' or device not in ('cpu', 'gpu')):
        raise ValueError('device must be cpu/gpu and requires the JAX backend')
    if energy_evaluation not in ('permutation', 'determinant'):
        raise ValueError('energy_evaluation must be permutation or determinant')
    if max_permutation_pairs < 1:
        raise ValueError('max_permutation_pairs must be positive')
    if initial_guess not in ('random', 'core', 'hf'):
        raise ValueError('initial_guess must be random, core or hf')
    if not np.isfinite(initial_perturbation) or initial_perturbation < 0:
        raise ValueError('initial_perturbation must be finite and nonnegative')
    mol = build_molecule(atom, basis, charge, multiplicity, unit)
    from scgvb.orbitals.active_space import prepare_active_space
    space = prepare_active_space(mol, frozen_core, inactive_mos, active_mos, active_space, pi_electrons, pi_normal_mode, orbitals, ao_electrons, ao_threshold)
    restart = None
    if initial_json is not None:
        from scgvb.orbitals.restart import restart_orbitals
        space, restart_C, restart_Cs, restart_info = restart_orbitals(
            mol, space, initial_json, allow_geometry_change=allow_geometry_change)
        restart = (restart_C, restart_Cs, restart_info)
    if optimize_core and (space is None or space.inactive.shape[1] == 0):
        raise ValueError('--optimize-core requires a nonempty inactive/core space')
    size = calculation_size(mol, space)
    if second_order:
        from scgvb.optimization.core_response import check_response_size
        check_response_size(size['active_orbital_dimension'], size['active_electrons'], mol.spin)
    if energy_evaluation == 'permutation' and size['permutation_pairs'] > max_permutation_pairs:
        raise ValueError(f"{size['active_electrons']} active electrons require {size['permutation_pairs']} "
                         f"permutation pairs (limit {max_permutation_pairs}). "
                         'Use --energy-evaluation determinant for a bounded active space, '
                         'or --check before increasing --max-permutation-pairs.')
    if backend == 'jax':
        from scgvb.energy.jax_backend import SCWF_Energy
        system = SCWF_Energy(mol, device=device, compute_fci=False,
                             energy_evaluation=energy_evaluation, active_space=space, determinant_size_profile=determinant_size_profile)
    else:
        from scgvb.energy.numpy_backend import SCWF_Energy
        system = SCWF_Energy(mol, compute_fci=False, energy_evaluation=energy_evaluation, active_space=space, determinant_size_profile=determinant_size_profile)
    from scgvb.orbitals.initial import initial_orbitals
    if restart is not None:
        C, _, initialization = restart
    elif space is None:
        C, initialization = initial_orbitals(mol, initial_guess, seed, initial_perturbation)
    else:
        C, initialization = space.initial_orbitals(initial_guess, seed, initial_perturbation, system.Hcore)
    Cs = np.ones(system.fs) / np.sqrt(system.fs)
    if restart is not None:
        C, Cs, initialization = restart
        norm = system.calcEnergy(C, Cs)[1]
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Restart wavefunction has invalid norm after projection')
        Cs = Cs / np.sqrt(norm)
    if verbose == 2:
        from scgvb.optimization.diagnostics import OptimizationDiagnostics
        system.diagnostics = OptimizationDiagnostics(diagnostics_dir or 'scwf_diagnostics', system)
        print('Optimization diagnostics:', system.diagnostics.path)
    Cs, spin_initialization = initial_spin_coefficients(
        system, C, Cs, method=initial_spin_guess, overlap_cutoff=initial_spin_overlap_cutoff,
        source='restart' if restart is not None else 'uniform')
    system.spin_initialization_info = spin_initialization
    if verbose == 2 and initial_spin_guess == 'eigen':
        system.diagnostics.snapshot(system, 'initial_spin', np.r_[C.ravel(), Cs],
                                    spin_initialization=spin_initialization)
    if verbose:
        print('Energy evaluation:', getattr(system, 'energy_evaluation', 'permutation'))
        print('Initial orbitals:', initialization)
        print('Initial spin coefficients:', spin_initialization)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always', RuntimeWarning)
        optimizer_options = dict(verbose=verbose, epsilon=epsilon,
                                 max_outer_iterations=max_outer_iterations,
                                 max_inner_iterations=max_inner_iterations,
                                 optimization_method=optimization_method,
                                 spin_overlap_cutoff=spin_overlap_cutoff,
                                 orbital_micro_iterations=orbital_micro_iterations,
                                 orbital_update=orbital_update, spin_solver=spin_solver,
                                 orbital_overlap_cutoff=orbital_overlap_cutoff, joint_correction=joint_correction,
                                 joint_correction_start=joint_correction_start)
        optimizer_options['second_order'] = second_order
        from scgvb.optimization.convergence import optimize_electronic
        electronic_result, second_order, fallback_info = optimize_electronic(
            system, C, Cs, optimize_core=optimize_core, second_order=second_order,
            second_order_fallback=second_order_fallback,
            core_max_iterations=core_max_iterations, core_gradient_tolerance=core_gradient_tolerance,
            optimizer_options=optimizer_options, verbose=verbose)
        total, C, Cs, system, space, core_info = electronic_result
        core_response = bool(second_order and optimize_core)
    electronic, norm = system.calcEnergy(C, Cs)
    gradient = system.rayleigh_gradient(np.r_[C.ravel(), Cs])
    from scgvb.spin.primitive import generate_branch_diagrams
    diagrams = generate_branch_diagrams((system.Ne + mol.spin) // 2,
                                        (system.Ne - mol.spin) // 2)
    ao_C = C if space is None else np.einsum('ij,ja->ia', space.basis, C)
    ao_S = mol.intor('int1e_ovlp')
    result = {
        'schema_version': 1,
        'molecule': {'atoms': [[mol.atom_symbol(i), mol.atom_coord(i, unit='Angstrom').tolist()]
                                for i in range(mol.natm)],
                     'coordinate_unit': 'Angstrom', 'charge': charge,
                     'multiplicity': multiplicity, 'basis': basis},
        'calculation': dict(size, backend=backend, determinant_size_profile=determinant_size_profile,
                            initialization=initialization,
                            initial_spin_guess=initial_spin_guess,
                            initial_spin_overlap_cutoff=initial_spin_overlap_cutoff,
                            spin_initialization=spin_initialization,
                            energy_evaluation=getattr(system, 'energy_evaluation', 'permutation'),
                            device=str(system.device) if backend == 'jax' else 'cpu',
                            optimizer=('two_step_scgvb_response' if core_response else 'two_step_scgvb') if optimize_core else (
                                'alternating_scgvb' if optimization_method == 'alternating'
                                else 'augmented_lagrangian_with_rayleigh_refinement'),
                            optimization_method=optimization_method,
                            core_response=core_response, second_order=second_order,
                            second_order_requested=second_order_requested,
                            second_order_fallback=second_order_fallback,
                            spin_overlap_cutoff=spin_overlap_cutoff if optimization_method == 'alternating' else None,
                            orbital_micro_iterations=orbital_micro_iterations if optimization_method == 'alternating' else None,
                            joint_correction=joint_correction,
                            joint_correction_start=joint_correction_start,
                            orbital_update=orbital_update if optimization_method == 'alternating' else None,
                            orbital_overlap_cutoff=orbital_overlap_cutoff if orbital_update == 'single_orbital' else None,
                            spin_solver=spin_solver if optimization_method == 'alternating' else None,
                            seed=seed, normalization_tolerance=epsilon,
                            max_outer_iterations=max_outer_iterations,
                            max_inner_iterations=max_inner_iterations),
        'core_optimization': core_info,
        'alternating_optimization': getattr(system, 'optimization_info', None),
        'second_order_optimization': getattr(system, 'second_order_info', None),
        'second_order_fallback': fallback_info,
        'total_energy_hartree': float(total),
        'electronic_numerator_hartree': float(electronic),
        'nuclear_energy_hartree': float(system.energy_nuc),
        'wavefunction_norm': float(norm),
        'normalization_error': float(norm - 1.),
        'physical_gradient_max_abs': float(np.max(np.abs(gradient))),
        'orbital_coefficients': ao_C.tolist(), 'spin_coefficients': Cs.tolist(),
        'ao_labels': mol.ao_labels(), 'spin_branch_diagrams': diagrams.tolist(),
        'ao_overlap': ao_S.tolist(),
        'orbital_overlap': np.einsum('ia,ij,jb->ab', ao_C, ao_S, ao_C).tolist(),
        'inactive_energy_hartree': float(system.inactive_energy),
        'active_space': {'selection': 'full'} if space is None else space.metadata,
        'inactive_orbital_coefficients': np.zeros((mol.nao_nr(), 0)).tolist() if space is None else space.inactive.tolist(),
        'optimization_basis_coefficients': None if space is None else space.basis.tolist(),
        'optimization_orbital_coefficients': C.tolist(),
        'warnings': [str(message.message) for message in messages],
    }
    if hasattr(system, 'diagnostics'):
        result['diagnostics_directory'] = str(system.diagnostics.path)
    from scgvb.analysis.population import population_analysis
    result['population_analysis'] = population_analysis(
        mol, ao_C, Cs, None if space is None else space.inactive, determinant_size_profile=determinant_size_profile)
    # Never serialize NaN as an apparently valid calculation result.
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', nargs='?', type=Path, help='TOML calculation input')
    parser.add_argument('--input', type=Path, help='TOML calculation input (same as positional file)')
    parser.add_argument('--determinant-size-profile', choices=('standard', 'large14'), default='standard',
                        help='Bounded determinant allocation profile; large14 enables up to 14e/14o')
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument('--atom', help='e.g. "H 0 0 0; H 0 0 1; H 0.8 0 0.5"')
    geometry.add_argument('--xyz', type=Path, help='Single-frame XYZ or SCGVB result JSON (top-level molecule.atoms)')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--multiplicity', type=int, default=1, help='2S+1 (singlet=1, doublet=2)')
    parser.add_argument('--unit', choices=('Angstrom', 'Bohr'), default='Angstrom')
    parser.add_argument('--backend', choices=('numpy', 'jax'), default='numpy')
    parser.add_argument('--device', choices=('cpu', 'gpu'))
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--initial-guess', choices=('random', 'core', 'hf'), default='random')
    parser.add_argument('--initial-perturbation', type=float, default=0.02,
                        help='S-metric perturbation of core/HF orbitals; 0 disables it')
    parser.add_argument('--initial-spin-guess', choices=('uniform', 'eigen'), default='uniform',
                        help='uniform: existing seed/restart; eigen: fixed-orbital spin solve before any optimizer')
    parser.add_argument('--initial-spin-overlap-cutoff', type=float, default=1e-10,
                        help='Relative spin overlap cutoff for initialization, independent of optimizer cutoff')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--max-outer-iterations', type=int, default=50)
    parser.add_argument('--max-inner-iterations', type=int, default=1000)
    parser.add_argument('--optimization-method', choices=('simultaneous', 'alternating'), default='simultaneous',
                        help='simultaneous: existing solver; alternating: fixed-orbital spin solves and orbital updates')
    parser.add_argument('--spin-overlap-cutoff', type=float, default=1e-10,
                        help='Relative spin overlap eigenvalue cutoff for alternating optimization')
    parser.add_argument('--orbital-micro-iterations', type=int, default=0,
                        help='Solve spin coefficients every N orbital steps within an alternating cycle; 0 keeps the full orbital block')
    parser.add_argument('--orbital-update', choices=('bfgs', 'wavefunction', 'single_orbital'), default='bfgs',
                        help='BFGS, wavefunction-preconditioned BFGS, or sequential one-orbital Rayleigh solves')
    parser.add_argument('--orbital-overlap-cutoff', type=float, default=1e-10,
                        help='Relative multi-electron overlap cutoff for one-orbital solves')
    parser.add_argument('--joint-correction', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional six-direction joint orbital/spin correction after each alternating cycle')
    parser.add_argument('--joint-correction-start', type=int, default=1,
                        help='First alternating sweep ending with a joint correction (1-based within each solve)')
    parser.add_argument('--spin-solver', choices=('canonical', 'projected'), default='canonical',
                        help='Optional centered Hamiltonian solve in the physical retained spin subspace')
    parser.add_argument('--max-permutation-pairs', type=int, default=100000)
    parser.add_argument('--active-space', choices=('full', 'benzene-pi', 'pi', 'ao'), default='full')
    parser.add_argument('--initial-json', type=Path, help='Saved SCGVB JSON; projects orbitals between bases and restores spin coefficients')
    parser.add_argument('--allow-geometry-change', action=argparse.BooleanOptionalAction, default=True,
                        help='Project initial-json onto the input geometry (default); --no-allow-geometry-change requires matching geometry')
    parser.add_argument('--geometry-method', choices=('none', 'bobyqa', 'quasi-newton'), default='none')
    parser.add_argument('--geometry-coordinates', choices=('auto', 'dlc', 'tric', 'cartesian', 'symmetry'), default='auto')
    parser.add_argument('--geometry-symmetry-operations', action='append',
                        help='Repeat generators, e.g. --geometry-symmetry-operations=-x,y,z')
    parser.add_argument('--geometry-freeze', action='append',
                        help='Repeat constraints fixed at their input values, e.g. "xyz 1" or "distance 2 3"')
    parser.add_argument('--geometry-set', action='append',
                        help='Repeat fixed targets, e.g. "distance 1 2 1.5" (Angstrom) or "angle 1 2 3 110" (degrees)')
    parser.add_argument('--geometry-constraint-tolerance', type=float, default=1e-7,
                        help='Maximum constraint error in Angstrom (positions/distances) or degrees (angles)')
    parser.add_argument('--geometry-restart-json', type=Path,
                        help='Resume geometry and electronic state from a geometry checkpoint')
    parser.add_argument('--geometry-max-seconds', type=float,
                        help='Wall-time budget per invocation; checked before each new electronic solve')
    parser.add_argument('--geometry-bonds', nargs='+', help='Override automatic bonds, e.g. 1-2 2-3 (one-based)')
    parser.add_argument('--geometry-max-iterations', type=int, default=100)
    parser.add_argument('--geometry-max-evaluations', type=int, default=2000)
    parser.add_argument('--geometry-difference-step', type=float, default=1e-3, help='Central difference step in Bohr')
    parser.add_argument('--geometry-gradient-tolerance', type=float, default=3e-4, help='RMS nuclear gradient in Eh/Bohr')
    parser.add_argument('--geometry-energy-tolerance', type=float, default=1e-6, help='Quasi-Newton energy tolerance in Eh')
    parser.add_argument('--geometry-step-tolerance', type=float, default=1.2e-3, help='RMS displacement tolerance in Angstrom')
    parser.add_argument('--geometry-trust-radius', type=float, default=0.1, help='Initial step radius in Angstrom')
    parser.add_argument('--geometry-max-radius', type=float, default=0.3, help='Maximum step/chart radius in Angstrom')
    parser.add_argument('--geometry-final-radius', type=float, default=1e-4, help='BOBYQA final scaled coordinate radius in Angstrom')
    parser.add_argument('--orbitals', nargs='+', help='MINAO target rows: 1 2px 2py A (roles A/I/F)')
    parser.add_argument('--ao-electrons', type=int, help='Assert selected active electron count')
    parser.add_argument('--ao-threshold', type=float, default=.2)
    parser.add_argument('--pi-normal-mode', choices=('global', 'local'), default='global',
                        help='global: one molecular plane; local: individual sp2 carbon planes')
    parser.add_argument('--pi-electrons', type=int, help='Assert the number of electrons selected by --active-space pi')
    parser.add_argument('--frozen-core', type=int, default=0,
                        help='Lowest HF spatial MOs kept doubly occupied; fixed unless --optimize-core')
    parser.add_argument('--optimize-core', action=argparse.BooleanOptionalAction,
                        help='Relax inactive/core orbitals except AO F targets, keeping double occupation')
    parser.add_argument('--core-max-iterations', type=int, default=50)
    parser.add_argument('--core-gradient-tolerance', type=float, default=1e-6)
    parser.add_argument('--second-order', '--core-response', dest='second_order',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Analytic SCGVB C/Cs corrections, plus outer response when optimizing core; core-response is a legacy alias')
    parser.add_argument('--second-order-fallback', action=argparse.BooleanOptionalAction, default=False,
                        help='Retry electronic nonconvergence once with second-order corrections; default off')
    parser.add_argument('--inactive-mos', nargs='+', type=int, help='One-based doubly occupied HF MO indices')
    parser.add_argument('--active-mos', nargs='+', type=int, help='One-based HF MOs spanning the active optimization space, including virtuals')
    parser.add_argument('--check', action='store_true', help='Check size without optimization; active-space selection runs HF')
    parser.add_argument('--verbose', nargs='?', const=1, type=int, choices=(0, 1, 2), default=0,
                        help='0: quiet, 1: progress, 2: progress and diagnostic snapshots')
    parser.add_argument('--no-verbose', dest='verbose', action='store_const', const=0)
    parser.add_argument('--energy-evaluation', choices=('permutation', 'determinant'),
                        default='permutation', help='determinant: exterior-product evaluation without permutation tables')
    parser.add_argument('--output', type=Path, default=Path('scwf_result.json'))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        provenance = None
        if args.input is not None and args.input_file is not None:
            raise ValueError('Use either --input or a positional input file, not both')
        source = args.input if args.input is not None else args.input_file
        if source is not None:
            from scgvb.io.calculation_input import input_arguments
            defaults, provenance = input_arguments(source)
            # A CLI geometry replaces the input geometry; other explicit CLI
            # options follow file options and therefore take precedence.
            if args.atom is not None or args.xyz is not None:
                defaults = [token for token in defaults if not token.startswith(('--atom=', '--xyz='))]
            args = parser.parse_args(defaults + argv)
        if args.atom is None and args.xyz is None:
            raise ValueError('Specify an input file, --atom, or --xyz')
        if args.initial_json is not None and args.initial_json.resolve() == args.output.resolve():
            raise ValueError('Restart source and output must be different files')
        if (args.xyz is not None and args.xyz.resolve() == args.output.resolve()
                and not (args.geometry_method != 'none' and args.geometry_restart_json is not None
                         and args.geometry_restart_json.resolve() == args.output.resolve())):
            raise ValueError('Geometry source and output must be different files')
        if args.geometry_method != 'none':
            from scgvb.optimization.geometry import validate_geometry_options
            validate_geometry_options(
                method=args.geometry_method, coordinates=args.geometry_coordinates,
                max_iterations=args.geometry_max_iterations, max_evaluations=args.geometry_max_evaluations,
                difference_step=args.geometry_difference_step, gradient_tolerance=args.geometry_gradient_tolerance,
                energy_tolerance=args.geometry_energy_tolerance, step_tolerance=args.geometry_step_tolerance,
                trust_radius=args.geometry_trust_radius, max_radius=args.geometry_max_radius,
                final_radius=args.geometry_final_radius,
                symmetry_operations=args.geometry_symmetry_operations, max_seconds=args.geometry_max_seconds,
                freeze=args.geometry_freeze, set_values=args.geometry_set,
                constraint_tolerance=args.geometry_constraint_tolerance)
        elif (args.geometry_restart_json is not None or args.geometry_symmetry_operations
              or args.geometry_freeze or args.geometry_set):
            raise ValueError('Geometry restart, symmetry and constraint options require geometry optimization')
        from scgvb.optimization.alternating import validate_options
        validate_options(args.optimization_method, args.spin_overlap_cutoff, args.orbital_micro_iterations,
                         args.orbital_update, args.spin_solver, args.orbital_overlap_cutoff, args.joint_correction,
                         args.joint_correction_start)
        from scgvb.optimization.initial_spin import validate_initial_spin
        validate_initial_spin(args.initial_spin_guess, args.initial_spin_overlap_cutoff)
        if (args.orbital_update != 'bfgs' or args.spin_solver != 'canonical' or args.joint_correction) and args.energy_evaluation != 'determinant':
            raise ValueError('Wavefunction geometry options require determinant energy evaluation')
        geometry_input = None
        if args.xyz:
            atom, geometry_input = read_geometry(args.xyz, unit=args.unit)
        else:
            atom = args.atom
        check_restart = args.initial_json
        if args.geometry_restart_json is not None:
            saved = json.loads(args.geometry_restart_json.read_text())
            checkpoint = saved.get('geometry_optimization', {}).get('checkpoint', {})
            if checkpoint.get('version') != 1 or 'best_state' not in checkpoint:
                raise ValueError('Invalid geometry restart checkpoint')
            previous_constraints = saved['geometry_optimization']
            if (previous_constraints.get('freeze_requested', []) != list(args.geometry_freeze or [])
                    or previous_constraints.get('set_requested', []) != list(args.geometry_set or [])
                    or previous_constraints.get('constraint_tolerance', 1e-7) != args.geometry_constraint_tolerance):
                raise ValueError('Geometry restart has incompatible constraint settings')
            check_restart = checkpoint['best_state']
            requested = build_molecule(atom, args.basis, args.charge, args.multiplicity, args.unit)
            if ([requested.atom_symbol(i) for i in range(requested.natm)]
                    != [a[0] for a in check_restart['molecule']['atoms']]):
                raise ValueError('Geometry restart requires the same atoms and atom order')
            atom = check_restart['molecule']['atoms']
            args.unit = 'Angstrom'
            geometry_input = dict(source=str(args.geometry_restart_json.resolve()), format='scgvb_json',
                selection='geometry_optimization.checkpoint.best_state.molecule.atoms',
                source_coordinate_unit='Angstrom', input_coordinate_unit='Angstrom',
                requested_geometry_input=geometry_input)
        inactive = None if args.inactive_mos is None else [i-1 for i in args.inactive_mos]
        active = None if args.active_mos is None else [i-1 for i in args.active_mos]
        if args.check:
            mol = build_molecule(atom, args.basis, args.charge, args.multiplicity, args.unit)
            if args.geometry_method != 'none':
                from scgvb.optimization.constraints import prepare_constraints
                from scgvb.optimization.coordinates import GeometryCoordinates
                mol, limits = prepare_constraints(
                    mol, freeze=args.geometry_freeze, set_values=args.geometry_set,
                    tolerance=args.geometry_constraint_tolerance, coordinates=args.geometry_coordinates,
                    symmetry_operations=args.geometry_symmetry_operations,
                    state=None if args.geometry_restart_json is None else saved['geometry_optimization'].get('constraints', []))
                geometry_chart = GeometryCoordinates(
                    mol, args.geometry_coordinates, bonds=args.geometry_bonds,
                    symmetry_operations=args.geometry_symmetry_operations, constraints=limits)
            from scgvb.orbitals.active_space import prepare_active_space
            space = prepare_active_space(mol, args.frozen_core, inactive, active, args.active_space, args.pi_electrons, args.pi_normal_mode, args.orbitals, args.ao_electrons, args.ao_threshold)
            if check_restart is not None:
                from scgvb.orbitals.restart import restart_orbitals
                space, _, _, restart_info = restart_orbitals(
                    mol, space, check_restart,
                    allow_geometry_change=args.allow_geometry_change or args.geometry_method != 'none')
            size = calculation_size(mol, space)
            if geometry_input is not None:
                size['geometry_input'] = geometry_input
            if check_restart is not None:
                size['initialization'] = restart_info
            size['active_space'] = {'selection': 'full'} if space is None else space.metadata
            if args.optimize_core and (space is None or space.inactive.shape[1] == 0):
                raise ValueError('--optimize-core requires a nonempty inactive/core space')
            size['determinant_size_profile'] = args.determinant_size_profile
            size['optimize_core'] = args.optimize_core
            size['second_order'] = args.second_order
            size['second_order_fallback'] = args.second_order_fallback
            size['core_response'] = bool(args.second_order and args.optimize_core)
            if args.second_order:
                from scgvb.optimization.core_response import check_response_size
                size['second_order_size'] = check_response_size(
                    size['active_orbital_dimension'], size['active_electrons'], mol.spin)
                if args.optimize_core:
                    size['core_response_size'] = size['second_order_size']
            size['optimization_method'] = args.optimization_method
            size['spin_overlap_cutoff'] = args.spin_overlap_cutoff if args.optimization_method == 'alternating' else None
            size['orbital_micro_iterations'] = args.orbital_micro_iterations if args.optimization_method == 'alternating' else None
            size['joint_correction'] = args.joint_correction
            size['joint_correction_start'] = args.joint_correction_start
            size['initial_spin_guess'] = args.initial_spin_guess
            size['initial_spin_overlap_cutoff'] = args.initial_spin_overlap_cutoff
            size['orbital_update'] = args.orbital_update if args.optimization_method == 'alternating' else None
            size['orbital_overlap_cutoff'] = args.orbital_overlap_cutoff if args.orbital_update == 'single_orbital' else None
            size['spin_solver'] = args.spin_solver if args.optimization_method == 'alternating' else None
            if args.geometry_method != 'none':
                size['geometry_optimization'] = dict(method=args.geometry_method, **geometry_chart.metadata)
            if args.energy_evaluation == 'determinant' or args.optimize_core:
                from scgvb.energy.determinants import check_determinant_size
                check_determinant_size(size['active_orbital_dimension'], size['active_electrons'], mol.spin, profile=args.determinant_size_profile)
            print(json.dumps(size, indent=2))
            return 0
        calculation_options = dict(basis=args.basis, charge=args.charge,
                           multiplicity=args.multiplicity, unit=args.unit, backend=args.backend,
                           device=args.device, seed=args.seed, epsilon=args.epsilon,
                           max_outer_iterations=args.max_outer_iterations,
                           max_inner_iterations=args.max_inner_iterations,
                           max_permutation_pairs=args.max_permutation_pairs, verbose=args.verbose,
                           energy_evaluation=args.energy_evaluation,
                           determinant_size_profile=args.determinant_size_profile,
                           initial_guess=args.initial_guess,
                           initial_perturbation=args.initial_perturbation,
                           initial_spin_guess=args.initial_spin_guess,
                           initial_spin_overlap_cutoff=args.initial_spin_overlap_cutoff,
                           frozen_core=args.frozen_core, inactive_mos=inactive,
                           active_mos=active, active_space=args.active_space, pi_electrons=args.pi_electrons, pi_normal_mode=args.pi_normal_mode,
                           optimize_core=args.optimize_core, core_max_iterations=args.core_max_iterations,
                           core_gradient_tolerance=args.core_gradient_tolerance,
                           second_order=args.second_order, second_order_fallback=args.second_order_fallback,
                           orbitals=args.orbitals, ao_electrons=args.ao_electrons, ao_threshold=args.ao_threshold, initial_json=args.initial_json,
                           diagnostics_dir=args.output.with_name(args.output.stem + '_diagnostics'),
                           optimization_method=args.optimization_method,
                           spin_overlap_cutoff=args.spin_overlap_cutoff,
                           orbital_micro_iterations=args.orbital_micro_iterations,
                           orbital_update=args.orbital_update, spin_solver=args.spin_solver,
                           orbital_overlap_cutoff=args.orbital_overlap_cutoff, joint_correction=args.joint_correction,
                           joint_correction_start=args.joint_correction_start,
                           allow_geometry_change=args.allow_geometry_change)
        if args.geometry_method == 'none':
            result = calculate(atom, **calculation_options)
        else:
            from scgvb.optimization.geometry import optimize_geometry
            result = optimize_geometry(
                atom, calculation_options=calculation_options, method=args.geometry_method,
                coordinates=args.geometry_coordinates, bonds=args.geometry_bonds,
                max_iterations=args.geometry_max_iterations, max_evaluations=args.geometry_max_evaluations,
                difference_step=args.geometry_difference_step, gradient_tolerance=args.geometry_gradient_tolerance,
                energy_tolerance=args.geometry_energy_tolerance, step_tolerance=args.geometry_step_tolerance,
                trust_radius=args.geometry_trust_radius, max_radius=args.geometry_max_radius,
                final_radius=args.geometry_final_radius, output=args.output, provenance=provenance,
                symmetry_operations=args.geometry_symmetry_operations,
                restart_json=args.geometry_restart_json, max_seconds=args.geometry_max_seconds,
                freeze=args.geometry_freeze, set_values=args.geometry_set,
                constraint_tolerance=args.geometry_constraint_tolerance)
        if provenance is not None:
            result['input_file'] = provenance
        if geometry_input is not None:
            result['geometry_input'] = geometry_input
        if args.geometry_method == 'none':
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        elif geometry_input is not None:
            from scgvb.optimization.geometry_checkpoint import atomic_json
            atomic_json(args.output, result)
        print(f"Total energy: {result['total_energy_hartree']:.12f} Eh")
        print(f"Norm error: {result['normalization_error']:.3e}")
        from scgvb.analysis.population import print_populations
        print_populations(result['population_analysis'])
        for message in result['warnings']:
            print(f'Warning: {message}', file=sys.stderr)
        print(f'Result: {args.output.resolve()}')
        if args.geometry_method != 'none':
            info = result['geometry_optimization']
            print(f"Geometry: {info['status']} ({info['energy_evaluations']} energy evaluations)")
            if not info['converged']:
                print(f"Geometry optimization stopped: {info['message']}", file=sys.stderr)
                return 1
        return 0
    except (ValueError, RuntimeError, FloatingPointError, OSError, ImportError) as exc:
        print(f'Calculation failed: {exc}', file=sys.stderr)
        fallback = getattr(exc, 'second_order_fallback_info', None)
        if fallback is not None:
            print(f"Automatic second-order retry: {fallback['status']}. "
                  f"First attempt: {fallback['first_order_error']}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
