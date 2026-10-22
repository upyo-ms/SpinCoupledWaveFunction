"""Strict TOML input mapped to the existing calculation CLI (no code execution)."""
import math
from pathlib import Path

# section: key -> (CLI option, expected type). MO indices remain one-based.
SECTIONS = {
    'molecule': {k: (k, t) for k, t in dict(atom=str, xyz=str, basis=str,
                  charge=int, multiplicity=int, unit=str).items()},
    'calculation': {k: (k, t) for k, t in dict(backend=str, device=str,
                  energy_evaluation=str, determinant_size_profile=str, initial_guess=str, initial_json=str, seed=int, initial_perturbation=float,
                  initial_spin_guess=str, initial_spin_overlap_cutoff=float, allow_geometry_change=bool).items()},
    'active_space': dict(mode=('active_space', str), **{k: (k, t) for k, t in dict(
                  pi_electrons=int, pi_normal_mode=str, frozen_core=int,
                  inactive_mos=list, active_mos=list, optimize_core=bool,
                  orbitals=list, ao_electrons=int, ao_threshold=float).items()}),
    'optimizer': dict(method=('optimization_method', str), **{k: (k, t) for k, t in dict(orbital_update=str, spin_solver=str, epsilon=float, max_outer_iterations=int,
                  max_inner_iterations=int, core_max_iterations=int,
                  core_gradient_tolerance=float, second_order=bool, second_order_backend=str,
                  second_order_fallback=bool, core_response=bool, max_permutation_pairs=int,
                  spin_overlap_cutoff=float, orbital_micro_iterations=int, joint_correction=bool,
                  joint_correction_start=int, orbital_overlap_cutoff=float).items()}),
    'output': dict(path=('output', str), verbose=('verbose', bool)),
    'geometry': {k: ('geometry_' + k, t) for k, t in dict(
                  method=str, coordinates=str, bonds=list, symmetry_operations=list,
                  freeze=list, set=list, constraint_tolerance=float,
                  restart_json=str, max_seconds=float, max_iterations=int, max_evaluations=int,
                  difference_step=float, gradient_tolerance=float, energy_tolerance=float,
                  step_tolerance=float, trust_radius=float, max_radius=float, final_radius=float).items()},
}


def input_arguments(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8')
    data = tomllib.loads(text)
    values = {}
    for section, entries in data.items():
        if section not in SECTIONS:
            raise ValueError(f'Unknown input section [{section}]')
        if not isinstance(entries, dict):
            raise ValueError(f'{section} must be a TOML table')
        for key, value in entries.items():
            if key not in SECTIONS[section]:
                raise ValueError(f'Unknown input key {section}.{key}')
            option, kind = SECTIONS[section][key]
            valid = type(value) is kind
            if section == 'output' and key == 'verbose':
                valid = type(value) is bool or (type(value) is int and 0 <= value <= 2)
            if kind is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            if kind is list:
                valid = type(value) is list and len(value) > 0 and all(
                    (type(i) is str and bool(i.strip())) if key in ('orbitals', 'bonds', 'symmetry_operations', 'freeze', 'set')
                    else (type(i) is int and i >= 1) for i in value)
            if not valid or (kind is str and not value.strip()):
                raise ValueError(f'Invalid value for {section}.{key} (expected {kind.__name__})')
            values[option] = value
    if ('second_order' in values and 'core_response' in values
            and values['second_order'] != values['core_response']):
        raise ValueError('optimizer.second_order and its legacy alias core_response disagree')
    if ('atom' in values) == ('xyz' in values):
        raise ValueError('Input [molecule] must specify exactly one of atom or xyz')
    if 'xyz' in values:
        values['xyz'] = str((path.parent / values['xyz']).resolve())
    if 'initial_json' in values:
        values['initial_json'] = str((path.parent / values['initial_json']).resolve())
    if 'geometry_restart_json' in values:
        values['geometry_restart_json'] = str((path.parent / values['geometry_restart_json']).resolve())
    values['output'] = str((path.parent / values['output']).resolve()) if 'output' in values else str(path.with_suffix('.json'))
    args = []
    for option, value in values.items():
        flag = '--' + option.replace('_', '-')
        if type(value) is bool:
            args.append(flag if value else '--no-' + option.replace('_', '-'))
        elif isinstance(value, list):
            if option in ('geometry_symmetry_operations', 'geometry_freeze', 'geometry_set'):
                args.extend(flag + '=' + str(i) for i in value)
            else:
                args.extend([flag, *(str(i) for i in value)])
        else:
            # The equals form safely preserves multiline geometries and values starting with '-'.
            args.append(flag + '=' + str(value))
    return args, {'path': str(path), 'text': text}
