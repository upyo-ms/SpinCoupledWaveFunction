"""Export nonorthogonal orbitals from calc_molecule JSON to Cube and an HTML viewer."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from pyscf import gto
from pyscf.tools import cubegen


def load_result(path):
    result = json.loads(Path(path).read_text())
    if result.get('schema_version') != 1:
        raise ValueError('Expected calc_molecule result schema_version 1')
    data = result['molecule']
    if data['coordinate_unit'] not in ('Angstrom', 'Bohr'):
        raise ValueError('Coordinate unit must be Angstrom or Bohr')
    mol = gto.M(atom=data['atoms'], basis=data['basis'], charge=data['charge'],
                spin=data['multiplicity'] - 1, unit=data['coordinate_unit'], verbose=0)
    C = np.asarray(result['orbital_coefficients'], dtype=float)
    ne = result.get('calculation', {}).get('active_electrons', mol.nelectron)
    if C.shape != (mol.nao_nr(), ne) or not np.isfinite(C).all():
        raise ValueError('Orbital coefficients must be finite and have shape (AO count, electron count)')
    inactive = np.asarray(result.get('inactive_orbital_coefficients',
                                     np.zeros((mol.nao_nr(), 0))), dtype=float)
    if (inactive.ndim != 2 or inactive.shape[0] != mol.nao_nr()
            or ne + 2 * inactive.shape[1] != mol.nelectron or not np.isfinite(inactive).all()):
        raise ValueError('Inactive orbital count/coefficients do not match electron count')
    if result['ao_labels'] != mol.ao_labels():
        raise ValueError('AO labels/order do not match the reconstructed molecule')
    S = mol.intor('int1e_ovlp')
    saved_S = np.asarray(result['ao_overlap'], dtype=float)
    if saved_S.shape != S.shape or not np.allclose(saved_S, S, rtol=1e-9, atol=1e-10):
        raise ValueError('AO overlap does not match: check geometry, basis and coordinate units')
    if (not np.allclose(np.einsum('ia,ij,jb->ab', inactive, S, inactive), np.eye(inactive.shape[1]), atol=1e-8)
            or not np.allclose(np.einsum('ia,ij,jb->ab', inactive, S, C), 0, atol=1e-8)):
        raise ValueError('Inactive orbitals must be orthonormal and orthogonal to active orbitals')
    # Active SCGVB orbitals first, then each doubly occupied inactive orbital once.
    C = np.column_stack((C, inactive))
    return mol, C, S


def export_orbitals(result_file, output_dir='orbitals', orbitals=None, grid=(64, 64, 64),
                    margin=3., normalize=False, isovalue=0.04, make_html=True,
                    viewer_js=None, max_grid_points=2000000):
    """Write phi_i(r) = sum_mu C[mu,i] chi_mu(r); never orthogonalize C.

    Public orbital numbers are one-based. Cube grid lengths/margin are in Bohr.
    """
    mol, C, S = load_result(result_file)
    source = json.loads(Path(result_file).read_text())
    active_count = source.get('calculation', {}).get('active_electrons', mol.nelectron)
    if len(grid) != 3 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in grid):
        raise ValueError('Grid requires three integer dimensions, each >= 2')
    if np.prod(grid, dtype=object) > max_grid_points:
        raise ValueError('Grid exceeds --max-grid-points')
    if not np.isfinite(margin) or margin <= 0 or not np.isfinite(isovalue) or isovalue <= 0:
        raise ValueError('Margin and isovalue must be finite and positive')
    selected = list(range(1, C.shape[1] + 1)) if orbitals is None else list(orbitals)
    if not selected or any(not isinstance(i, (int, np.integer)) or i < 1 or i > C.shape[1] for i in selected):
        raise ValueError(f'Orbital numbers must lie between 1 and {C.shape[1]}')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate orbital numbers')
    coeff = C[:, np.asarray(selected) - 1].copy()
    norms = np.einsum('ia,ij,ja->a', coeff, S, coeff)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('Selected orbitals must have finite positive norms')
    scales = 1 / np.sqrt(norms) if normalize else np.ones(len(selected))
    coeff *= scales
    if viewer_js is not None and not Path(viewer_js).is_file():
        raise ValueError('--viewer-js must point to a local 3Dmol JavaScript file')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cube = cubegen.Cube(mol, *grid, resolution=None, margin=margin)
    # Molecular Cube boxes are diagonal. Build their coordinates explicitly;
    # no large matrix product is needed for this Cartesian grid.
    fractional = np.stack(np.meshgrid(cube.xs, cube.ys, cube.zs, indexing='ij'), axis=-1)
    coords = fractional.reshape(-1, 3) * np.diag(cube.box) + cube.boxorig
    values = np.empty((len(coords), len(selected)))
    for start in range(0, len(coords), 8000):
        stop = min(start + 8000, len(coords))
        values[start:stop] = np.einsum('pi,ia->pa', mol.eval_gto('GTOval', coords[start:stop]), coeff)
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite orbital values on the grid')
    xyz = mol.tostring(format='xyz')
    (output / 'molecule.xyz').write_text(xyz + '\n')
    records, viewer_records = [], []
    for column, number in enumerate(selected):
        name = f'orbital_{number:03d}.cube'
        field = values[:, column].reshape(tuple(grid))
        cube.write(field, str(output / name), comment=f'Nonorthogonal orbital {number}; amplitude in Bohr^(-3/2)')
        records.append({'orbital': int(number), 'file': name,
                        'role': 'active' if number <= active_count else 'inactive_doubly_occupied',
                        'original_norm_squared': float(norms[column]),
                        'coefficient_scale': float(scales[column]),
                        'minimum': float(field.min()), 'maximum': float(field.max())})
        if make_html:
            viewer_records.append(dict(records[-1], cube=(output / name).read_text()))
    manifest = {'source_result': str(Path(result_file).resolve()),
                'normalized_individually': bool(normalize),
                'grid_shape': list(grid), 'margin_bohr': margin,
                'origin_bohr': cube.boxorig.tolist(),
                'grid_vectors_bohr': (cube.box / (np.asarray(grid) - 1)[:, None]).tolist(),
                'display_isovalue': isovalue,
                'displayed_orbital_overlap': np.einsum('ia,ij,jb->ab', coeff, S, coeff).tolist(),
                'orbitals': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    if make_html:
        script = 'https://3Dmol.org/build/3Dmol-min.js'
        if viewer_js is not None:
            destination = output / '3Dmol-min.js'
            if Path(viewer_js).resolve() != destination.resolve():
                shutil.copyfile(viewer_js, destination)
            script = '3Dmol-min.js'
        payload = json.dumps({'xyz': xyz, 'orbitals': viewer_records,
                              'isovalue': isovalue, 'normalized': bool(normalize)}, allow_nan=False)
        # Do not allow embedded text to terminate the JSON script element.
        payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        template = Path(__file__).with_name('orbital_viewer.html').read_text()
        page = template.replace('__SCWF_DATA__', payload).replace('__VIEWER_SCRIPT__', html.escape(script, quote=True))
        (output / 'index.html').write_text(page)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='JSON produced by calc_molecule.py')
    parser.add_argument('--output-dir', type=Path, default=Path('orbitals'))
    parser.add_argument('--orbitals', nargs='+', type=int, help='One-based orbital numbers; default all')
    parser.add_argument('--grid', nargs=3, type=int, default=(64, 64, 64), metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--margin', type=float, default=3., help='Box margin in Bohr')
    parser.add_argument('--isovalue', type=float, default=0.04, help='Initial HTML surface amplitude')
    parser.add_argument('--normalize', action='store_true', help='Normalize each orbital individually (never orthogonalize)')
    parser.add_argument('--no-html', action='store_true', help='Only write Cube, XYZ and manifest')
    parser.add_argument('--viewer-js', type=Path, help='Local 3Dmol-min.js for offline viewing')
    parser.add_argument('--max-grid-points', type=int, default=2000000)
    args = parser.parse_args(argv)
    try:
        manifest = export_orbitals(args.result, args.output_dir, args.orbitals, args.grid,
                                   args.margin, args.normalize, args.isovalue, not args.no_html,
                                   args.viewer_js, args.max_grid_points)
        print(f"Exported {len(manifest['orbitals'])} orbitals to {args.output_dir.resolve()}")
        if not args.no_html:
            print(f"Open in a browser: {(args.output_dir / 'index.html').resolve()}")
            if args.viewer_js is None:
                print('HTML viewer requires internet to load 3Dmol.js; Cube files work offline.')
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f'Orbital export failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
