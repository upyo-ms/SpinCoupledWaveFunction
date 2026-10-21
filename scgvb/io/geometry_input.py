"""Single input geometry from standard XYZ or the top-level SCGVB JSON result."""
import json
from pathlib import Path

import numpy as np
from pyscf.lib import param

BOHR = param.BOHR


def read_xyz(path):
    """Read exactly one XYZ frame; trajectory selection is never implicit."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError('XYZ requires atom count, comment and atom records')
    count = int(lines[0].strip())
    records = lines[2:]
    while records and not records[-1].strip():
        records.pop()
    if count < 1 or len(records) != count:
        raise ValueError('XYZ atom count must match a single frame')
    atoms = []
    for line in records:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError('XYZ records must contain element x y z')
        position = [float(value) for value in fields[1:]]
        if not np.isfinite(position).all():
            raise ValueError('Coordinates must be finite')
        atoms.append((fields[0], position))
    return atoms


def _unit(unit):
    names = {'angstrom': ('Angstrom', 1.), 'ang': ('Angstrom', 1.), 'a': ('Angstrom', 1.),
             'bohr': ('Bohr', BOHR), 'au': ('Bohr', BOHR), 'a.u.': ('Bohr', BOHR)}
    if not isinstance(unit, str) or unit.lower() not in names:
        raise ValueError('Geometry coordinate unit must be Angstrom or Bohr')
    return names[unit.lower()]


def read_geometry(path, unit='Angstrom'):
    """Return (atoms, source info), converting JSON coordinates to input units.

    A JSON file supplies only molecule.atoms and coordinate_unit. Calculation
    settings, histories and electronic coefficients are not imported here.
    """
    path = Path(path).resolve()
    target_unit, target_scale = _unit(unit)
    if path.suffix.lower() != '.json':
        return read_xyz(path), dict(source=str(path), format='xyz', selection='single_frame',
                                    source_coordinate_unit=target_unit, input_coordinate_unit=target_unit)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('molecule'), dict):
        raise ValueError('Geometry JSON must contain top-level molecule.atoms')
    spec = data['molecule']
    records = spec.get('atoms')
    if not isinstance(records, list) or not records:
        raise ValueError('Geometry JSON molecule.atoms must be a nonempty atom list')
    source_unit, source_scale = _unit(spec.get('coordinate_unit', 'Angstrom'))
    atoms = []
    for record in records:
        if (not isinstance(record, (list, tuple)) or len(record) != 2
                or not isinstance(record[0], str) or not record[0].strip()
                or not isinstance(record[1], (list, tuple)) or len(record[1]) != 3
                or not all(type(value) in (int, float) for value in record[1])):
            raise ValueError('Geometry JSON atoms must contain [element, [x, y, z]] records')
        coords = np.asarray(record[1], dtype=float)*(source_scale/target_scale)
        if not np.isfinite(coords).all():
            raise ValueError('Geometry JSON coordinates must be finite')
        atoms.append((record[0], coords.tolist()))
    return atoms, dict(source=str(path), format='scgvb_json', selection='molecule.atoms',
                       source_coordinate_unit=source_unit, input_coordinate_unit=target_unit)
