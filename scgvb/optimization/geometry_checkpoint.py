"""Atomic geometry checkpoints and disk-backed electronic evaluation reuse."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def model_signature(mol, options):
    excluded = {'initial_json', 'allow_geometry_change', 'unit', 'verbose', 'diagnostics_dir'}
    settings = {k: str(v) if isinstance(v, Path) else v
                for k, v in options.items() if k not in excluded}
    settings.update(atom_symbols=[mol.atom_symbol(i) for i in range(mol.natm)],
                    charge=mol.charge, multiplicity=mol.spin + 1)
    # Reusing old energies after an implementation change would silently mix
    # energy surfaces. A frozen HPC source bundle keeps this stable on restart.
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        source.update(str(path.relative_to(root)).encode() + b'\0')
        source.update(path.read_bytes())
    settings['implementation_sha256'] = source.hexdigest()
    encoded = json.dumps(settings, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class ElectronicCache:
    def __init__(self, output, signature, previous=()):
        self.signature = signature
        self.directory = None
        if output is not None:
            root = Path(output).resolve().with_name(Path(output).stem + '_geometry_cache')
            root.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        self.directories = ([] if self.directory is None else [self.directory])
        self.directories.extend(Path(p) for p in previous if Path(p) not in self.directories)

    def name(self, coords):
        key = hashlib.sha256(np.asarray(coords, dtype=np.float64).tobytes()).hexdigest()
        return self.signature + '-' + key + '.json'

    def load(self, coords):
        for directory in self.directories:
            path = directory / self.name(coords)
            if path.is_file():
                data = json.loads(path.read_text())
                if (data['model_signature'] != self.signature
                        or not np.array_equal(data['coordinates_bohr'], coords)):
                    raise ValueError('Electronic geometry-cache entry is incompatible')
                return data['result']
        return None

    def store(self, coords, result):
        if self.directory is not None:
            atomic_json(self.directory / self.name(coords), dict(
                model_signature=self.signature, coordinates_bohr=np.asarray(coords).tolist(), result=result))

    def count(self):
        return sum(len(list(d.glob(self.signature + '-*.json'))) for d in self.directories if d.is_dir())
