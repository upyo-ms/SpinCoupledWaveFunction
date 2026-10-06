"""Opt-in, lossless snapshots for diagnosing SCGVB optimization failures."""
import json
from pathlib import Path
import tempfile

import numpy as np


class OptimizationDiagnostics:
    def __init__(self, directory, system):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        self.solve = 0
        self.event = 0
        space = system.active_space
        self.basis = np.eye(len(system.S)) if space is None else space.basis.copy()
        self.inactive = np.zeros((system.mol.nao_nr(), 0)) if space is None else space.inactive.copy()
        self.ecore = system.energy_nuc + system.inactive_energy
        self.macro = 0
        from Spin_Function.PrimitiveSpinFunc import generate_branch_diagrams
        np.savez_compressed(self.path/'reference.npz', ao_overlap=system.mol.intor('int1e_ovlp'),
                            spin_branch_diagrams=generate_branch_diagrams(
                                (system.Ne + system.mol.spin)//2, (system.Ne - system.mol.spin)//2))
        (self.path/'metadata.json').write_text(json.dumps({
            'schema_version': 1, 'molecule': system.mol.dumps(), 'active_electrons': system.Ne,
            'spin': system.mol.spin, 'ao_labels': system.mol.ao_labels(),
            'energy_evaluation': system.energy_evaluation,
            'backend': 'jax' if hasattr(system, 'device') else 'numpy',
            'device': str(getattr(system, 'device', 'cpu')),
            'note': 'Diagnostic snapshots, not restart JSON. h1/eri use the optimization basis; eri has chemist ordering. Total energy = numerator/norm + ecore.'
        }, indent=2)+'\n')

    def start(self, system):
        self.solve += 1
        np.savez_compressed(self.path/f'solve-{self.solve:04d}.npz',
                            S=system.S, h1=system.Hcore, eri=system.JK,
                            basis=self.basis, inactive=self.inactive, ecore=self.ecore,
                            macro_iteration=self.macro)

    def snapshot(self, system, stage, x=None, **details):
        self.event += 1
        record = dict(event=self.event, solve=self.solve, macro_iteration=self.macro,
                      stage=stage, **details)
        if x is None and stage == 'orbital_update':
            name = f'event-{self.event:05d}-{stage}.npz'
            np.savez_compressed(self.path/name, basis=self.basis, inactive=self.inactive)
            record['file'] = name
        if x is not None:
            # Preserve raw parameters as well as normalized spin coefficients.
            arrays = {'x': np.asarray(x).copy()}
            try:
                C, Cs = system.split_x(x)
                gram = C.T @ system.S @ C
                arrays.update(C=C, Cs=Cs, ao_C=self.basis @ C, orbital_overlap=gram)
                scale = np.sqrt(np.diag(gram))
                eig = np.linalg.eigvalsh(gram / np.outer(scale, scale))
                arrays['unit_orbital_overlap_eigenvalues'] = eig
                e, n = system.calcEnergy(C, Cs)
                record.update(numerator=float(e), norm=float(n),
                              energy=float(e/n + self.ecore),
                              unit_orbital_norm=float(system.unit_orbital_wavefunction_norm(x)))
                gradient = system.rayleigh_gradient(x)
                arrays['physical_gradient'] = gradient
                record['physical_gradient_max_abs'] = float(np.max(np.abs(gradient)))
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                record['evaluation_error'] = str(exc)
            name = f'event-{self.event:05d}-{stage}.npz'
            np.savez_compressed(self.path/name, **arrays)
            record['file'] = name
        # Nonfinite diagnostic numbers are represented by strings in valid JSON.
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = str(value)
        with (self.path/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
