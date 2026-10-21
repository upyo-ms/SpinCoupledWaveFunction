"""Hard geometry constraints, with persistent targets in Angstrom/degrees.

Input examples: freeze=['xyz 1', 'distance 2 3']; set_values=['angle 1 2 3 110'].
The Jacobian and solver residuals use Bohr/radians, never energy derivatives.
"""
import numpy as np
from pyscf.lib import param

BOHR = param.BOHR


def null_space(matrix):
    """Orthonormal null space with an absolute floor for roundoff-only rows."""
    matrix = np.asarray(matrix, dtype=float)
    _, s, right = np.linalg.svd(matrix, full_matrices=True)
    rank = np.count_nonzero(s > 1e-10 * max(1., s[0] if len(s) else 0.))
    return right[rank:].T


class GeometryConstraints:
    def __init__(self, coords, *, freeze=None, set_values=None, tolerance=1e-7, state=None):
        from geometric.internal import Distance, Angle, Dihedral, CartesianX, CartesianY, CartesianZ
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('constraint_tolerance must be finite and positive')
        self.tolerance = float(tolerance)
        self.reference = np.asarray(coords, dtype=float).ravel().copy()
        self.natoms = len(self.reference) // 3
        self.primitives, self.targets, self.factors, self.entries = [], [], [], []
        factories = dict(distance=Distance, angle=Angle, dihedral=Dihedral,
                         x=CartesianX, y=CartesianY, z=CartesianZ)
        seen = {}

        def add(kind, atoms, target=None):
            count = dict(distance=2, angle=3, dihedral=4, x=1, y=1, z=1)[kind]
            if (len(atoms) != count or len(set(atoms)) != count
                    or any(type(i) is not int or i < 1 or i > self.natoms for i in atoms)):
                raise ValueError(f'Invalid atom indices for {kind} constraint: {atoms}')
            if kind in ('distance', 'angle', 'dihedral'):
                atoms = list(min(tuple(atoms), tuple(reversed(atoms))))
            primitive = factories[kind](*(i - 1 for i in atoms))
            factor = 180. / np.pi if kind in ('angle', 'dihedral') else BOHR
            if target is None:
                target = float(primitive.value(self.reference)) * factor
            target = float(target)
            if (not np.isfinite(target) or (kind == 'distance' and target <= 0)
                    or (kind == 'angle' and not 0 < target < 180)):
                raise ValueError(f'Invalid {kind} constraint target: {target}')
            if kind == 'dihedral':
                target = (target + 180.) % 360. - 180.
            key = kind, tuple(atoms)
            if key in seen:
                if abs(seen[key] - target) > 1e-10:
                    raise ValueError(f'Conflicting targets for {kind} {atoms}')
                return
            seen[key] = target
            self.primitives.append(primitive)
            self.targets.append(target / factor)
            self.factors.append(factor)
            self.entries.append(dict(type=kind, atoms=list(atoms), value=target))

        if state is not None:
            for entry in state:
                if entry['type'] not in factories:
                    raise ValueError('Invalid constraint type in geometry checkpoint')
                add(entry['type'], entry['atoms'], entry['value'])
        else:
            for specifications, explicit in ((freeze, False), (set_values, True)):
                if specifications is None:
                    continue
                if not isinstance(specifications, (list, tuple)) or not all(isinstance(v, str) for v in specifications):
                    raise ValueError('Geometry freeze/set must be lists of constraint strings')
                for specification in specifications:
                    fields = specification.split()
                    if not fields:
                        raise ValueError('Empty geometry constraint')
                    kind = fields[0].lower()
                    try:
                        if kind in ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'):
                            if len(fields) != (2 + len(kind) if explicit else 2):
                                raise ValueError('Wrong number of Cartesian constraint fields')
                            for n, axis in enumerate(kind):
                                add(axis, [int(fields[1])], float(fields[2 + n]) if explicit else None)
                        elif kind in ('distance', 'angle', 'dihedral'):
                            count = dict(distance=2, angle=3, dihedral=4)[kind]
                            if len(fields) != 1 + count + int(explicit):
                                raise ValueError('Wrong number of internal constraint fields')
                            add(kind, [int(i) for i in fields[1:1 + count]], float(fields[-1]) if explicit else None)
                        else:
                            raise ValueError('Unknown constraint type')
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f'Invalid geometry constraint {specification!r}: {exc}') from exc
        self.targets, self.factors = np.asarray(self.targets), np.asarray(self.factors)
        # Reject undefined dihedrals/linear angles before an electronic solve.
        self.jacobian(self.reference)

    @property
    def active(self):
        return bool(self.primitives)

    @property
    def has_cartesian(self):
        return any(e['type'] in ('x', 'y', 'z') for e in self.entries)

    def residual(self, coords):
        x = np.asarray(coords, dtype=float).ravel()
        values = np.asarray([p.value(x) for p in self.primitives]) - self.targets
        for i, entry in enumerate(self.entries):
            if entry['type'] == 'dihedral':
                values[i] = (values[i] + np.pi) % (2 * np.pi) - np.pi
        if not np.isfinite(values).all():
            raise ValueError('Geometry constraint is undefined at this structure')
        return values

    def jacobian(self, coords):
        x = np.asarray(coords, dtype=float).ravel()
        matrix = np.asarray([p.derivative(x).ravel() for p in self.primitives]).reshape(-1, x.size)
        if not np.isfinite(matrix).all():
            raise ValueError('Geometry constraint derivative is undefined (check linear angles/dihedrals)')
        return matrix

    def error(self, coords):
        return float(np.max(np.abs(self.residual(coords) * self.factors), initial=0.))

    def satisfied(self, coords):
        return self.error(coords) <= self.tolerance

    def project(self, coords, basis=None):
        """Nearest local feasible structure; conflicting nonlinear constraints fail."""
        x = np.asarray(coords, dtype=float).ravel().copy()
        basis = np.eye(x.size) if basis is None else basis
        for _ in range(100):
            if self.error(x) <= self.tolerance * .1:
                return x
            residual = self.residual(x)
            restricted = np.einsum('ij,jk->ik', self.jacobian(x), basis)
            step = np.linalg.lstsq(restricted, -residual, rcond=1e-10)[0]
            dx = np.einsum('ij,j->i', basis, step)
            largest = np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1))
            scale = min(1., .25 / max(largest, 1e-30))
            error = np.linalg.norm(residual)
            for _ in range(20):
                trial = x + scale * dx
                if np.linalg.norm(self.residual(trial)) < error:
                    x = trial
                    break
                scale *= .5
            else:
                break
        raise ValueError('Geometry constraints cannot be satisfied; check conflicting or singular constraints')

    def gauge_basis(self, coords):
        """Remove only rigid motions that preserve the Cartesian anchors.

        Removing all six rigid modes before freezing absolute positions would
        incorrectly remove some genuine internal relaxation directions.
        """
        xyz = np.asarray(coords).reshape(-1, 3)
        centered = xyz - xyz.mean(axis=0)
        rigid = np.column_stack([np.tile(a, (len(xyz), 1)).ravel() for a in np.eye(3)]
                                + [np.cross(a, centered).ravel() for a in np.eye(3)])
        u, s, _ = np.linalg.svd(rigid, full_matrices=False)
        rigid = u[:, s > 1e-10 * max(1., s[0])]
        allowed = np.einsum('ij,jk->ik', rigid, null_space(np.einsum('ij,jk->ik', self.jacobian(coords), rigid)))
        return null_space(allowed.T)

    def projected_gradient(self, coords, gradient):
        tangent = null_space(self.jacobian(coords))
        components = np.einsum('ij,i->j', tangent, np.asarray(gradient).ravel())
        return np.einsum('ij,j->i', tangent, components)

    def report(self, coords):
        residual = self.residual(coords) * self.factors
        return [dict(entry, error=float(error), unit='degree' if entry['type'] in ('angle', 'dihedral') else 'Angstrom')
                for entry, error in zip(self.entries, residual)]


def prepare_constraints(mol, *, freeze=None, set_values=None, tolerance=1e-7,
                        coordinates='auto', symmetry_operations=None, state=None):
    """Resolve freeze targets once, and satisfy explicit targets before energy calls."""
    limits = GeometryConstraints(mol.atom_coords(), freeze=freeze, set_values=set_values,
                                 tolerance=tolerance, state=state)
    if limits.active:
        basis = None
        if coordinates == 'symmetry':
            from .coordinates import vibrational_basis
            from .symmetry import invariant_displacements
            basis, _ = invariant_displacements(mol, vibrational_basis(mol.atom_coords()), symmetry_operations)
        projected = limits.project(mol.atom_coords(), basis=basis)
        mol = mol.copy()
        mol.set_geom_(projected.reshape(-1, 3), unit='Bohr')
    return mol, limits
