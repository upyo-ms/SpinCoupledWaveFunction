"""Automatic molecular coordinates and fixed local charts for energy-only DFO.

Cartesian coordinates and chart displacements are in Bohr. geomeTRIC owns
bond detection, ring/linear-angle primitives and fragment rotation coordinates.
"""
import numpy as np
from pyscf.lib import param

BOHR = param.BOHR  # Angstrom per Bohr


class CoordinateError(RuntimeError):
    """An internal-coordinate target could not be mapped to a valid structure."""


def vibrational_basis(coords):
    """Orthonormal Cartesian complement of rigid translation and rotation.

    SVD determines the rigid rank, including the five modes of a linear molecule.
    This is unweighted: the chart metric is Cartesian displacement in Bohr.
    """
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    centered = xyz - xyz.mean(axis=0)
    axes = np.eye(3)
    rigid = np.column_stack(
        [np.tile(axis, (len(xyz), 1)).ravel() for axis in axes]
        + [np.cross(axis, centered).ravel() for axis in axes])
    u, s, _ = np.linalg.svd(rigid, full_matrices=True)
    rank = np.count_nonzero(s > max(1., s[0]) * 1e-10)
    return u[:, rank:]


class GeometryCoordinates:
    """A deterministic local chart; never rebuild its basis during a DFO solve.

    The primitive B matrix is restricted to vibrational Cartesian directions.
    Its SVD removes redundant primitives and rigid modes, and scales independent
    combinations so the initial chart Jacobian has unit singular values. Thus
    a chart step has Cartesian displacement units even with mixed angles/lengths.
    """
    def __init__(self, mol, coordinates='auto', *, bonds=None, symmetry_operations=None, constraints=None):
        if coordinates not in ('auto', 'dlc', 'tric', 'cartesian', 'symmetry'):
            raise ValueError('coordinates must be auto, dlc, tric, cartesian or symmetry')
        if coordinates != 'symmetry' and symmetry_operations:
            raise ValueError('symmetry_operations require coordinates="symmetry"')
        if mol.natm < 2:
            raise ValueError('Geometry optimization requires at least two atoms')
        self.reference = mol.atom_coords().ravel().copy()
        self.cartesian_basis = vibrational_basis(self.reference)
        self.symmetry_operations = symmetry_operations
        self.constraints = constraints if constraints is not None and constraints.active else None
        self.constraint_ambient = None
        self.primitive = None
        if coordinates == 'symmetry':
            from .symmetry import invariant_displacements
            self.cartesian_basis, metadata = invariant_displacements(
                mol, self.cartesian_basis, symmetry_operations)
            self.dimension = self.cartesian_basis.shape[1]
            self.kind = 'symmetry'
            self.metadata = dict(coordinates=self.kind, dimension=self.dimension,
                                 chart_units='Bohr', **metadata)
            self._constrain()
            return
        try:
            from geometric.molecule import Molecule
            from geometric.internal import PrimitiveInternalCoordinates
        except ImportError as exc:
            raise ImportError('Geometry optimization requires geometric; install scgvb[geometry]') from exc
        self.reference = mol.atom_coords().ravel().copy()
        self.cartesian_basis = vibrational_basis(self.reference)
        self.dimension = self.cartesian_basis.shape[1]
        molecule = Molecule()
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        molecule.xyzs = [mol.atom_coords(unit='Angstrom')]
        molecule.comms = ['SCGVB geometry optimization']
        if bonds is not None:
            edges = set()
            for bond in bonds:
                try:
                    i, j = (int(v) - 1 for v in bond.split('-'))
                except (ValueError, AttributeError) as exc:
                    raise ValueError('geometry bonds must be one-based pairs such as 1-2') from exc
                if not (0 <= i < mol.natm and 0 <= j < mol.natm) or i == j:
                    raise ValueError(f'Invalid geometry bond: {bond}')
                edges.add(tuple(sorted((i, j))))
            molecule.bonds = sorted(edges)
            molecule.top_settings['read_bonds'] = True
        molecule.build_topology(force_bonds=False)
        self.molecule = molecule
        self.kind = ('dlc' if len(molecule.molecules) == 1 else 'tric') if coordinates == 'auto' else coordinates
        self.primitive = None
        primitives = []
        if self.kind != 'cartesian':
            self.primitive = PrimitiveInternalCoordinates(
                molecule, connect=self.kind == 'dlc', addcart=False, connect_isolated=False)
            b = self.primitive.wilsonB(self.reference)
            u, s, _ = np.linalg.svd(np.einsum('ij,jk->ik', b, self.cartesian_basis), full_matrices=False)
            if len(s) < self.dimension or s[-1] <= max(1., s[0]) * 1e-8:
                raise CoordinateError('Internal coordinates do not span all molecular motions; use cartesian or revise bonds')
            self.transform = (u[:, :self.dimension] / s[:self.dimension]).T
            primitives = [str(p) for p in self.primitive.Internals]
        self.metadata = dict(
            coordinates=self.kind, dimension=self.dimension,
            primitive_coordinates=primitives,
            bonds=[[int(i) + 1, int(j) + 1] for i, j in molecule.bonds],
            fragments=[[int(i) + 1 for i in sorted(fragment.nodes())] for fragment in molecule.molecules],
            chart_units='Bohr', rigid_modes_removed=3 * mol.natm - self.dimension)
        self._constrain()

    def _constrain(self):
        if self.constraints is None:
            return
        from .constraints import null_space
        if not self.constraints.satisfied(self.reference):
            raise ValueError('Coordinate reference does not satisfy geometry constraints')
        ambient = self.cartesian_basis
        if self.kind != 'symmetry' and self.constraints.has_cartesian:
            ambient = self.constraints.gauge_basis(self.reference)
        restricted = np.einsum('ij,jk->ik', self.constraints.jacobian(self.reference), ambient)
        tangent = np.einsum('ij,jk->ik', ambient, null_space(restricted))
        self.constraint_ambient = np.ascontiguousarray(ambient)
        self.cartesian_basis = np.ascontiguousarray(tangent)
        self.dimension = tangent.shape[1]
        if self.primitive is not None:
            restricted = np.einsum('ij,jk->ik', self.primitive.wilsonB(self.reference), tangent)
            u, s, _ = np.linalg.svd(restricted, full_matrices=False)
            if self.dimension and (len(s) < self.dimension or s[-1] <= max(1., s[0]) * 1e-8):
                raise CoordinateError('Constrained internal coordinates lost rank; use cartesian coordinates')
            self.transform = np.ascontiguousarray((u[:, :self.dimension] / s[:self.dimension]).T)
        self.metadata.update(dimension=self.dimension, constraints=self.constraints.entries,
                             constraint_tolerance=self.constraints.tolerance,
                             constraint_rank=ambient.shape[1] - self.dimension,
                             rigid_modes_removed=self.reference.size - ambient.shape[1])

    def from_cartesian(self, coords):
        x = np.asarray(coords, dtype=float).ravel()
        if self.primitive is None:
            return self.cartesian_basis.T @ (x - self.reference)
        return self.transform @ self.primitive.calcDiff(x, self.reference)

    def to_cartesian(self, displacement):
        y = np.asarray(displacement, dtype=float)
        if y.shape != (self.dimension,) or not np.isfinite(y).all():
            raise ValueError('Invalid internal-coordinate displacement')
        if self.dimension == 0:
            return self.reference.copy()
        if self.primitive is None and self.constraints is None:
            return self.reference + self.cartesian_basis @ y
        # Always start from the fixed reference, rather than the previous trial.
        # Otherwise the same BOBYQA variable can map to different geometries.
        x = self.reference.copy()
        for _ in range(80):
            residual = y - self.from_cartesian(x)
            feasible = self.constraints is None or self.constraints.satisfied(x)
            if self.constraints is not None:
                residual = np.r_[residual, -self.constraints.residual(x)]
            error = np.linalg.norm(residual)
            if error <= 1e-8 and feasible:
                return x
            # Fix the Cartesian gauge as well as the internal basis. Recomputing
            # it would drop a bending direction when a linear reference bends.
            tangent = self.cartesian_basis if self.constraints is None else self.constraint_ambient
            internal_jacobian = (self.cartesian_basis.T if self.primitive is None else
                                 np.einsum('ij,jk->ik', self.transform, self.primitive.wilsonB(x)))
            if self.constraints is not None:
                internal_jacobian = np.vstack((internal_jacobian, self.constraints.jacobian(x)))
            jacobian = np.einsum('ij,jk->ik', internal_jacobian, tangent)
            dy, _, rank, _ = np.linalg.lstsq(jacobian, residual, rcond=1e-10)
            if rank != tangent.shape[1]:
                raise CoordinateError('Internal-coordinate inverse lost rank')
            dx = tangent @ dy
            largest = np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1))
            scale = min(1., .2 / max(largest, 1e-30))
            for _ in range(16):
                trial = x + scale * dx
                trial_residual = y - self.from_cartesian(trial)
                if self.constraints is not None:
                    trial_residual = np.r_[trial_residual, -self.constraints.residual(trial)]
                trial_error = np.linalg.norm(trial_residual)
                if np.isfinite(trial_error) and trial_error < error:
                    x = trial
                    break
                scale *= .5
            else:
                raise CoordinateError('Internal-coordinate inverse failed to reduce its residual')
        raise CoordinateError('Internal-coordinate inverse did not converge')

    def optimizer_coordinates(self):
        """geomeTRIC's coordinate object for the gradient-based optimizer."""
        if self.kind == 'symmetry':
            raise ValueError('Symmetry coordinates currently require BOBYQA')
        from geometric.internal import CartesianCoordinates, DelocalizedInternalCoordinates
        if self.constraints is not None:
            limits = self.constraints
            from scipy.linalg import qr
            constraint_jacobian = limits.jacobian(self.reference)
            _, triangular, pivots = qr(constraint_jacobian.T, mode='economic', pivoting=True)
            diagonal = np.abs(np.diag(triangular))
            rank = np.count_nonzero(diagonal > 1e-10 * max(1., np.max(diagonal, initial=0.)))
            independent = sorted(pivots[:rank])

            class ConstrainedDLC(DelocalizedInternalCoordinates):
                def newCartesian_withConstraint(self, xyz, dQ, thre=.1, verbose=0):
                    trial = super().newCartesian_withConstraint(xyz, dQ, thre, verbose)
                    return limits.project(trial)

            # Absolute Cartesian anchors require translation/rotation coordinates
            # too; HDLC spans all 3N directions without changing the optimizer.
            addcart = limits.has_cartesian or self.kind == 'cartesian'
            return ConstrainedDLC(
                self.molecule, build=True, connect=self.kind == 'dlc' and not addcart,
                addcart=addcart, connect_isolated=False, conmethod=1,
                constraints=[limits.primitives[i] for i in independent],
                cvals=limits.targets[independent].tolist())
        if self.kind == 'cartesian':
            return CartesianCoordinates(self.molecule)
        return DelocalizedInternalCoordinates(
            self.molecule, build=True, connect=self.kind == 'dlc', addcart=False,
            connect_isolated=False)

    def state(self):
        """Serialize the fixed chart, including its exact floating-point basis."""
        return dict(coordinates=self.kind, reference_bohr=self.reference.tolist(),
                    cartesian_basis=self.cartesian_basis.tolist(),
                    transform=None if self.primitive is None else self.transform.tolist(),
                    symmetry_operations=self.symmetry_operations,
                    constraint_ambient=None if self.constraint_ambient is None else self.constraint_ambient.tolist(),
                    constraints=None if self.constraints is None else self.constraints.entries)

    def restore_state(self, state):
        basis = np.asarray(state['cartesian_basis'], dtype=float)
        reference = np.asarray(state['reference_bohr'], dtype=float)
        if (state['coordinates'] != self.kind or basis.shape != self.cartesian_basis.shape
                or reference.shape != self.reference.shape or not np.isfinite(basis).all()
                or not np.allclose(reference, self.reference, atol=1e-10, rtol=0)
                or not np.allclose(np.einsum('ki,kj->ij', basis, basis), np.eye(self.dimension), atol=1e-9)
                or not np.allclose(np.einsum('ik,jk->ij', basis, basis),
                                   np.einsum('ik,jk->ij', self.cartesian_basis, self.cartesian_basis), atol=1e-8)):
            raise ValueError('Restart coordinate chart is incompatible with its geometry or symmetry')
        self.reference, self.cartesian_basis = reference.copy(), basis.copy()
        if self.constraints is not None:
            ambient = np.asarray(state.get('constraint_ambient'), dtype=float)
            if (state.get('constraints') != self.constraints.entries
                    or ambient.shape != self.constraint_ambient.shape or not np.isfinite(ambient).all()
                    or not np.allclose(np.einsum('ik,jk->ij', ambient, ambient),
                                       np.einsum('ik,jk->ij', self.constraint_ambient, self.constraint_ambient), atol=1e-8)):
                raise ValueError('Restart coordinate constraints are incompatible')
            self.constraint_ambient = ambient.copy()
        if self.primitive is not None:
            transform = np.asarray(state['transform'], dtype=float)
            if transform.shape != self.transform.shape or not np.isfinite(transform).all():
                raise ValueError('Invalid restart internal-coordinate transform')
            self.transform = transform.copy()
