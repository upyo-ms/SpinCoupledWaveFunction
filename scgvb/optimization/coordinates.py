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
    def __init__(self, mol, coordinates='auto', *, bonds=None):
        if coordinates not in ('auto', 'dlc', 'tric', 'cartesian'):
            raise ValueError('coordinates must be auto, dlc, tric or cartesian')
        try:
            from geometric.molecule import Molecule
            from geometric.internal import PrimitiveInternalCoordinates
        except ImportError as exc:
            raise ImportError('Geometry optimization requires geometric; install scgvb[geometry]') from exc
        if mol.natm < 2:
            raise ValueError('Geometry optimization requires at least two atoms')
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

    def from_cartesian(self, coords):
        x = np.asarray(coords, dtype=float).ravel()
        if self.primitive is None:
            return self.cartesian_basis.T @ (x - self.reference)
        return self.transform @ self.primitive.calcDiff(x, self.reference)

    def to_cartesian(self, displacement):
        y = np.asarray(displacement, dtype=float)
        if y.shape != (self.dimension,) or not np.isfinite(y).all():
            raise ValueError('Invalid internal-coordinate displacement')
        if self.primitive is None:
            return self.reference + self.cartesian_basis @ y
        # Always start from the fixed reference, rather than the previous trial.
        # Otherwise the same BOBYQA variable can map to different geometries.
        x = self.reference.copy()
        for _ in range(80):
            residual = y - self.from_cartesian(x)
            error = np.linalg.norm(residual)
            if error <= 1e-8:
                return x
            # Fix the Cartesian gauge as well as the internal basis. Recomputing
            # it would drop a bending direction when a linear reference bends.
            tangent = self.cartesian_basis
            internal_jacobian = np.einsum('ij,jk->ik', self.transform, self.primitive.wilsonB(x))
            jacobian = np.einsum('ij,jk->ik', internal_jacobian, tangent)
            dy, _, rank, _ = np.linalg.lstsq(jacobian, residual, rcond=1e-10)
            if rank != self.dimension:
                raise CoordinateError('Internal-coordinate inverse lost rank')
            dx = tangent @ dy
            largest = np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1))
            scale = min(1., .2 / max(largest, 1e-30))
            for _ in range(16):
                trial = x + scale * dx
                trial_error = np.linalg.norm(y - self.from_cartesian(trial))
                if np.isfinite(trial_error) and trial_error < error:
                    x = trial
                    break
                scale *= .5
            else:
                raise CoordinateError('Internal-coordinate inverse failed to reduce its residual')
        raise CoordinateError('Internal-coordinate inverse did not converge')

    def optimizer_coordinates(self):
        """geomeTRIC's coordinate object for the gradient-based optimizer."""
        from geometric.internal import CartesianCoordinates, DelocalizedInternalCoordinates
        if self.kind == 'cartesian':
            return CartesianCoordinates(self.molecule)
        return DelocalizedInternalCoordinates(
            self.molecule, build=True, connect=self.kind == 'dlc', addcart=False,
            connect_isolated=False)
