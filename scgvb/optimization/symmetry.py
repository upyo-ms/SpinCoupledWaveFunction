"""Finite point-symmetry constraints, independent of molecular connectivity.

Generators are orthogonal Cartesian matrices or signed-axis strings. Atomic
permutations are inferred by element and position; their invariant displacement
space is intersected with the complement of rigid translations and rotations.
"""
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def operation_matrix(text):
    """Read 'x,-y,z' or nine row-major matrix entries; never evaluate code."""
    if not isinstance(text, str):
        raise ValueError('Symmetry operations must be strings')
    parts = [part.strip() for part in text.split(',')]
    matches = [re.fullmatch(r'([+-]?)([xyz])', part) for part in parts]
    if len(parts) == 3 and all(matches):
        matrix = np.zeros((3, 3))
        for row, match in enumerate(matches):
            matrix[row, 'xyz'.index(match[2])] = -1. if match[1] == '-' else 1.
    else:
        try:
            values = [float(part) for part in text.replace(';', ' ').replace(',', ' ').split()]
            matrix = np.asarray(values).reshape(3, 3)
        except ValueError as exc:
            raise ValueError('Use signed axes x,-y,z or nine symmetry-matrix entries') from exc
    if (not np.isfinite(matrix).all()
            or not np.allclose(np.einsum('ki,kj->ij', matrix, matrix), np.eye(3), atol=1e-9, rtol=0)):
        raise ValueError('Symmetry operation must be a finite orthogonal 3 by 3 matrix')
    return matrix


def invariant_displacements(mol, vibrational, operations):
    """Return an orthonormal, symmetry-preserving Cartesian displacement basis."""
    if not operations:
        raise ValueError('Symmetry coordinates require symmetry_operations')
    xyz = mol.atom_coords()
    origin = xyz.mean(axis=0)
    centered = xyz - origin
    symbols = np.array([mol.atom_symbol(i) for i in range(mol.natm)])
    constraints, generators = [], []
    for text in operations:
        matrix = operation_matrix(text)
        transformed = np.einsum('ij,aj->ai', matrix, centered)
        permutation = np.empty(mol.natm, dtype=int)
        for symbol in sorted(set(symbols)):
            indices = np.flatnonzero(symbols == symbol)
            distances = np.linalg.norm(transformed[indices, None] - centered[None, indices], axis=2)
            rows, cols = linear_sum_assignment(distances)
            if np.max(distances[rows, cols]) > 1e-7:
                raise ValueError(f'Initial geometry does not obey symmetry operation {text!r}')
            if np.any(np.sum(distances < 1e-7, axis=1) != 1):
                raise ValueError('Symmetry operation has ambiguous atom mapping')
            permutation[indices[rows]] = indices[cols]
        block = np.zeros((3 * mol.natm, 3 * mol.natm))
        for i, j in enumerate(permutation):
            block[3*i:3*i+3, 3*j:3*j+3] += np.eye(3)
            block[3*i:3*i+3, 3*i:3*i+3] -= matrix
        constraints.append(block)
        generators.append(dict(operation=text, matrix=matrix.tolist(),
                               atom_permutation_one_based=(permutation + 1).tolist()))
    restricted = np.einsum('ij,jk->ik', np.vstack(constraints), vibrational)
    # The restricted constraint may be entirely zero (e.g. a symmetric
    # diatomic stretch). A relative-only rank test would classify roundoff
    # as a nonzero constraint and incorrectly remove that last variable.
    _, singular, right = np.linalg.svd(restricted, full_matrices=True)
    rank = np.count_nonzero(singular > 1e-10 * max(1., singular[0]))
    allowed = right[rank:].T
    basis = np.einsum('ij,jk->ik', vibrational, allowed)
    if basis.shape[1] == 0:
        raise ValueError('Symmetry constraints leave no internal geometry variables')
    return basis, dict(symmetry_operations=list(operations), symmetry_generators=generators,
                       symmetry_origin_bohr=origin.tolist(),
                       unconstrained_dimension=vibrational.shape[1])
