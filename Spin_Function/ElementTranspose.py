"""Adjacent-transposition matrices in the branching-diagram basis."""
import numpy as np


def elementary_transposition(Ne, fs, BD):
    if Ne < 1:
        raise ValueError('At least one electron is required')
    bd = np.asarray(BD)
    if bd.shape != (fs, Ne):
        raise ValueError('Branch-diagram shape must be (fs, Ne)')
    matrices = {'E': np.eye(fs)}
    # Young-table coordinates: alpha -> count, beta -> count - 1.
    alpha = bd == 1
    positions = np.where(alpha, np.cumsum(alpha, axis=1),
                         np.cumsum(~alpha, axis=1) - 1)
    diagonal = -1.0 / (positions[:, :-1] - positions[:, 1:])
    lookup = {tuple(row): i for i, row in enumerate(bd)}
    rows = np.arange(fs)
    for k in range(Ne - 1):
        matrix = np.zeros((fs, fs))
        matrix[rows, rows] = diagonal[:, k]
        for i, row in enumerate(bd):
            swapped = list(row)
            swapped[k], swapped[k + 1] = swapped[k + 1], swapped[k]
            j = lookup.get(tuple(swapped))
            if j is not None and j != i:
                matrix[i, j] = np.sqrt(1 - diagonal[i, k] ** 2)
        matrices[f'{k},{k + 1}'] = matrix
    # These real symmetric transpositions are involutions; no inversion needed.
    for k in range(Ne - 1):
        matrices[f'{k + 1},{k}'] = matrices[f'{k},{k + 1}'].copy()
    return matrices
