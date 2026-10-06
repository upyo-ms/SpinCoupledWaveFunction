"""Spin representations with the original basis and permutation ordering."""
from functools import lru_cache

import numpy as np

from .dimensions import spin_function_dim
from .primitive import generate_branch_diagrams
from .transpositions import elementary_transposition
from .arrangements import _permutation_data


@lru_cache(maxsize=4)
def _representations(Ne, Sm):
    fs = spin_function_dim(Ne, Sm, 0)
    na, nb = (Ne + Sm - 1) // 2, (Ne - Sm + 1) // 2
    bd = generate_branch_diagrams(na, nb)
    elementary = elementary_transposition(Ne, fs, bd)
    arrangements, decompositions, signs = _permutation_data(Ne)
    representations = {}
    for order in arrangements:
        matrix = np.eye(fs)
        for swap in decompositions[order]:
            if swap != 1:
                matrix = elementary[f'{swap[0]},{swap[1]}'] @ matrix
        matrix *= signs[order]
        matrix.setflags(write=False)
        representations[order] = matrix
    return fs, arrangements, representations


def GenerateElecArrangement_SymRe(Ne, Sm):
    # Validate before conversion so fractional quantum numbers are not truncated.
    spin_function_dim(Ne, Sm, 0)
    Ne, Sm = int(Ne), int(Sm)
    fs, arrangements, representations = _representations(Ne, Sm)
    orders = [list(order) for order in arrangements]
    dn, indices = [], []
    for i, left in enumerate(arrangements):
        inverse_left = np.argsort(left)
        for j, right in enumerate(arrangements):
            # Retain the public list/np.matrix API and independent matrix storage.
            # Signed spin overlap <P_left chi | P_right chi>:
            # U(left).T @ U(right), not U(left) @ U(right).
            relative = tuple(right[k] for k in inverse_left)
            dn.append(np.matrix(representations[relative], copy=True))
            indices.append([orders[i], orders[j]])
    return fs, dn, indices
