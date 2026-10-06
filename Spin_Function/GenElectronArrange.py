import numpy as np
from functools import lru_cache
from itertools import permutations
from sympy.combinatorics import Permutation


def DiscElementT(PermuTrans):
    Disc = []

    for i in range(len(PermuTrans)):
        Disc.append(abs(PermuTrans[i][1]-PermuTrans[i][0]))

    Disc = np.array(Disc)

    return (all(Disc < 2))


def Decompose2ElementT(PermuTrans):
    if len(PermuTrans) == 0:
        PermuTrans.append(1)
    else:
        while DiscElementT(PermuTrans) == False:
            for i in range(len(PermuTrans)):
                if (PermuTrans[i][1]-PermuTrans[i][0]) > 1:
                    tmp = [(PermuTrans[i][1], PermuTrans[i][1]-1), (PermuTrans[i][0], PermuTrans[i][1]-1), (PermuTrans[i][1]-1, PermuTrans[i][1])]

                    PermuTrans[i:i+1] = tmp

    return PermuTrans


@lru_cache(maxsize=4)
def _permutation_data(Ne):
    """Immutable unique permutations, signs and adjacent factorizations."""
    if Ne < 1:
        raise ValueError("At least one electron is required")
    arrangements = tuple(permutations(range(Ne)))
    decompositions = {}
    signs = {}
    for order in arrangements:
        permutation = Permutation(list(order))
        decompositions[order] = tuple(Decompose2ElementT(permutation.transpositions()))
        signs[order] = (-1) ** permutation.parity()
    return arrangements, decompositions, signs


def _compose(left, right):
    # Preserve SymPy's original Dr * Dl convention exactly.
    return tuple(left[j] for j in right)


def electron_arrange(Ne):
    arrangements, decompositions, signs = _permutation_data(Ne)
    orders = [list(order) for order in arrangements]
    indices, transpositions, parity = [], [], []
    for i, left in enumerate(arrangements):
        inverse_left = np.argsort(left)
        for j, right in enumerate(arrangements):
            combined = tuple(right[k] for k in inverse_left)
            indices.append([orders[i], orders[j]])
            transpositions.append(list(decompositions[combined]))
            parity.append(signs[left] * signs[right])
    return indices, transpositions, parity
