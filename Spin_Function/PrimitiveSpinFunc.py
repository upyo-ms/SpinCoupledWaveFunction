"""Primitive spin functions in the historical branching-diagram order."""
from itertools import combinations
import math

import numpy as np


def Coeff_Calc(BD, BDz, PD, PDz):
    """Accumulate the original coupling factors, vectorized over both bases."""
    coeff = np.ones((len(PD), len(BD)))
    for k in range(BD.shape[1]):
        up = BD[:, k] == 1
        z = BDz[:, k]
        p = PD[:, k, None]
        pz = PDz[:, k, None]
        numerator = np.where(up, z + p * pz, z - p * pz + 1)
        denominator = np.where(up, 2 * z, 2 * z + 2)
        ratio = numerator / denominator
        # The legacy implementation leaves negative factors unchanged.
        # Keep that convention and the original electron-wise product order.
        factor = np.sqrt(np.maximum(ratio, 0))
        factor = np.where((~up) & (p == 1), -factor, factor)
        coeff *= np.where(ratio >= 0, factor, 1.0)
    return coeff


def BranchDia(PD, PDz):
    valid = np.all(PDz >= 0, axis=1)
    return PD[valid], PDz[valid]


def PriSpinfunc(Np, Na, Nb, I_b):
    beta = np.asarray(I_b, dtype=int).reshape(Np, Nb) - 1
    pd = np.ones((Np, Na + Nb))
    pd[np.arange(Np)[:, None], beta] = -1
    return pd, np.cumsum(pd, axis=1) / 2


def _primitive_diagrams(Na, Nb):
    # Descending combinations reproduce the old counter's reverse colex order.
    n = Na + Nb
    count = math.comb(n, Nb)
    beta = np.asarray(list(combinations(range(n, 0, -1), Nb)), dtype=int)
    return PriSpinfunc(count, Na, Nb, beta)


def generate_branch_diagrams(Na, Nb):
    """Build just the branch basis, without the unused coefficient matrix."""
    pd, pdz = _primitive_diagrams(Na, Nb)
    return BranchDia(pd, pdz)[0]


def PrSpinFunc_gen(Na, Nb):
    pd, pdz = _primitive_diagrams(Na, Nb)
    bd, bdz = BranchDia(pd, pdz)
    return bd, pd, Coeff_Calc(bd, bdz, pd, pdz)
