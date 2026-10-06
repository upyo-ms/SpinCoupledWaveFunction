"""Reproducible real, symmetric integral data for evaluator benchmarks."""
from types import SimpleNamespace
import numpy as np
from Spin_Function.SpinFuncDege import spin_function_dim


def synthetic_system(n=12, seed=23):
    from NonorthgonalEnergy import SCWF_Energy
    s = SCWF_Energy.__new__(SCWF_Energy)
    s.Ne, s.Sm, s.S = n, 1, np.eye(n)
    s.fs = spin_function_dim(n, 1, 0)
    s.mol = SimpleNamespace(spin=0)
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n,n))
    s.Hcore = (h+h.T)/n
    factors = rng.normal(size=(n,n,n)) / n
    factors = (factors + factors.transpose(0,2,1))/2
    s.JK = np.einsum('apq,ars->pqrs', factors, factors)
    s.energy_evaluation = 'determinant'
    s._determinant_evaluator = None
    s.energy_nuc = s.inactive_energy = 0.
    C = np.eye(n) + .1*rng.normal(size=(n,n)) / np.sqrt(n)
    cs = rng.normal(size=s.fs)
    return s, np.r_[C.ravel(), cs]

