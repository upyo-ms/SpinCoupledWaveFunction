"""Physical orbital guesses; no constraints on subsequent SCWF optimization."""
import numpy as np
from scipy.linalg import eigh
from pyscf import scf


def initial_orbitals(mol, method='random', seed=40, perturbation=0.02):
    """Return one spatial orbital per electron and initialization metadata.

    Doubly occupied MOs give consecutive orbital pairs. Singly occupied MOs
    follow them. Perturbations have the requested S-metric norm before the
    final per-column normalization. Spin coupling coefficients are untouched.
    """
    if method not in ('random', 'core', 'hf'):
        raise ValueError('initial orbitals must be random, core or hf')
    if not np.isfinite(perturbation) or perturbation < 0:
        raise ValueError('initial perturbation must be finite and nonnegative')
    rng = np.random.RandomState(seed)
    n, ne = mol.nao_nr(), mol.nelectron
    info = {'method': method, 'seed': int(seed),
            'perturbation': float(perturbation) if method != 'random' else 0.}
    if method == 'random':
        return rng.rand(n, ne), info
    S = mol.intor('int1e_ovlp')
    if method == 'hf':
        mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Hartree-Fock initial orbital calculation did not converge')
        orbitals = np.asarray(mf.mo_coeff)
        occupations = np.asarray(mf.mo_occ)
        indices = np.r_[np.repeat(np.flatnonzero(occupations == 2), 2),
                        np.flatnonzero(occupations == 1)]
        info['reference'] = 'RHF' if mol.spin == 0 else 'ROHF'
    else:
        H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        _, orbitals = eigh(H, S)
        na = (ne + mol.spin) // 2
        nb = ne - na
        indices = np.r_[np.repeat(np.arange(nb), 2), np.arange(nb, na)]
        info['reference'] = 'core Hamiltonian'
    if len(indices) != ne:
        raise ValueError('Initial MO occupations do not match electron count')
    C, _ = orbital_guess_from_reference(orbitals, indices, S, method, seed, perturbation)
    return C, info


def orbital_guess_from_reference(orbitals, indices, S, method, seed, perturbation):
    """Shared initialization in AO or reduced active-space coordinates."""
    n, ne = len(S), len(indices)
    rng = np.random.RandomState(seed)
    info = {'method': method, 'seed': int(seed),
            'perturbation': float(perturbation) if method != 'random' else 0.}
    if method == 'random':
        return rng.rand(n, ne), info
    info['reference'] = 'HF active space' if method == 'hf' else 'core Hamiltonian'
    C = orbitals[:, indices].copy()
    if perturbation:
        # All MOs form an S-orthonormal basis, so this gives balanced AO noise
        # without favoring AO basis functions with a particular overlap scale.
        noise = rng.normal(size=(n, ne))
        noise /= np.linalg.norm(noise, axis=0)
        C += perturbation * np.einsum('ij,ja->ia', orbitals, noise)
    norms = np.einsum('ia,ij,ja->a', C, S, C)
    if not np.isfinite(C).all() or np.any(norms <= 0):
        raise FloatingPointError('Invalid initial orbital coefficients')
    C /= np.sqrt(norms)
    return C, info
