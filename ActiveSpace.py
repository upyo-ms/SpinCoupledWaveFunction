"""Fixed doubly occupied orbitals and an S-orthonormal optimization space.

Python MO indices are zero-based. This module prepares the initial spaces;
optional inactive orbital relaxation is implemented in CoreOptimization.py.
"""
from dataclasses import dataclass
import numpy as np
from pyscf import scf, ao2mo
from Spin_Function.SpinFuncDege import spin_function_dim


@dataclass
class ActiveSpace:
    basis: np.ndarray
    inactive: np.ndarray
    electrons: int
    occupations: np.ndarray
    metadata: dict

    def integrals(self, mol):
        h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        density = 2 * np.einsum('pi,qi->pq', self.inactive, self.inactive)
        j, k = scf.hf.get_jk(mol, density)
        potential = j - .5 * k
        constant = np.einsum('ij,ji->', density, h + .5 * potential)
        effective_h = np.einsum('ip,ij,jq->pq', self.basis, h + potential, self.basis)
        eri = ao2mo.kernel(mol, self.basis, compact=False).reshape((self.basis.shape[1],) * 4)
        return effective_h, eri, float(constant)

    def initial_orbitals(self, method, seed, perturbation, h):
        from InitialOrbitals import orbital_guess_from_reference
        if method == 'core':
            _, orbitals = np.linalg.eigh(h)
            nb = int(np.count_nonzero(self.occupations == 2))
            na = self.electrons - nb
            indices = np.r_[np.repeat(np.arange(nb), 2), np.arange(nb, na)]
        else:
            orbitals = np.eye(self.basis.shape[1])
            indices = np.r_[np.repeat(np.flatnonzero(self.occupations == 2), 2),
                            np.flatnonzero(self.occupations == 1)]
        return orbital_guess_from_reference(orbitals, indices, np.eye(len(orbitals)),
                                            method, seed, perturbation)


def _indices(values, n, name):
    values = list(values)
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n for i in values):
        raise ValueError(f'{name} must contain zero-based MO indices in [0, {n})')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} contains duplicate indices')
    return values


def prepare_active_space(mol, frozen_core=0, inactive_mos=None, active_mos=None,
                         active_space='full', pi_electrons=None, pi_normal_mode='global',
                         orbitals=None, ao_electrons=None, ao_threshold=.2):
    if active_space not in ('full', 'benzene-pi', 'pi', 'ao'):
        raise ValueError('active_space must be full, benzene-pi, pi or ao')
    if pi_normal_mode not in ('global', 'local') or (pi_normal_mode != 'global' and active_space != 'pi'):
        raise ValueError('Local pi normals require active_space=pi')
    if pi_electrons is not None:
        if active_space != 'pi' or not isinstance(pi_electrons, (int, np.integer)) or pi_electrons < 2:
            raise ValueError('pi_electrons requires active_space=pi and an integer >= 2')
    if not isinstance(frozen_core, (int, np.integer)) or frozen_core < 0:
        raise ValueError('frozen_core must be a nonnegative number of spatial orbitals')
    if frozen_core and inactive_mos is not None:
        raise ValueError('Use frozen_core or inactive_mos, not both')
    ao_data = None
    if active_space == 'ao':
        if frozen_core or inactive_mos is not None or active_mos is not None:
            raise ValueError('AO selection cannot be combined with manual MO selection')
        if (not np.isfinite(ao_threshold) or not 0 < ao_threshold < 1):
            raise ValueError('ao_threshold must be strictly between 0 and 1')
        if ao_electrons is not None and (type(ao_electrons) is not int or ao_electrons < 2):
            raise ValueError('ao_electrons must be an integer >= 2')
        from AOSpace import ao_targets
        ao_data = ao_targets(mol, orbitals)  # Validate labels before starting HF.
    elif orbitals is not None or ao_electrons is not None or ao_threshold != .2:
        raise ValueError('orbitals/ao_electrons/ao_threshold require active_space=ao')
    if active_space == 'full' and frozen_core == 0 and inactive_mos is None and active_mos is None:
        return None
    if active_space == 'benzene-pi':
        if frozen_core or inactive_mos is not None or active_mos is not None:
            raise ValueError('benzene-pi cannot be combined with manual MO selection')
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        if (symbols.count('C') != 6 or symbols.count('H') != 6 or len(symbols) != 12
                or mol.charge != 0 or mol.spin != 0):
            raise ValueError('benzene-pi requires neutral singlet C6H6')
        if np.ptp(mol.atom_coords()[:, 2]) > 1e-6:
            raise ValueError('benzene-pi requires the molecule in the xy plane (C 2pz targets)')
    targets = None
    if active_space == 'pi':
        if frozen_core or inactive_mos is not None or active_mos is not None:
            raise ValueError('pi cannot be combined with manual MO selection')
        from PiSpace import pi_targets
        targets = pi_targets(mol, pi_normal_mode)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('HF for active-space selection did not converge')
    mo, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    n = mo.shape[1]
    metadata = {'selection': active_space, 'reference': 'RHF' if mol.spin == 0 else 'ROHF',
                'reference_energy_hartree': float(mf.e_tot), 'inactive_orbitals_optimized': False}
    if active_space == 'ao':
        from AOSpace import select_ao_space
        basis, inactive, ne, active_occ, ao_metadata = select_ao_space(mf, ao_data, ao_electrons, ao_threshold)
        metadata.update(ao_metadata)
    elif active_space == 'pi':
        from PiSpace import select_pi_space
        basis, inactive, ne, active_occ, pi_metadata = select_pi_space(mf, targets, pi_electrons)
        metadata.update(pi_metadata)
    elif active_space == 'benzene-pi':
        from pyscf.mcscf import avas
        nact, ne, mo = avas.avas(mf, ['C 2pz'], threshold=.2)
        if nact != 6 or ne != 6:
            raise ValueError(f'AVAS did not identify a 6e/6o pi space (got {ne}e/{nact}o)')
        ni = (mol.nelectron - ne) // 2
        inactive = mo[:, :ni]
        basis = mo[:, ni:ni+nact]
        active_occ = np.r_[np.full(ne // 2, 2.), np.zeros(nact - ne // 2)]
        metadata.update(avas_targets=['C 2pz'], avas_threshold=.2)
    else:
        ii = _indices(range(frozen_core) if inactive_mos is None else inactive_mos, n, 'inactive_mos')
        aa = _indices([i for i in range(n) if i not in ii] if active_mos is None else active_mos,
                      n, 'active_mos')
        if set(ii) & set(aa):
            raise ValueError('Active and inactive MO selections overlap')
        if any(occ[i] != 2 for i in ii):
            raise ValueError('Every inactive MO must be doubly occupied in the HF reference')
        if any(occ[i] > 0 for i in range(n) if i not in ii and i not in aa):
            raise ValueError('Every occupied HF orbital must be active or inactive')
        inactive, basis, active_occ = mo[:, ii], mo[:, aa], occ[aa]
        ne = mol.nelectron - 2 * len(ii)
        metadata.update(selection='manual', inactive_mos_zero_based=ii, active_mos_zero_based=aa)
    if ne < 2:
        raise ValueError('At least two active electrons are required')
    spin_function_dim(ne, mol.spin + 1, 0)
    if basis.shape[1] < (ne + mol.spin) // 2:
        raise ValueError('Too few active spatial basis functions')
    metadata.update(active_electrons=int(ne), active_orbital_dimension=int(basis.shape[1]),
                    inactive_orbitals=int(inactive.shape[1]))
    return ActiveSpace(basis, inactive, int(ne), active_occ, metadata)
