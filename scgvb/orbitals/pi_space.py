"""AVAS-style selection using carbon 2p orbitals normal to a molecular plane.

The target projector uses MINAO reference orbitals, so it is independent of
molecular orientation and of the number of p shells in the calculation basis.
Global normals preserve planar selection. Local normals support twisted sp2 hydrocarbons.
"""
import numpy as np
from pyscf import gto


def local_pi_normals(coords, symbols):
    """Infer three sigma neighbours by explicit distance cutoffs, then fit a plane.

    These are geometric selection heuristics, not a bond-order assignment.
    Reject ambiguous coordination and strongly pyramidal carbon centres.
    """
    carbons = [i for i, symbol in enumerate(symbols) if symbol == 'C']
    normals, neighbours, deviations = [], [], []
    for atom in carbons:
        vectors = coords - coords[atom]
        distances = np.linalg.norm(vectors, axis=1)
        cutoffs = np.asarray([1.85 if symbol == 'C' else 1.30 for symbol in symbols])
        bonded = [j for j in range(len(symbols)) if j != atom and distances[j] <= cutoffs[j]]
        if len(bonded) != 3 or any(distances[j] < .5 for j in bonded):
            raise ValueError(f'Local pi carbon {atom+1} requires exactly three sigma neighbours; '
                             f'found {len(bonded)} with C-C <= 1.85 A / C-H <= 1.30 A')
        directions = vectors[bonded] / distances[bonded, None]
        _, singular, axes = np.linalg.svd(directions, full_matrices=False)
        if singular[1] < .2:
            raise ValueError(f'Local pi plane is ambiguous at carbon {atom+1}')
        normal = axes[-1]
        normal *= 1 if normal[np.argmax(np.abs(normal))] >= 0 else -1
        deviation = float(np.degrees(np.arcsin(np.clip(np.max(np.abs(directions @ normal)), 0, 1))))
        if deviation > 10.:
            raise ValueError(f'Local pi carbon {atom+1} is too pyramidal: '
                             f'sigma-bond deviation {deviation:.2f} degrees exceeds 10')
        normals.append(normal)
        neighbours.append(bonded)
        deviations.append(deviation)
    return np.asarray(normals), neighbours, deviations


def pi_targets(mol, normal_mode='global'):
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    if any(s not in ('C', 'H') for s in symbols) or symbols.count('C') < 2:
        raise ValueError('pi currently supports hydrocarbons with at least two carbons')
    coords = mol.atom_coords(unit='Angstrom')
    center = coords.mean(axis=0)
    _, singular, axes = np.linalg.svd(coords - center, full_matrices=False)
    if singular[1] < 1e-6:
        raise ValueError('Cannot determine a pi plane from a linear geometry')
    normal = axes[-1]
    normal *= 1 if normal[np.argmax(np.abs(normal))] >= 0 else -1
    residual = float(np.max(np.abs((coords-center) @ normal)))
    carbons = [i for i, s in enumerate(symbols) if s == 'C']
    if normal_mode not in ('global', 'local'):
        raise ValueError('pi normal mode must be global or local')
    metadata = {'pi_atoms_zero_based': carbons, 'pi_normal_mode': normal_mode,
                'pi_planarity_error_angstrom': residual, 'pi_target_basis': 'minao'}
    if normal_mode == 'global':
        if residual > 1e-4:
            raise ValueError('pi requires a planar geometry in global mode; '
                             'use --pi-normal-mode local for locally planar sp2 carbons')
        normals = np.tile(normal, (len(carbons), 1))
        metadata['pi_plane_normal'] = normal.tolist()
    else:
        normals, neighbours, deviations = local_pi_normals(coords, symbols)
        metadata.update(pi_local_normals=normals.tolist(),
                        pi_sigma_neighbours_zero_based=neighbours,
                        pi_sigma_bond_plane_deviation_degrees=deviations,
                        pi_bond_cutoffs_angstrom={'C-C': 1.85, 'C-H': 1.30},
                        pi_max_sigma_plane_deviation_degrees=10.)
    reference = mol.copy()
    reference.atom = mol._atom
    reference.unit = 'Bohr'
    reference.symmetry = False
    reference.basis = 'minao'
    reference.build(False, False)
    carbons = [i for i, s in enumerate(symbols) if s == 'C']
    targets = np.zeros((reference.nao_nr(), len(carbons)))
    labels = reference.ao_labels(fmt=False)
    for column, atom in enumerate(carbons):
        for component, direction in enumerate(('x', 'y', 'z')):
            matches = [i for i, label in enumerate(labels)
                       if label[0] == atom and label[2] == '2p' and label[3] == direction]
            if len(matches) != 1:
                raise ValueError('Could not identify carbon 2p targets in MINAO')
            targets[matches[0], column] = normals[column, component]
    sref = reference.intor_symmetric('int1e_ovlp')
    cross = gto.intor_cross('int1e_ovlp', reference, mol)
    target_s = np.einsum('ia,ij,jb->ab', targets, sref, targets)
    target_cross = np.einsum('ia,ij->aj', targets, cross)
    return target_s, target_cross, metadata


def select_pi_space(mf, targets, expected_electrons=None, threshold=.2):
    target_s, target_cross, metadata = targets
    mo, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    cross = np.einsum('ai,ij->aj', target_cross, mo)
    projector = np.einsum('ai,aj->ij', cross, np.linalg.solve(target_s, cross))
    inactive, selected, occupations, weights = [], [], [], {}
    for occupation, name in [(2, 'occupied'), (1, 'singly_occupied'), (0, 'virtual')]:
        indices = np.flatnonzero(occ == occupation)
        block = projector[np.ix_(indices, indices)]
        w, rotation = np.linalg.eigh(block)
        keep = w >= threshold
        if occupation == 1 and not np.all(keep):
            raise ValueError('All singly occupied reference orbitals must belong to the selected pi space')
        orbitals = np.einsum('ij,jk->ik', mo[:, indices], rotation)
        if occupation == 2:
            inactive = orbitals[:, ~keep]
        selected.append(orbitals[:, keep])
        occupations.extend([occupation] * int(keep.sum()))
        weights[name] = w.tolist()
    basis = np.column_stack(selected)
    electrons = int(sum(occupations))
    if basis.shape[1] != len(metadata['pi_atoms_zero_based']):
        raise ValueError(f'pi projector selected {basis.shape[1]} orbitals for '
                         f"{len(metadata['pi_atoms_zero_based'])} carbon targets; "
                         'a simple carbon pi space was not identified')
    if expected_electrons is not None and electrons != expected_electrons:
        raise ValueError(f'pi selection found {electrons} electrons, expected {expected_electrons}; '
                         'the electron count is checked, not forced')
    metadata.update(pi_projection_threshold=threshold, pi_projection_eigenvalues=weights,
                    pi_expected_electrons=None if expected_electrons is None else int(expected_electrons))
    return basis, inactive, electrons, np.asarray(occupations), metadata
