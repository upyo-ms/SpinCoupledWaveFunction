"""MINAO targets for active, inactive and permanently frozen MO subspaces."""
import numpy as np
from pyscf import gto


def ao_targets(mol, rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('AO selection requires a nonempty orbitals list')
    reference = mol.copy()
    reference.atom, reference.unit = mol._atom, 'Bohr'
    reference.symmetry, reference.basis = False, 'minao'
    reference.build(False, False)
    labels = reference.ao_labels(fmt=False)
    lookup = {(label[0]+1, label[2]+label[3]): i for i, label in enumerate(labels)}
    groups = {role: [] for role in ('A', 'I', 'F')}
    assignments, seen = [], set()
    for row in rows:
        parts = row.split() if isinstance(row, str) else []
        if len(parts) < 3 or not parts[0].isdigit() or parts[-1] not in groups:
            raise ValueError(f'Invalid AO specification {row!r}; use "1 2px 2py 2pz A" (A/I/F)')
        atom, role = int(parts[0]), parts[-1]
        if not 1 <= atom <= mol.natm:
            raise ValueError(f'AO atom index {atom} does not exist (1..{mol.natm})')
        for orbital in parts[1:-1]:
            # Match a shell label exactly: 2p expands to all its components,
            # while 2pz remains an individual component (never a prefix match).
            matches = [(label[0]+1, label[2]+label[3]) for label in labels
                       if label[0]+1 == atom and label[2] == orbital]
            if not matches and (atom, orbital) in lookup:
                matches = [(atom, orbital)]
            if not matches:
                available = ', '.join(k[1] for k in lookup if k[0] == atom)
                raise ValueError(f'AO {atom} {orbital} does not exist in MINAO for '
                                 f'{mol.atom_symbol(atom-1)}; available: {available}')
            for key in matches:
                if key in seen:
                    raise ValueError(f'Duplicate/conflicting AO assignment: {atom} {key[1]}')
                seen.add(key)
                groups[role].append(lookup[key])
                assignments.append({'atom': atom, 'orbital': key[1], 'role': role})
    s = reference.intor_symmetric('int1e_ovlp')
    cross = gto.intor_cross('int1e_ovlp', reference, mol)
    targets = {role: (s[np.ix_(indices, indices)], cross[indices])
               for role, indices in groups.items() if indices}
    return targets, assignments


def select_ao_space(mf, target_data, expected_electrons=None, threshold=.2):
    targets, assignments = target_data
    mo, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    projectors = {}
    for role, (s, cross) in targets.items():
        b = np.einsum('ai,ij->aj', cross, mo)
        projectors[role] = np.einsum('ai,aj->ij', b, np.linalg.solve(s, b))
    empty = np.empty((mo.shape[1], 0))
    occupied = np.eye(mo.shape[1])[:, occ == 2]
    single = np.eye(mo.shape[1])[:, occ == 1]
    virtual = np.eye(mo.shape[1])[:, occ == 0]
    weights = {}

    def split(vectors, role, name):
        p = projectors[role]
        block = np.einsum('ia,ij,jb->ab', vectors, p, vectors)
        w, u = np.linalg.eigh(block)
        weights[name] = w.tolist()
        rotated = np.einsum('ia,ab->ib', vectors, u)
        return rotated[:, w >= threshold], rotated[:, w < threshold]

    def conflict(vectors, role):
        if vectors.shape[1] and role in projectors:
            block = np.einsum('ia,ij,jb->ab', vectors, projectors[role], vectors)
            if np.linalg.eigvalsh(block).max() >= threshold:
                raise ValueError(f'AO targets for {role} overlap an incompatible selected space; '
                                 'revise the targets or projection threshold')

    frozen, inactive = empty, empty
    # F and I select only doubly occupied HF combinations. Neither may remove a SOMO.
    for role in ('F', 'I'):
        if role not in projectors:
            continue
        conflict(single, role)
        if role == 'I':
            conflict(frozen, role)
        chosen, occupied = split(occupied, role, role+'_occupied')
        if not chosen.shape[1]:
            raise ValueError(f'AO {role} targets select no doubly occupied orbitals')
        if role == 'F':
            frozen = chosen
        else:
            inactive = chosen
    if 'A' in projectors:
        conflict(np.column_stack((frozen, inactive)), 'A')
        active_occ, rest = split(occupied, 'A', 'A_occupied')
        active_single, rejected_single = split(single, 'A', 'A_singly_occupied')
        if rejected_single.shape[1]:
            raise ValueError('All singly occupied HF orbitals must belong to the AO active space')
        active_virtual, _ = split(virtual, 'A', 'A_virtual')
        inactive = np.column_stack((inactive, rest))
    else:
        active_occ, active_single, active_virtual = occupied, single, virtual
    active = np.column_stack((active_occ, active_single, active_virtual))
    occupations = np.r_[np.full(active_occ.shape[1], 2.), np.ones(active_single.shape[1]),
                        np.zeros(active_virtual.shape[1])]
    ne = int(occupations.sum())
    if expected_electrons is not None and ne != expected_electrons:
        raise ValueError(f'AO selection found {ne} active electrons, expected {expected_electrons}')
    core = np.column_stack((frozen, inactive))
    metadata = dict(ao_target_basis='minao', ao_assignments=assignments,
                    ao_projection_threshold=threshold, ao_projection_eigenvalues=weights,
                    ao_expected_electrons=expected_electrons,
                    frozen_inactive_orbitals=frozen.shape[1])
    return (np.einsum('ij,jk->ik', mo, active), np.einsum('ij,jk->ik', mo, core),
            ne, occupations, metadata)
