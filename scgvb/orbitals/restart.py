"""Transfer a saved SCGVB state between AO bases at an unchanged geometry."""
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
from pyscf import gto
from scgvb.spin.primitive import generate_branch_diagrams


def restart_orbitals(mol, space, path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    try:
        spec = data['molecule']
        old = gto.M(atom=spec['atoms'], basis=spec['basis'], unit=spec['coordinate_unit'],
                    charge=spec['charge'], spin=spec['multiplicity']-1, verbose=0)
        if (old.natm != mol.natm or old.charge != mol.charge or old.spin != mol.spin
                or [old.atom_symbol(i) for i in range(old.natm)] != [mol.atom_symbol(i) for i in range(mol.natm)]
                or not np.allclose(old.atom_coords(), mol.atom_coords(), atol=1e-7, rtol=0)):
            raise ValueError('Restart requires the same atoms, order, geometry, charge and multiplicity')
        ne = mol.nelectron if space is None else space.electrons
        diagrams = generate_branch_diagrams((ne+mol.spin)//2, (ne-mol.spin)//2)
        if not np.array_equal(diagrams, data['spin_branch_diagrams']):
            raise ValueError('Restart active electron count or spin basis does not match')
        def array(key, columns):
            a = np.asarray(data[key], dtype=float)
            if a.shape != (old.nao_nr(), columns) or not np.isfinite(a).all():
                raise ValueError(f'Invalid restart {key} shape or values')
            return a
        c_old = array('orbital_coefficients', ne)
        ncore = 0 if space is None else space.inactive.shape[1]
        core_old = array('inactive_orbital_coefficients', ncore)
        cs = np.asarray(data['spin_coefficients'], dtype=float)
        if cs.shape != (len(diagrams),) or not np.isfinite(cs).all() or np.linalg.norm(cs) == 0:
            raise ValueError('Invalid restart spin coefficients')
        if data['active_space'].get('frozen_inactive_orbitals', 0) != (0 if space is None else space.metadata.get('frozen_inactive_orbitals', 0)):
            raise ValueError('Restart frozen core partition does not match')
        s = mol.intor_symmetric('int1e_ovlp')
        cross = gto.intor_cross('int1e_ovlp', mol, old)
        # Least-squares projection: S_new C_new = S_new,old C_old.
        transform = np.linalg.solve(s, cross)
        c = np.einsum('ij,ja->ia', transform, c_old)
        def orthonormalize(a):
            gram = np.einsum('ia,ij,jb->ab', a, s, a)
            w, v = np.linalg.eigh(gram)
            if len(w) and w.min() < 1e-10:
                raise ValueError('Restart projection loses orbital rank in the target basis')
            return np.einsum('ia,ak,k,bk->ib', a, v, 1/np.sqrt(w), v)
        if space is not None:
            space = deepcopy(space)
            core = np.einsum('ij,ja->ia', transform, core_old)
            # Preserve the F subspace independently of the relaxable I subspace.
            nf = int(space.metadata.get('frozen_inactive_orbitals', 0))
            frozen = orthonormalize(core[:, :nf])
            rest = core[:, nf:]
            rest -= np.einsum('ia,ja,jk,kb->ib', frozen, frozen, s, rest)
            core = np.column_stack((frozen, orthonormalize(rest)))
            source_dimension = np.asarray(data['optimization_basis_coefficients']).shape[1]
            if source_dimension > space.basis.shape[1]:
                raise ValueError('Restart active orbital dimension exceeds the target dimension')
            basis_old = array('optimization_basis_coefficients', source_dimension)
            basis = np.einsum('ij,ja->ia', transform, basis_old)
            basis -= np.einsum('ia,ja,jk,kb->ib', core, core, s, basis)
            basis = orthonormalize(basis)
            missing = space.basis.shape[1] - basis.shape[1]
            if missing:
                occupied = np.column_stack((core, basis))
                # Fill added dimensions with the new selection's complement.
                extra = space.basis - np.einsum('ia,ja,jk,kb->ib', occupied, occupied, s, space.basis)
                gram = np.einsum('ia,ij,jb->ab', extra, s, extra)
                w, v = np.linalg.eigh(gram)
                if w[-missing] < 1e-10:
                    raise ValueError('Cannot extend the projected restart active space')
                extra = np.einsum('ia,ab,b->ib', extra, v[:, -missing:], 1/np.sqrt(w[-missing:]))
                basis = np.column_stack((basis, extra))
            c = np.einsum('ia,ij,jb->ab', basis, s, c)
            space.inactive, space.basis = core, basis
            space.metadata.update(restart_source=str(path), initial_space_from_restart=True)
        info = dict(method='json', source=str(path), source_basis=spec['basis'],
                    projection='AO overlap least squares', perturbation=0.)
        return space, c, cs, info
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f'Incomplete or invalid restart JSON: {exc}') from exc
