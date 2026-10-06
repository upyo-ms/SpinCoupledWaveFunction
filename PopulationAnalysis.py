"""Mulliken, Lowdin and meta-Lowdin charge and spin populations from the normalized SCGVB density."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from pyscf import fci
from DeterminantEnergy import make_ci_builder


def scgvb_density(overlap, C, Cs, spin, inactive=None, *, determinant_size_profile='standard'):
    """Return AO alpha/beta densities; inactive orbitals stay doubly occupied.

    An exact QR representation in the span of C bounds the determinant space
    by the electron count. No singular vectors or orbital directions are dropped,
    including when C is rank deficient. CI amplitudes are derived, never optimized.
    """
    S, C, Cs = np.asarray(overlap), np.asarray(C), np.asarray(Cs)
    if C.ndim != 2 or C.shape[0] != len(S):
        raise ValueError('Invalid AO orbital coefficients for population analysis')
    if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in (S, C, Cs)):
        raise ValueError('Population analysis requires finite real coefficients')
    core = np.zeros((len(S), 0)) if inactive is None else np.asarray(inactive)
    if (core.ndim != 2 or core.shape[0] != len(S) or np.iscomplexobj(core)
            or not np.isfinite(core).all()):
        raise ValueError('Invalid inactive orbital coefficients')
    if not np.allclose(np.einsum('pi,pq,qj->ij', core, S, core), np.eye(core.shape[1]), atol=1e-8, rtol=0.):
        raise ValueError('Inactive orbitals are not S-orthonormal')
    # Normalize columns only for checking orthogonality; their scales are redundant.
    norms = np.sqrt(np.einsum('pi,pq,qi->i', C, S, C))
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('Invalid active orbital norms')
    if not np.allclose(np.einsum('pi,pq,qj->ij', core, S, C / norms), 0., atol=1e-8, rtol=0.):
        raise ValueError('Core and active orbitals are not S-orthogonal')
    # Some SciPy/LAPACK builds leave entries in the unused upper triangle
    # despite cholesky(clean=True). A dense product must not consume them:
    # solve_triangular below ignores that triangle, which otherwise changes
    # the represented wavefunction while still conserving electron count.
    L = np.tril(cholesky(S, lower=True))
    if not np.allclose(np.einsum('ik,jk->ij', L, L), S, atol=1e-10, rtol=1e-10):
        raise RuntimeError('Population overlap factorization failed reconstruction')
    Q, R = np.linalg.qr(np.einsum('ji,jk->ik', L, C), mode='reduced')
    basis = solve_triangular(L.T, Q, lower=False)
    if not np.allclose(np.einsum('ij,jk->ik', basis, R), C, atol=1e-10, rtol=1e-10):
        raise RuntimeError('Population orbital transformation changed the SCGVB state')
    ne, n = C.shape[1], Q.shape[1]
    system = SimpleNamespace(S=np.eye(n), Ne=ne, Sm=spin + 1, fs=len(Cs), determinant_size_profile=determinant_size_profile)
    ci = make_ci_builder(system)(R, Cs)
    norm = np.linalg.norm(ci)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('SCGVB wavefunction has zero or invalid norm')
    ci /= norm
    na = (ne + spin) // 2
    da, db = fci.direct_spin1.make_rdm1s(ci, n, (na, ne-na))
    core_density = np.einsum('pi,qi->pq', core, core)
    return tuple(np.einsum('pi,ij,qj->pq', basis, d, basis) + core_density for d in (da, db))


def mulliken_analysis(mol, C, Cs, inactive=None, *, determinant_size_profile='standard'):
    S = mol.intor('int1e_ovlp')
    da, db = scgvb_density(S, C, Cs, mol.spin, inactive, determinant_size_profile=determinant_size_profile)
    # diag(D S), partitioned using PySCF's AO-to-atom slices.
    population = np.einsum('ij,ji->i', da + db, S)
    spin_population = np.einsum('ij,ji->i', da - db, S)
    return _atomic_populations(mol, population, spin_population, 'Mulliken')


def _atomic_populations(mol, population, spin_population, method):
    electron_error = float(population.sum() - mol.nelectron)
    spin_error = float(spin_population.sum() - mol.spin)
    if (not np.isfinite(population).all() or not np.isfinite(spin_population).all()
            or abs(electron_error) > 1e-7 or abs(spin_error) > 1e-7):
        raise RuntimeError(f'{method} density failed electron/spin count conservation')
    atoms = []
    for a, (_, _, start, stop) in enumerate(mol.aoslice_by_atom()):
        electrons = float(population[start:stop].sum())
        atoms.append({'index': a + 1, 'symbol': mol.atom_symbol(a),
                      'electron_population': electrons,
                      'charge': float(mol.atom_charge(a) - electrons),
                      'spin_population': float(spin_population[start:stop].sum())})
    return {'method': method, 'analysis_version': 2,
            'density_source': 'normalized SCGVB plus doubly occupied inactive orbitals',
            'charge_unit': 'e', 'spin_convention': 'N_alpha - N_beta (2 M_S)',
            'atoms': atoms, 'total_charge': float(sum(a['charge'] for a in atoms)),
            'total_electron_population': float(population.sum()),
            'total_spin_population': float(spin_population.sum()),
            'electron_count_error': electron_error, 'spin_count_error': spin_error}


def bond_order_analysis(mol, density, spin_density, overlap, lowdin_root):
    """Real 1-RDM indices: spin-corrected Mayer and total-density Wiberg-Lowdin.

    Conventions follow ORCA manual 6.1, population analysis Eqs. 5.12/5.13.
    For correlated states these are density-based indices, not pair cumulants.
    """
    ps = np.einsum('ij,jk->ik', density, overlap)
    rs = np.einsum('ij,jk->ik', spin_density, overlap)
    pl = np.einsum('ip,pq,jq->ij', lowdin_root, density, lowdin_root)
    kernels = {'mayer': ps * ps.T + rs * rs.T, 'wiberg_lowdin': pl * pl.T}
    slices = mol.aoslice_by_atom()[:, 2:]
    matrices = {}
    for key, kernel in kernels.items():
        matrix = np.zeros((mol.natm, mol.natm))
        for a, (start, stop) in enumerate(slices):
            for b in range(a+1, mol.natm):
                first, last = slices[b]
                matrix[a, b] = matrix[b, a] = kernel[start:stop, first:last].sum()
        if not np.isfinite(matrix).all():
            raise RuntimeError('Nonfinite bond-order matrix')
        matrices[key] = matrix.tolist()
    pairs = [{'atom1': a+1, 'symbol1': mol.atom_symbol(a),
              'atom2': b+1, 'symbol2': mol.atom_symbol(b),
              **{key: matrix[a][b] for key, matrix in matrices.items()}}
             for a in range(mol.natm) for b in range(a+1, mol.natm)]
    return {'analysis_version': 1, 'atom_indexing': 'one-based, input atom order',
            'density_source': 'normalized SCGVB including inactive electrons',
            'definitions': {
                'mayer': 'sum_A,B [(P S)_mu,nu (P S)_nu,mu + (R S)_mu,nu (R S)_nu,mu]; R=P_alpha-P_beta',
                'wiberg_lowdin': 'sum_A,B (P_L)_mu,nu (P_L)_nu,mu; P_L=S^(1/2) P S^(1/2); P=P_alpha+P_beta'},
            'matrix_diagonal': 'zero; intra-atomic indices are not reported',
            'print_threshold': .05, 'matrices': matrices, 'pairs': pairs}


def population_analysis(mol, C, Cs, inactive=None, *, determinant_size_profile='standard'):
    """Analyze one shared SCGVB density in three AO population partitions.

    Legacy top-level fields retain Mulliken values; `methods` holds all results.
    Lowdin means symmetric S**(-1/2), without atomic pre-orthogonalization.
    Meta-Lowdin uses PySCF's ANO pre-orthogonalization explicitly.
    """
    from pyscf.lo import orth
    S = mol.intor('int1e_ovlp')
    da, db = scgvb_density(S, C, Cs, mol.spin, inactive, determinant_size_profile=determinant_size_profile)
    density, spin_density = da + db, da - db
    methods = {'mulliken': _atomic_populations(
        mol, np.einsum('ij,ji->i', density, S),
        np.einsum('ij,ji->i', spin_density, S), 'Mulliken')}
    w, v = np.linalg.eigh(S)
    if np.any(w <= 0):
        raise ValueError('Lowdin analysis requires positive-definite AO overlap')
    root = np.einsum('ik,k,jk->ij', v, np.sqrt(w), v)
    meta = orth.orth_ao(mol, method='meta_lowdin', pre_orth_ao='ANO', s=S)
    gram = np.einsum('pi,pq,qj->ij', meta, S, meta)
    if not np.allclose(gram, np.eye(len(S)), atol=1e-8, rtol=0.):
        raise RuntimeError('Meta-Lowdin AO orthogonalization failed')
    meta_transform = np.einsum('pi,pq->iq', meta, S)
    for key, label, transform in [('lowdin', 'Lowdin', root),
                                   ('meta_lowdin', 'meta-Lowdin', meta_transform)]:
        # diag(T D T.T), no SCGVB reoptimization or changes to the density.
        pop = np.einsum('ip,pq,iq->i', transform, density, transform)
        spin_pop = np.einsum('ip,pq,iq->i', transform, spin_density, transform)
        methods[key] = _atomic_populations(mol, pop, spin_pop, label)
    methods['lowdin']['orthogonalization'] = 'symmetric AO overlap; no pre-orthogonalization'
    methods['meta_lowdin']['pre_orthogonalization'] = 'ANO'
    return dict(methods['mulliken'], methods=methods,
                bond_orders=bond_order_analysis(mol, density, spin_density, S, root))


def print_populations(analysis):
    methods = analysis.get('methods', {'mulliken': analysis})
    for quantity, title, total in [('charge', 'Atomic charges / e', 'total_charge'),
                                   ('spin_population', 'Atomic spin populations (N_alpha - N_beta)', 'total_spin_population')]:
        print(title)
        print(' Atom ' + ''.join(f"{data['method']:>16s}" for data in methods.values()))
        for i, atom in enumerate(next(iter(methods.values()))['atoms']):
            print(f"{atom['index']:3d} {atom['symbol']:>2s}" + ''.join(
                f"{data['atoms'][i][quantity]:16.7f}" for data in methods.values()))
        print(' Total' + ''.join(f"{data[total]:16.7f}" for data in methods.values()))

    if 'bond_orders' in analysis:
        bonds = analysis['bond_orders']
        threshold = bonds['print_threshold']
        print(f'Bond orders (dimensionless; either |index| >= {threshold:g}; all pairs in JSON)')
        print(' Atom pair              Mayer   Wiberg-Lowdin')
        for pair in bonds['pairs']:
            if max(abs(pair['mayer']), abs(pair['wiberg_lowdin'])) >= threshold:
                label = f"{pair['atom1']} {pair['symbol1']} - {pair['atom2']} {pair['symbol2']}"
                print(f" {label:<20s}{pair['mayer']:10.6f}{pair['wiberg_lowdin']:16.6f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze saved SCGVB results without reoptimizing')
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from calc_molecule import build_molecule
    result = json.loads(args.result.read_text())
    m = result['molecule']
    mol = build_molecule(m['atoms'], m['basis'], m['charge'], m['multiplicity'], m.get('coordinate_unit', 'Angstrom'))
    analysis = population_analysis(mol, np.asarray(result['orbital_coefficients']),
                                np.asarray(result['spin_coefficients']),
                                np.asarray(result['inactive_orbital_coefficients']) if 'inactive_orbital_coefficients' in result else None,
                                determinant_size_profile=result.get('calculation', {}).get('determinant_size_profile', 'standard'))
    output = args.output or args.result.with_name(args.result.stem + '_population.json')
    if output.resolve() == args.result.resolve():
        parser.error('Population output must differ from the source result')
    output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    print_populations(analysis)
    print(f'Result: {output.resolve()}')


if __name__ == '__main__':
    main()
