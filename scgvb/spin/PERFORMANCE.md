# Spin-function performance improvements (2026-09-17)

> This records the original performance work. Subsequent LiH validation identified a missing inverse permutation connecting the left and right permutations; pair matrices were corrected to `U(left).T @ U(right)`.
> Agreement with all pair matrices and energies of the old implementation is no longer used as a correctness criterion.
> Validation combines legacy data for individual permutation matrices and spin coefficients with an independent determinant expansion.
> See `PHYSICS_FIX.md` in the repository root for details.

## Changes

- Generate primitive spin configurations with `itertools.combinations`, preserving legacy configuration order.
- Replace row-by-row branch-diagram deletion with masked extraction.
- Vectorize coupling coefficients over spin configurations and basis functions, preserving the electron multiplication order and sign convention.
- Look up branch diagrams after adjacent transpositions in a dictionary instead of comparing against the entire basis sequentially.
- Remove numerical matrix inversion because each adjacent transposition is its own inverse.
- Reduce permutation decomposition and matrix-representation calculations from `(N!)²` configuration pairs to `N!` distinct permutations.
- Avoid generating unused primitive spin-coefficient matrices when constructing matrix representations.
- Cache up to four internal calculations by electron count and multiplicity. Returned matrices are copied separately so that caller mutations do not affect other matrices or later results.

Public function names, return structures, the `np.matrix` type, basis and electron-configuration ordering, the `Dr * Dl` composition order, and sign conventions are retained.

## Validation

```bash
python -m unittest tests.test_spin_function tests.test_energy_regression -v
```

All 11 tests passed. Legacy reference data is stored in `tests/data/spin_legacy_reference.npz`.

- Compared coefficients and adjacent-transposition matrices with the old implementation for every allowed spin with 2–8 electrons.
- Compared all permutation representations for every allowed spin with 2–4 electrons and for the five-electron doublet.
- Basis and configuration ordering matched exactly; coefficients and matrices agreed with `atol=rtol=1e-13`.
- Verified eigenvalues S(S+1) using an independently constructed S² operator in the primitive spin basis.
- Checked orthonormal spin coefficients, squares of adjacent-transposition matrices, and the braid relation.
- Compared energies and normalization factors using old and new permutation representations for multiple inputs.
- Ran the dissociation limit and all 24 distance points in `calc_diatomic_mol.py`, plus the triplet in `test.py`, with both implementations. The largest absolute energy difference in this environment was zero.

These checks verify preservation of values across this change; they do not establish the validity of the entire existing energy formula.

## Measured timings

All measurements used the same environment: Python 3.12 / NumPy 2.1.1, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.
Each value is the shortest of three runs. Caches were cleared before each cold-call comparison.

| Operation | Old implementation | New implementation | Speedup |
|---|---:|---:|---:|
| All permutation representations, four-electron singlet, cold call | 27.78 ms | 2.24 ms | About 12× |
| All permutation representations, five-electron doublet, cold call | 1037.21 ms | 35.67 ms | About 29× |
| All permutation representations, five-electron doublet, repeated call | 996.44 ms | 28.35 ms | About 35× |
| Primitive spin functions and coefficients, eight-electron singlet | 10.26 ms | 0.16 ms | About 64× |
| Adjacent-transposition matrices, eight-electron singlet | 1.47 ms | 0.11 ms | About 14× |

These are not speedups for the entire program. The public API still returns matrices for all configuration pairs, so output count and storage remain proportional to `(N!)²`.
Removing that limitation requires an interface change on the energy-evaluation side as well.
