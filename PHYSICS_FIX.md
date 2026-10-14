# Corrections to the energy formulas identified while investigating LiH divergence

Numerical agreement with the old implementation did not guarantee physical correctness.
The following changes correct energy evaluation formulas rather than relaxing optimizer tolerances.
Previously reported energies for H₂, HeH⁺, and other systems therefore also change.

## Corrections

1. **Overlap between left and right spin permutations**
   For signed permutation representations U, the pair matrix is `U(left).T @ U(right)`.
   The old composition omitted the inverse permutation on the left.
   Identical left and right permutations must give the identity, and the complete set of pair matrices must satisfy Hermiticity.
   Spin-function coefficients and basis ordering are unchanged.

2. **Two-electron integral indices**
   PySCF spatial AO integrals use chemists' notation `(ij|kl)`.
   With bra orbitals `l,n` and ket orbitals `r,t`, the required integral is `(lr|nt)`.
   This corrects the old `(ln|rt)` contraction.

3. **Spatial AO integral retrieval**
   Integrals are requested explicitly, for example with `mol.intor('int1e_ovlp')`.
   Taking every other spinor integral and extracting its real part did not match the spatial AO basis of Li, which includes p orbitals.
   Overlap, one-electron, and two-electron integrals now use the same AO basis.

Permutation sums in the energy are evaluated with array operations, and duplicate SCF runs during integral retrieval have been removed.
The augmented Lagrangian method, complex-step gradients, and normalization tolerance `1e-8` are retained.

## Scale freedom in optimization

Each orbital occurs once in every term, so dividing orbital columns by nonzero constants changes only the overall wavefunction scale.
First equalizing column magnitudes and then multiplying every column by `Delta**(-1/(2*Ne))` normalizes the same physical state.
Tests verify that the energy ratio `Ee/Delta` is invariant under this operation.

This rescaling is applied to the inner solution. The result is accepted when normalization error is below `1e-8` and the infinity norm of the `Ee/Delta` gradient is at most `1e-6`.
Normalization alone does not count as convergence without checking physical stationarity.
This avoids repeatedly increasing the penalty coefficient solely to correct the overall amplitude.

If inner BFGS stops because of precision loss or the iteration limit while a physical gradient remains, the state from before the current outer iteration is normalized and improved by optimizing `Ee/Delta`.
Orbitals from the failed inner calculation are not used as the fallback starting point because their linear dependence may have increased.
The fallback checks both normalization error `1e-8` and the physical-gradient limit `1e-6`.
This addresses the stagnation reproduced for LiH at 0.6 Å, which increasing the penalty coefficient alone did not resolve.

## Independent validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
    tests.test_physical_energy tests.test_spin_function tests.test_energy_regression tests.test_optimizer -v
```

`test_physical_energy.py` constructs CI coefficients in an orthogonal AO basis from nonorthogonal orbitals using determinant minors.
It evaluates the same wavefunction expectation with PySCF's FCI Hamiltonian action, without reusing the corrected permutation pairs or integral contractions.
Energy and norm agreement is checked for H₂ singlets and triplets, HeH⁺, Li doublets, and LiH using several random orbital and spin-coefficient sets.
The energy is also checked against the FCI lower bound for the same molecule, basis, and spin.

An optimization regression test was added for LiH / STO-3G / 10 Å / seed 40.
LiH at 0.6 Å, using the third random initial guess of the distance scan, is also tested for optimization, normalization, physical gradients, and energy agreement with an independent determinant expansion.
This local optimization does not guarantee the global optimum from every starting point.
The distance scan in `calc_diatomic_mol.py` initializes randomly only at the first distance; subsequent points start from copies of the previously converged orbitals and spin coefficients.
It assumes a continuous scan with the same molecule, basis, and AO ordering.
Restarting from positive random orbitals at every distance caused stagnation with strong linear dependence, especially for LiH at 1.1 Å.
The dissociation-limit calculation retains its independent initial guess.
This continuation method also does not guarantee a global minimum; comparing multiple states requires calculations with other starting points.
The existing AL termination criterion is retained. If that termination path is used after inner-BFGS precision loss, normalization and the existing gradient limit `1e-4` are checked, and a warning is issued.

## Execution checks (2026-09-17)

- All 25 regression tests passed.
- The current LiH / STO-3G / seed 40 script completed the 10 Å dissociation limit and all 21 points from 0.5 to 2.5 Å in 0.1 Å increments. PDF and EPS output was also checked.
- The largest normalization error across all 22 calculations was approximately `2.28e-13`.
- The 10 Å energy was `-7.782107830840841 Eh`, compared with FCI at `-7.782418402409652 Eh`. The energy difference from the independent determinant expansion was approximately `2.73e-13 Eh`.
- Validation used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg`.
