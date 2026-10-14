# Nonorthogonal SCWF single-point calculations for polyatomic molecules

By default, `calc_molecule.py` accepts molecular coordinates and optimizes orbital and spin coefficients at a single geometry, without a distance scan or nuclear-coordinate optimization.
Geometry optimization is available explicitly through `[geometry]` or `--geometry-method`; see [GEOMETRY_OPTIMIZATION.md](GEOMETRY_OPTIMIZATION.md).
Full CI is not run. SCF is used for `--initial-guess hf` initial orbitals or frozen-closed-shell/active-space selection.
Both NumPy and JAX use the existing augmented Lagrangian method and stagnation fallback.

## XYZ input

```bash
# H3+: three atoms, two electrons, singlet, JAX CPU
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --multiplicity 1 \
    --basis sto-3g --backend jax --device cpu --output h3_plus.json

# H4: four atoms, four electrons, singlet, original NumPy backend
python calc_molecule.py --xyz examples/h4.xyz --output h4.json
```

XYZ input is a single frame containing an atom count, a comment line, and atom lines of the form `element x y z`.
Charge and multiplicity are specified explicitly in the command, not read from the XYZ comment.
Coordinates default to Angstrom; use `--unit Bohr` for Bohr coordinates.

## Specifying coordinates directly

```bash
python calc_molecule.py --atom 'H 0 0 0; H 0.9 0 0; H 0.45 0.7794228634 0' \
    --charge 1 --multiplicity 1 --output h3_plus.json
```

`--multiplicity` means **2S+1**, one greater than PySCF's `mol.spin=2S`.
Singlet=1, doublet=2, triplet=3. Consistency with the electron count is checked.
Defaults are charge 0, multiplicity 1, sto-3g, NumPy, and random seed 40.
Select JAX with `--backend jax`, and explicitly select a GPU on NVIDIA systems with `--device gpu`.
`--device` is available only with JAX.

## Checking calculation size

```bash
python calc_molecule.py --xyz examples/h4.xyz --check
```

For full-space calculations, `--check` reports electron count, AO count, spin-function count, configuration-pair count, determinant count, and storage bytes for dense spin matrices alone, without generating integrals or optimizing.
With an active space, HF runs for selection but SCGVB optimization does not run.

The default `permutation` evaluator uses `(N!)²` configuration pairs and stops above 100,000 pairs.
`--max-permutation-pairs` can change the limit but does not improve computational scaling.
`--energy-evaluation determinant` avoids permutation tables and uses a separate determinant-space size limit.
Frozen closed-shell orbitals and active spaces are also supported. See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for CH4/benzene examples and constraints.
At least two active electrons are required.

## Output

The `--output` JSON file (default `scwf_result.json`) stores:

- Input geometry, standardized to Angstrom in output, charge, multiplicity, basis, and calculation settings.
- Total energy, nuclear repulsion energy, and electronic numerator `Ee`, in Hartree.
- Wavefunction norm `Delta`, normalization error, and the maximum absolute gradient of `Ee/Delta`.
- `orbital_coefficients`: coefficient matrix C, with AO rows and nonorthogonal-orbital columns.
- `spin_coefficients`: spin-basis coefficients Cs.
- `ao_labels`, AO overlap S, orbital overlap `C.T @ S @ C`, and spin branch diagrams.
- Warnings during optimization, including warnings when a weaker stationarity criterion was used.

Mutual orthogonality of active orbitals is not imposed. Active and inactive orbitals are orthogonal when inactive orbitals are specified.
Squared spin coefficients are not interpreted as general state probabilities.
On single-point optimization failure, the CLI exits with code 1 and writes no new result. Successful calculations overwrite existing result files.
Local solutions and convergence paths depend on the starting point; change it with `--seed`.
Set iteration limits and normalization tolerance with `--max-outer-iterations`, `--max-inner-iterations`, and `--epsilon`.

`--verbose` shows AL internal constraint errors as well as `constraint` and `physical_gradient` (the maximum absolute gradient of Ee/Delta) for normalized candidates.
A candidate improved by auxiliary Rayleigh-quotient optimization and satisfying normalization is carried into the next iteration even if its gradient has not converged.
The penalty coefficient increases only when constraint-error reduction is insufficient.
If there is no improved candidate and constraint error fails to decrease by at least 5% for five consecutive iterations, or if the coefficient reaches 1e8, optimization stops with diagnostics.
This stop does not indicate convergence and cannot be bypassed simply by increasing the iteration limit.
Small normalization error with a large physical gradient does not establish energy convergence; relaxing tolerances alone cannot guarantee it.

## Python API

```python
from calc_molecule import calculate, read_xyz

result = calculate(read_xyz('examples/h3_plus.xyz'), charge=1, multiplicity=1,
                   backend='jax', device='cpu')
print(result['total_energy_hartree'])
print(result['orbital_overlap'])
```

When using the low-level class directly, specify `SCWF_Energy(mol, compute_fci=False)`. The JAX class also accepts `device='cpu'`.
The low-level default `compute_fci=True` is retained only for compatibility with the existing diatomic comparison script and independent validation tests.
The general molecular interface always sets it to False and provides no FCI execution option.

## Execution checks

On a Mac CPU with STO-3G, triangular H3+ gave approximately `-1.255812305378 Eh`, and linear H4 with 1 Å spacing gave approximately `-2.164905132939 Eh` with JAX.
H3+ agreement with NumPy was also checked. These results apply to the specified geometries and starting points.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_molecule_interface -v
```

## Gradient stagnation in Li2 and related systems

`unit_orbital_norm` is the wavefunction norm when each nonorthogonal orbital and the spin coefficients are normalized individually.
It diagnoses cancellations hidden by overall scaling; it is not a convergence tolerance.
Very small values can indicate a numerically unstable coefficient representation.
`normalized_constraint` alone cannot establish convergence.

Both NumPy and JAX accept `--energy-evaluation determinant`.
It evaluates the same nonorthogonal wavefunction through a determinant expansion without Full CI optimization.
This experimental alternative has bounded determinant-space support and does not guarantee Li2 convergence.
The existing `permutation` evaluator remains the default. See `JAX_BACKEND.md` for details.

## Physical initial orbitals

`--initial-guess hf` uses occupied RHF orbitals for closed shells and ROHF orbitals for open shells.
SCF nonconvergence stops execution; unconverged orbitals are not silently accepted.
`--initial-guess core` solves the generalized one-electron Hamiltonian eigenproblem and occupies the lowest levels, without an SCF treatment of electron repulsion.
The original random initialization is `--initial-guess random`, retained as the default for compatibility.

Doubly occupied MOs are assigned to consecutive orbital pairs, followed by singly occupied MOs.
Cs retains equal initial coefficients. For HF/core starts, a small random perturbation with S-metric norm 0.02 is added to each orbital, followed by individual normalization, to break the symmetry of identical paired orbitals.
`--seed` makes this reproducible. `--initial-perturbation 0` disables perturbations, but exactly identical pairs may have difficulty moving along directions that separate the pair.
MO occupations and mutual orbital orthogonality are not constrained during optimization.

```bash
python calc_molecule.py --xyz examples/li2.xyz --backend jax --device cpu \
  --max-permutation-pairs 600000 --initial-guess hf --verbose
```

In Python, use `calculate(..., initial_guess='hf', initial_perturbation=0.02)`.
To generate only orbitals, import `initial_orbitals` from `scgvb.orbitals.initial` and call `C, metadata = initial_orbitals(mol, method='hf')`.
Initialization settings are saved under `calculation.initialization` in result JSON.
HF provides a potentially better starting point; it does not guarantee SCWF convergence or a global minimum.

## Mulliken, Löwdin, and meta-Löwdin charges and spin populations

After calculation, a comparison table shows Mulliken, Löwdin, and meta-Löwdin charges and spin populations for every atom. No additional input settings are needed. Result JSON stores them in:

- `population_analysis.methods.mulliken`
- `population_analysis.methods.lowdin`
- `population_analysis.methods.meta_lowdin`

For compatibility, existing values directly under `population_analysis` remain Mulliken values.
Within each method's `atoms`, `index` is one-based, `charge` is atomic charge in e, `electron_population` is the assigned electron count, and `spin_population` is the alpha-minus-beta electron count.

The one-particle density matrix is constructed from the normalized SCGVB wavefunction. Each doubly occupied inactive orbital contributes one alpha and one beta electron; core-relaxed calculations use the final core coefficients.
This is neither a simple sum of squared orbital coefficients nor charges from a separate HF calculation.
For AO density D and overlap S, `diag(D S)` is summed per atom and subtracted from nuclear charge.
Electron and spin conservation are checked. Atomic charges sum to molecular charge, and spin populations sum to `2 M_S = multiplicity - 1`.

To analyze an existing result without reoptimizing:

```bash
python -m scgvb.analysis.population allyl_active.json --output allyl_population.json
```

If output is omitted, `_population.json` is appended to the original stem, preserving the calculation result.
Analysis expands determinants in the space spanned by SCGVB orbitals, avoiding a large full-AO CI expansion.
It does not truncate orbital directions and supports linearly dependent orbitals. Analysis with 12 active electrons is also tested.
Mulliken charges and spin populations depend on the basis and partitioning scheme; they are not atomic oxidation states.

Löwdin analysis uses symmetric orthogonalization without atomic pre-orthogonalization and sums `diag(S^(1/2) D S^(1/2))` per atom.
Meta-Löwdin uses PySCF's `orth_ao` with explicit `ANO` pre-orthogonalization. Core, valence, and remaining AO groups are orthogonalized separately; `T = X.T @ S` is used to sum `diag(T D T.T)`.
These AO groups are distinct from the SCGVB active/inactive orbital selection.
All three methods use the same final SCGVB density and do not change optimization or energy.
Charge and spin totals are checked for every method; basis dependence remains.

### Older charge-analysis output

With `analysis_version = 2`, the unused triangle of the Cholesky factor is explicitly zeroed and reconstruction of the overlap matrix and SCGVB orbitals is checked.
On some environments, the previous version mixed this unused triangle into the density transformation, giving incorrect atomic distributions despite correct electron and spin totals.
Mulliken, Löwdin, and meta-Löwdin were all affected.
Saved wavefunction coefficients and optimized energies are unaffected. Reprocess affected older output with `scgvb.analysis.population`.
The correction reconstructs density properly from the original SCGVB wavefunction; it does not average charges over symmetric atoms.
