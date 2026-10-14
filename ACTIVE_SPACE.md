# SCGVB with closed-shell orbitals and active spaces

Full SCGVB remains the default. Only explicitly selected options freeze doubly occupied inactive orbitals and treat the remaining electrons with SCGVB.
Adding `--optimize-core` also optimizes inactive orbitals while preserving double occupation.
Orbital tracking across geometry scans and automatic accuracy guarantees are not provided.

## Optimizing core orbitals with double occupation preserved

```bash
python calc_molecule.py --atom 'Li 0 0 0; H 0 0 1.6' --basis cc-pvdz \
  --frozen-core 1 --optimize-core --initial-guess hf \
  --backend jax --device cpu --energy-evaluation determinant \
  --output lih_relaxed_core.json
```

`--frozen-core` / `--inactive-mos` select the initial closed-shell orbitals.
With `--optimize-core`, all selected closed-shell orbitals are relaxed; otherwise they remain frozen.
An empty inactive space is an error for core optimization.
`--active-space pi --optimize-core` is also supported, relaxing all inactive closed shells, including 1s and sigma orbitals.
Active-space dimension and electron count remain unchanged, but the active space also rotates, so this is not strictly constrained to the initial projected pi subspace.

`scgvb/optimization/core.py` uses PySCF's two-step MCSCF orbital optimizer.
The inner solve optimizes only nonorthogonal SCGVB orbitals and Cs, then constructs one- and two-particle density matrices from their determinant expansion.
It neither optimizes independent FCI coefficients nor solves an FCI eigenproblem.
The outer solve rotates between closed-shell, active, and virtual spaces, preserving core double occupation and core–active orthogonality.
Active SCGVB orbitals are not constrained to be mutually orthogonal.

`--core-max-iterations` limits outer iterations (default 50).
`--core-gradient-tolerance` sets the 2-norm tolerance of the energy gradient with respect to orbital rotations (default 1e-6).
Inner physical gradients and normalization, and outer convergence, are all checked; failure is not returned as a converged result.
JSON `core_optimization` records the initial frozen-core energy, iteration history, final orbital gradient, initial core coefficients, and double occupations.
The usual orbital-coefficient fields contain optimized coefficients.

NumPy/JAX and both energy evaluators are supported.
The requested JAX device is used for inner SCGVB evaluation; outer integral transformations, density matrices, and orbital rotations currently run on the CPU. JAX evaluators must also be rebuilt each iteration.
The active determinant-space size limit applies even when permutation evaluation is selected.
Performance and convergence of orbital relaxation for 12-pi-electron systems require separate evaluation; frozen-core timings do not carry over automatically.

## Determinant-size profiles: 14-electron support

The default `standard` profile retains at most six electrons of each spin and two million elements per major array.
Explicit `large14` selection allows up to 14 electrons and 14 orbitals, at most seven electrons of each spin, and 12 million elements per major array.
This is a bounded extension, not an unlimited override or a guarantee of GPU memory availability or convergence.
A 14-electron/14-orbital singlet has 11,778,624 determinants and 429 spin functions.

```bash
python calc_molecule.py --input phenazine.toml --determinant-size-profile large14
# Check the selected size profile only
python calc_molecule.py --input phenazine.toml --determinant-size-profile large14 --check
```

In TOML, specify `determinant_size_profile = "large14"` under `[calculation]`.
In Python, use `calculate(..., determinant_size_profile='large14')` or `SCWF_Energy(..., determinant_size_profile='large14')`, for either NumPy or JAX.
Energy evaluation is configured separately; use `energy_evaluation='determinant'` for 14 electrons.
This option does not lift the permutation-sum limit.

The selected profile propagates into core-relaxation inner solves, final evaluation, and density analysis, and is saved as `calculation.determinant_size_profile` in result JSON.
Specify the same profile in the input or CLI when restarting; reading restart JSON does not automatically raise the default limits.
Density reanalysis uses the saved profile; older JSON is treated as `standard`.

On one rikyu GB200, independent 14e/14o energy/gradient checks and phenazine convergence with both frozen inactive orbitals and all inactive orbitals relaxed were verified.
Polyene energy/gradient evaluation took about 1.48 seconds with approximately 3.22 GiB peak actual GPU usage.
Recheck performance and memory for other spins, molecules, bases, or optimization methods. These measurements concern 14-electron singlets.

## Examples

Run from the project root. PySCF, NumPy, and SciPy are required; JAX is also needed with `--backend jax`.
JAX is recommended for optimization with eight active electrons.

```bash
# CH4: freeze one C 1s orbital (two electrons), leaving eight active electrons
python calc_molecule.py --xyz examples/ch4.xyz --frozen-core 1 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output ch4_active.json

# Benzene: neutral singlet in the xy plane, six pi electrons
python calc_molecule.py --xyz examples/benzene.xyz --active-space benzene-pi \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output benzene_pi.json

# Check selection and size before optimization (HF runs for active-space selection)
python calc_molecule.py --xyz examples/benzene.xyz --active-space benzene-pi \
  --energy-evaluation determinant --check

# Small active spaces can also use NumPy
python calc_molecule.py --atom 'Li 0 0 0; H 0 0 1.6' --frozen-core 1 \
  --energy-evaluation determinant --initial-guess hf --output lih_active.json

# Existing full SCGVB: default behavior is unchanged
python calc_molecule.py --xyz examples/h4.xyz --output h4_full.json
```

## General planar pi systems: eight pi electrons in COT

`--active-space pi` selects an active space using carbon 2p targets in planar hydrocarbons, not only benzene.
It derives the molecular-plane normal from coordinates and combines each carbon's MINAO 2px/2py/2pz along that normal, so the molecule need not be aligned with the xy plane.
AVAS-type projections in both occupied and virtual subspaces are diagonalized; eigenvalues at least 0.2 are retained.
All singly occupied ROHF orbitals must belong to the pi space.

```bash
python calc_molecule.py --xyz examples/COT.xyz \
  --active-space pi --pi-electrons 8 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output cot_pi.json
```

Add `--check` to the same command to check without optimization.
In Python, use `calculate(..., active_space='pi', pi_electrons=8, ...)`.
`--pi-electrons 8` checks that eight electrons were selected; if omitted, the count follows from projection.
It does not force the electron count and stops on mismatch.

For `examples/COT.xyz`, eight electrons in eight orbitals are active out of 56 electrons; the remaining 48 electrons occupy 24 frozen closed-shell orbitals. There are 14 singlet spin functions.
STO-3G/JAX CPU/seed=40/HF initialization converged to `-303.880757161246 Eh`, with maximum absolute gradient 1.28e-8 and norm error below 8.89e-16.
CASCI with the same frozen closed shells gave `-303.884874619423 Eh`; HF gave `-303.674298594311 Eh`.
Results and independent references are stored locally as `examples/active_space/cot_pi_*.json`.
This is a single-point calculation at the supplied planar geometry, not a geometry optimization or a guarantee of a global minimum.

Automatic global `pi` selection currently supports only non-linear C/H systems with all atoms within 1e-4 Å of the molecular plane.
It constructs one pi target per carbon and checks that the selected orbital count equals the carbon count.
Planarity and projection do not fully establish chemical suitability. For systems mixing saturated carbons, inspect orbitals and specify a manual space rather than accepting automatic selection directly.
Use the local mode below for nonplanar sp2 carbon systems. Heteroatom pi contributions and selection of only some pi atoms are not supported.
`pi` cannot be combined with manual MO selection or `--frozen-core`.
For compatibility, `benzene-pi` retains its existing restriction to C6H6 in the xy plane.

## Nonplanar pi systems: biphenyl

With `--pi-normal-mode local`, the three sigma-bond directions around each carbon define a local-plane normal. MINAO 2px/2py/2pz are combined along it to form a pi target.
The molecule need not be globally planar. AVAS-type projection, effective frozen-core integrals, and SCGVB evaluators are shared.
The two aromatic rings are solved together as a 12-electron problem including inter-ring interactions, rather than independently.
For nonplanar systems, pi/sigma separation is generally an approximation defined by the selected model space, not exact symmetry separation.

```bash
python calc_molecule.py --xyz examples/biphenyl_45deg.xyz \
  --active-space pi --pi-normal-mode local --pi-electrons 12 \
  --energy-evaluation determinant --backend jax --device cpu \
  --initial-guess hf --output biphenyl_result.json
```

Use `--device gpu` on GPU systems; the Python setting is `pi_normal_mode='local'`.
The default `global` mode retains a single molecular plane and rejects nonplanar geometries.

The included biphenyl is a validation geometry: two regular hexagons with C–C=1.397 Å, joined by a C–C=1.49 Å bond and twisted by 45°. It is not geometry-optimized.
With STO-3G, selection of 12 electrons in 12 orbitals and 35 frozen closed-shell orbitals was verified, as was energy and directional-derivative agreement with independent NumPy/PySCF evaluation.
SCGVB optimization convergence for this example has not yet been verified.

Local-mode bond detection uses geometric cutoffs C–C <= 1.85 Å and C–H <= 1.30 Å.
Each carbon must have exactly three neighbors; sigma-bond tilts greater than 10° from the local plane are rejected.
sp3 carbons, strongly pyramidal geometries, and structures near bond cleavage are not automatically treated as pi systems.
Bond partners, local normals, out-of-plane tilts, and distance thresholds are saved in result JSON.
Consider manual MO selection when distance-based detection is unsuitable, for example in crowded geometries.

## Selection options and constraints

- `--frozen-core N`: freeze the N lowest-energy HF spatial orbitals with double occupation. N counts orbitals, not electrons; double occupation is checked.
- `--inactive-mos 1 2 ...`: specify frozen HF orbitals with one-based CLI indices. Do not combine with `--frozen-core`.
- `--active-mos 2 3 4 ...`: specify the HF subspace for expanding active SCGVB orbitals. Include antibonding/virtual orbitals needed to describe bonding changes, not just occupied orbitals. If omitted, the full orthogonal complement of the inactive space is used.
- Every occupied HF orbital must be included in either the inactive or active space. Duplicates, out-of-range indices, active/inactive overlaps, and fewer than two active electrons are errors.
- `--active-space benzene-pi`: select six electrons in six orbitals with PySCF AVAS `C 2pz` and threshold 0.2. This requires neutral singlet C6H6 in the xy plane. Failure to obtain 6e/6o stops execution. Six core and 12 occupied sigma orbitals are frozen; orbitals cannot leave the active pi space. Do not combine with manual-selection options.

MO indices depend on the HF result and need not identify the same orbitals across degeneracies or geometry changes.
Manual selection currently refers to canonical HF orbitals, without localization or tracking.
Initial guesses `hf` / `core` / `random` are generated only within the active space.

## Shared engine

`scgvb/orbitals/active_space.py` constructs frozen orbitals `inactive` and an S-orthonormal `basis`.
Optimization variables X and output AO coefficients C satisfy `C = basis @ X`.
This always preserves orthogonality to inactive orbitals while retaining mutual nonorthogonality of active orbitals.
Full-space calculations still optimize AO coefficients directly.

For closed-shell density D, PySCF's J/K routines compute `V = J[D] - K[D]/2`.

- Effective one-electron integrals: `basis.T @ (h + V) @ basis`
- Two-electron integrals: ordinary electron-repulsion integrals transformed into the active basis
- Closed-shell constant: `Tr[D h] + Tr[D V]/2`
- Total energy: `E_nuc + E_inactive + numerator / norm`

Spin functions, orbital initialization, normalization, augmented Lagrangian optimization, and Rayleigh-quotient gradients/refinement are shared with full-space calculations.
`scgvb/energy/numpy_backend.py` and `scgvb/energy/jax_backend.py` handle both problems.
`calcEnergy` returns the active numerator and norm. The closed-shell constant is excluded from the optimized numerator and added to the final total energy.

`--energy-evaluation determinant` generates no electron-permutation tables.
The existing JAX exterior-product evaluator was extended and a NumPy implementation added.
Nonorthogonal SCGVB orbitals are expanded in determinants on an orthogonal basis, then the Hamiltonian is applied.
Determinant coefficients follow from orbitals and spin coefficients; they are not independent optimization variables.
NumPy uses PySCF's FCI Hamiltonian-contraction routines without solving an FCI eigenproblem.
This differs from a generalized Slater–Condon evaluator for pairs of nonorthogonal determinants.

Cost also depends on the basis determinant space, so scaling is bounded.
With the default `standard` profile, NumPy and JAX allow at most six electrons of each spin and cap determinant-coefficient arrays, minor arrays of each order, primitive spin-coefficient arrays, and transformed two-electron integral arrays at two million elements each.
This supports a 12-electron/12-orbital singlet with 853,776 determinants and 132 spin functions.
The JAX backend uses polynomial minor recursion, checkpointing, a sum of squared one-body operators for Hamiltonian action, and a custom VJP to avoid orbital-count²×determinant-count intermediates and their reverse-mode records.
See [HPC_README.md](HPC_README.md) for details and portable execution examples.
Excess sizes stop execution before integrals or spin expansions are generated, although HF space selection runs earlier.

STO-3G CH4 has n=8, eight active electrons, 14 singlet spin functions, and 4,900 basis determinants.
The benzene pi system has n=6, six electrons, five spin functions, and 400 basis determinants.
These counts do not mean the same variational space as CASSCF.
The old `permutation` evaluator remains for comparison and can evaluate active spaces within its size limit.
`--max-permutation-pairs` applies only to that evaluator.

## Python API

```python
from calc_molecule import calculate

result = calculate('Li 0 0 0; H 0 0 1.6',
                   inactive_mos=[0],       # Python indices are zero-based
                   active_mos=[1, 2, 3, 4, 5],
                   energy_evaluation='determinant', initial_guess='hf')
```

In the low-level API, pass the result of `prepare_active_space(mol, ...)` to `SCWF_Energy(mol, compute_fci=False, active_space=space, energy_evaluation='determinant')`.
`space=None` selects the full-space calculation.
Active-space calculations require `compute_fci=False`: full FCI is not the same variational problem as the frozen-closed-shell approximation, so it is not run implicitly as a reference.

## Output and orbital display

Fields have been added to the existing schema version 1.
`orbital_coefficients` contains active SCGVB AO coefficients (AO count × active-electron count); `inactive_orbital_coefficients` stores one column per frozen closed-shell orbital.
`optimization_basis_coefficients` and `optimization_orbital_coefficients` also save the optimization subspace and variables.
`active_space` records selection details and HF references.

`electronic_numerator_hartree` is the active effective-Hamiltonian numerator, `inactive_energy_hartree` is the closed-shell constant, and `nuclear_energy_hartree` contains only nuclear repulsion.
`wavefunction_norm` and gradients follow the active-SCGVB convention excluding frozen closed shells.

```bash
python generate_orbitals.py benzene_pi.json --output-dir benzene_orbitals
```

Active orbitals are exported first, followed by one orbital per inactive column.
HTML and the manifest distinguish inactive doubly occupied orbitals.
Older full-space result files remain readable.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Tests check energy, norm, and complex-step gradients against the old permutation evaluator; agreement with JAX automatic differentiation; gradients at singular subminors; agreement with full SCGVB including frozen closed shells; and effective one-electron integrals/closed-shell constants against PySCF CASCI.
MO-selection validation, size limits, and the path from optimization to JSON/Cube output are also tested.

Measurements using the included geometries, STO-3G, HF initial guesses, seed=40, and JAX CPU:

| System | HF / Eh | Frozen-closed-shell SCGVB / Eh | CASCI with the same frozen closed shells / Eh | Maximum absolute gradient |
|---|---:|---:|---:|---:|
| CH4, eight active electrons | -39.726724240941 | -39.794445033744 | -39.805558524322 | 1.04e-7 |
| C6H6, six pi electrons | -227.890600566936 | -227.989954186596 | -227.997024016905 | 5.49e-9 |

Norm errors were at most 6e-16 in both cases. SCGVB energies were verified to lie below HF and above CASCI in the same space.
CASCI was run separately for validation and is not part of normal calculations.
These are local optimization results, not guarantees of global minima or convergence for other geometries or bases.
Results and references are stored locally in `examples/active_space/`.

## Explicit atomic-orbital targets

`active_space='ao'` supports MINAO target rows such as `orbitals=['1 2px 2py A',
'1 1s F']`. Atom indices are one-based; A selects active orbitals, I selects
relaxable doubly occupied orbitals, and F selects permanently frozen doubly
occupied orbitals. F columns lead the inactive basis and are excluded from
PySCF orbital rotations even with core optimization enabled. See INPUT_FILE.md
for selection defaults, projection thresholds, validation, and an executable
`examples/inputs/allyl_ao.toml` input.
