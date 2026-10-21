# Calculation input files

`calc_molecule.py` supports TOML input. Comments start with `#`, booleans are `true` / `false`, and strings are quoted.
Existing command-line options remain available.

Initial guesses and optimization are described in:

- [Simultaneous and alternating optimization](#simultaneous-and-alternating-optimization): BFGS, wavefunction metrics, single-orbital solves, spin-update intervals, and norm protection.
- [Independent spin initialization and joint-correction start](#independent-spin-initialization-and-joint-correction-start): initial spin solving, restart behavior, and correction start indices.
- [Experimental joint corrections on HPC](HPC_README.md#comparing-joint-corrections-from-a-saved-state): diagnostic APIs for residual-direction additions and spin-response corrections.

```bash
python calc_molecule.py examples/inputs/allyl.toml
# Equivalent command
python calc_molecule.py --input examples/inputs/allyl.toml
# Check only the active space and calculation size (HF selection still runs)
python calc_molecule.py examples/inputs/allyl.toml --check
# Keep the same input while changing device and output destination
python calc_molecule.py examples/inputs/allyl.toml --device gpu --output allyl_gpu.json
```

## Allyl-radical example

```toml
[molecule]
xyz = "../allyl.xyz"
basis = "cc-pvdz"
charge = 0
multiplicity = 2

[calculation]
backend = "jax"
device = "cpu"
energy_evaluation = "determinant"
initial_guess = "hf"

[active_space]
mode = "pi"
pi_electrons = 3
optimize_core = true

[output]
path = "allyl_active.json"
```

This example is `examples/inputs/allyl.toml`.
See `lih.toml` for LiH core relaxation and `biphenyl.toml` for a nonplanar 12-pi-electron biphenyl calculation with frozen core on a GPU.

To put coordinates directly in the input instead of using XYZ, replace `[molecule] xyz` with the following.
Do not specify both.

```toml
atom = """
Li 0.0 0.0 0.0
H  0.0 0.0 1.6
"""
unit = "Angstrom"
```

### Coordinates from result JSON

`molecule.xyz` accepts a single-frame XYZ file or an SCGVB result `.json` file.
For JSON, it reads only the top-level `molecule.atoms`, using
`molecule.coordinate_unit` (Angstrom if absent) and converting to the input `unit`.
It does not select a history entry or a checkpoint automatically. Multi-frame
XYZ trajectories still require extracting a single frame first.
Basis, charge, multiplicity, and active-space settings come from the new input.

To reuse both coordinates and the wavefunction, specify the result in both places:

```toml
[molecule]
xyz = "optimized_sto3g.json"
basis = "cc-pvdz"

[calculation]
initial_json = "optimized_sto3g.json"
```

Keep the required active-space settings in the new input. `xyz` alone imports
coordinates; `initial_json` supplies orbital and spin-coupling coefficients.
All file paths are relative to the TOML file. Use a different output path to
preserve the source file.

To restart after changing coordinates, use `xyz = "slightly_changed.xyz"`
(or a different geometry JSON) with `initial_json = "previous.json"`.
The input coordinates are retained and the saved wavefunction is projected onto
them by default. Set `calculation.allow_geometry_change = false` to require
matching coordinates instead. Atom types and order must still match.
The output `geometry_input` records the coordinate source and selected JSON field;
`calculation.initialization` records the wavefunction source and projection policy.

## Settings

Omitted settings use existing CLI defaults. Key names use underscores.

| Section | Available keys |
|---|---|
| `molecule` | `atom` or `xyz` (required), `basis`, `charge`, `multiplicity`, `unit` |
| `calculation` | `backend`, `device`, `energy_evaluation`, `initial_guess`, `initial_json`, `allow_geometry_change`, `seed`, `initial_perturbation`, `initial_spin_guess`, `initial_spin_overlap_cutoff`, `determinant_size_profile` |
| `active_space` | `mode`, `pi_electrons`, `pi_normal_mode`, `frozen_core`, `inactive_mos`, `active_mos`, `optimize_core`, `orbitals`, `ao_electrons`, `ao_threshold` |
| `optimizer` | `method`, `spin_overlap_cutoff`, `orbital_overlap_cutoff`, `orbital_micro_iterations`, `orbital_update`, `spin_solver`, `joint_correction`, `joint_correction_start`, `epsilon`, `max_outer_iterations`, `max_inner_iterations`, `core_max_iterations`, `core_gradient_tolerance`, `second_order`, `second_order_fallback`, `core_response` (legacy alias), `max_permutation_pairs` |
| `output` | `path`, `verbose` |
| `geometry` | `method`, `coordinates`, `bonds`, `symmetry_operations`, `freeze`, `set`, `constraint_tolerance`, `restart_json`, `max_seconds`, `max_iterations`, `max_evaluations`, `difference_step`, `gradient_tolerance`, `energy_tolerance`, `step_tolerance`, `trust_radius`, `max_radius`, `final_radius` |

`calculation.allow_geometry_change` defaults to `true`: `initial_json` is projected onto the input geometry when coordinates differ. Use `false` to require a match.
`geometry.freeze = ["xyz 1", "distance 2 3"]` holds specified quantities at their input values.
`geometry.set = ["angle 2 1 3 105"]` fixes explicit targets (Angstrom/degrees).
Both BOBYQA and quasi-Newton support these settings; see
[fixed geometry degrees of freedom](GEOMETRY_OPTIMIZATION.md#fixed-positions-distances-angles-and-dihedrals).
Select geometry optimization with `[geometry] method = "bobyqa"` or `"quasi-newton"`.
BOBYQA accepts `coordinates = "symmetry"` with a string list of Cartesian
`symmetry_operations`, for example `["-x,y,z", "x,-y,z", "x,y,-z"]` for D2h.
`geometry.restart_json` resumes a saved geometry checkpoint, including its
electronic state and disk evaluation cache. `geometry.max_seconds` limits
wall time per invocation between electronic solves. See
[Geometry optimization](GEOMETRY_OPTIMIZATION.md#geometry-restart-and-time-limits).
Omitting the method or using `"none"` selects a single-point calculation.
See [Geometry optimization](GEOMETRY_OPTIMIZATION.md) for units, internal coordinates, and convergence criteria.

`active_space.mode` accepts `full` / `pi` / `benzene-pi` / `ao`.
MO indices are **one-based**, as in the CLI: `inactive_mos = [1]`, `active_mos = [2, 3]`.
Settings and combination constraints match [ACTIVE_SPACE.md](ACTIVE_SPACE.md).
`optimize_core = true` relaxes selected inactive orbitals while retaining double occupation.

```toml
[optimizer]
max_outer_iterations = 50
max_inner_iterations = 1000
core_max_iterations = 50
core_gradient_tolerance = 1e-6
```

Core relaxation normalizes each initial augmented-Hessian (AH) Davidson direction,
including the previous macro iteration's correction. This prevents a small but
useful direction from being discarded by the AH overlap cutoff near convergence.
The physical rotation is still solved by the AH method with the same step limits
and convergence thresholds;
`core_optimization.ah_seed_history` records the seed source and scale.

The orbital step cap shrinks by a factor of 0.3 only for an energy increase
larger than the core energy tolerance (1e-9 Eh). On an energy plateau, an
unconverged orbital gradient instead restores the cap geometrically toward
its configured maximum; a converged gradient holds the cap. A resolved energy
decrease also restores it. This avoids shrinking every step simply because the
energy change is small. AH trust checks and the full orbital-gradient threshold
remain active. `core_optimization.ah_step_history` records these decisions.

Optional `[optimizer] second_order = true` enables analytic joint C/Cs Newton
corrections in each internal SCGVB solve. With `optimize_core = true`, it also
adds the internal response to the outer orbital Hessian. With
`optimize_core = false`, the internal corrections remain active. The default is
`false`; CLI `--second-order` / `--no-second-order` overrides a file setting.
The old `core_response` key and `--[no-]core-response` flags are aliases for the
same common option; old enabled settings now include internal corrections too.
Contradictory TOML/API aliases are rejected; an explicit CLI flag takes
precedence over the file regardless of which spelling it uses. This bounded
CPU method retains the convergence criteria. See [second-order SCGVB
corrections](SECOND_ORDER.md) for the method, allocation limit and diagnostics.

To try the existing optimizer first and enable second-order corrections only
after electronic nonconvergence, set:

```toml
[optimizer]
second_order = false
second_order_fallback = true
```

The retry is made once per electronic calculation, with the same iteration
limits and convergence criteria. It applies with either `optimize_core` setting,
including electronic evaluations during geometry optimization. `second_order =
true` still enables corrections from the start. Automatic retry defaults to
`false`; use `--second-order-fallback` / `--no-second-order-fallback` to override
the input, or `calculate(..., second_order_fallback=True)` in Python. See
[automatic retry](SECOND_ORDER.md#automatic-retry-after-nonconvergence) for seed
selection, size limits, and result records.

## Simultaneous and alternating optimization

Omitting the method uses the existing simultaneous optimizer (`simultaneous`). Alternating optimization must be selected explicitly.

```toml
[optimizer]
method = "alternating" # "simultaneous" selects the original method
spin_overlap_cutoff = 1e-10
max_outer_iterations = 50
max_inner_iterations = 1000
```

Use `--optimization-method alternating --spin-overlap-cutoff 1e-10` in the CLI, or `calculate(..., optimization_method='alternating')` in Python.
Existing inputs retain their method unless explicitly modified; there is no automatic switching.
`--check` reports method and cutoff, and result JSON stores `calculation.optimization_method`.
An Fe(CO)5 input is provided at `examples/inputs/fe_co5_alternating.toml`.

Alternating optimization constructs the spin-function Hamiltonian and overlap at fixed nonorthogonal orbitals, then solves a generalized eigenproblem for spin coefficients.
It fixes those coefficients for BFGS orbital updates, then resolves spin coefficients.
Only individual orbital normalization is imposed, not mutual orthogonality.
Determinant coefficients are derived from SCGVB orbitals and spin coefficients, not independent FCI variables.
`max_outer_iterations` counts alternating cycles; `max_inner_iterations` limits accepted steps in each orbital update.
With `optimize_core=true`, the selected method is also used inside every closed-shell relaxation step.

Adding `orbital_micro_iterations = 5` splits each cycle's orbital updates into blocks of at most five accepted steps, resolving spin coefficients after every block.
BFGS history resets when spin coefficients change.
`max_inner_iterations` limits accepted steps across all blocks of the cycle; the norm floor is shared across that cycle and is not reduced at block boundaries.
If neither orbitals nor spin coefficients can change, the cycle stops updating and checks the full gradient.
Spin-update counts, acceptance fractions, and residuals are saved in result JSON `micro_history` and detailed logs.
Omission or `0` retains the original long orbital block. This option applies only to `alternating`.
Use `--orbital-micro-iterations 5` in the CLI or `orbital_micro_iterations=5` in Python.

Alternating optimization also provides these optional choices. Defaults remain `orbital_update = "bfgs"` and `spin_solver = "canonical"`.

```toml
[optimizer]
method = "alternating"
orbital_micro_iterations = 10
orbital_update = "wavefunction"
spin_solver = "projected"
```

`wavefunction` constructs an orbital metric from derivatives of the normalized many-electron wavefunction and preconditions the initial BFGS inverse Hessian for each orbital block.
Relative damping 1e-8 affects only search directions, not physical energies or mutual orbital overlaps.
Each normalized-wavefunction step must satisfy `sqrt(1 - |overlap|²) <= 0.05`, along with the existing orbital-step limit 0.1, energy-descent condition, and norm floor.
The metric is rebuilt for each block after spin updates.

`projected` constructs actual determinant expansions from retained overlap directions and the current state, then orthogonalizes with their wavefunction overlap.
It evaluates the reference-energy-centered Hamiltonian directly in this small space, reducing precision loss from subtracting large matrices after construction.
Spin-eigenfunction basis order, the SCGVB variational space, and convergence based on the full physical gradient are unchanged.

Both choices require `energy_evaluation = "determinant"`.
For large systems with more than 36 orbital variables, `wavefunction` updates use the JAX backend.
CLI options are `--orbital-update wavefunction --spin-solver projected`; Python arguments use the same names with underscores.
Diagnostics also record orbital-trial rejection reasons: norm, nonfinite values, energy, or wavefunction change.

`orbital_update = "single_orbital"` fixes other orbitals and spin coefficients and solves a generalized eigenproblem for the Rayleigh quotient of the SCGVB wavefunction, which is linear in one orbital's coefficients.
Its overlap is the many-electron wavefunction overlap including other orbitals and spin coefficients, not the AO overlap.
`orbital_overlap_cutoff` (default 1e-10) selects numerically retained directions, always including the current state as one direction.
The actual determinant-vector metric is used for orthogonalization and projection of H−E.
Determinant coefficients are derived from orbitals and spin coefficients rather than optimized independently.

The eigenvector solution is a candidate. Acceptance requires strict decrease of the original physical energy, the cycle-shared norm floor, an S-metric orbital-step limit of 0.1, and a normalized-wavefunction distance limit of 0.05.
If needed, the mixing fraction toward the candidate is repeatedly halved.
Nonorthogonal orbitals are never mutually orthogonalized.
Existing convergence criteria based on the full physical gradient 1e-6 and normalization error remain unchanged.

For this method, `max_inner_iterations` limits attempted single-orbital eigensolves per cycle; `orbital_iterations` counts accepted updates.
Attempt counts, candidates, acceptance fractions, rejection reasons, and matrix-construction/eigensolve timings are saved in `orbital_geometry`.
Orbital-index cycling continues across spin updates and cycles.
Example settings to update all 12 orbitals before a spin solve:

```toml
[optimizer]
method = "alternating"
orbital_update = "single_orbital"
orbital_overlap_cutoff = 1e-10
spin_solver = "projected"
orbital_micro_iterations = 12
max_inner_iterations = 12
max_outer_iterations = 3
```

`single_orbital` requires `energy_evaluation = "determinant"` with both NumPy and JAX.
It constructs derived determinant vectors for one orbital at a time, not all orbitals simultaneously.
Projection-basis width remains fixed to avoid GPU recompilation when the retained rank changes; zero directions are removed in the physical metric.
The existing simultaneous optimizer, BFGS, and wavefunction-metric BFGS remain available.

`spin_overlap_cutoff` is relative to the largest eigenvalue of the spin-overlap matrix.
Directions at or below the cutoff are excluded, so results may depend on the cutoff.
This eigenvalue threshold is not used as the wavefunction-norm floor.
At the start of each alternating cycle, the norm with individually normalized orbitals is computed; half of it becomes the common protection floor for spin and orbital updates.
The next cycle recomputes the floor from its accepted starting state.
Orbital updates convert the S metric to Euclidean coordinates and constrain only individual orbital norms.
Each accepted orbital change is limited to S-metric norm 0.1, with Armijo backtracking checking physical energy decrease and the norm floor.
Trials below the floor halve the step and retry. Rejected points are not assigned zero gradients; invalid starting points are errors.
Failure to update does not count as convergence while a gradient remains.
If orbital updates change the retained space and the spin solution lies above the incoming state, that valid incoming state is added as one direction to the solve space.
This preserves a variational comparison with the incoming state without restoring every discarded eigendirection.
Spin updates check the same floor and nonincreasing physical energy, backtracking toward incoming coefficients when necessary.
If no update is acceptable, incoming coefficients are retained and full-gradient checks continue.
Final convergence requires the full physical gradient, including discarded directions, to be at most the existing `1e-6` and wavefunction-norm error below `epsilon`.
Energy stagnation alone is not convergence.

Result JSON `alternating_optimization` records update history, overlap eigenvalues, retained rank, cutoffs, and gradients.
`spin_incoming_preserved` indicates whether the incoming state was added; `spin_subspace_dimension` is the dimension actually solved afterward.
`cycle_reference_norm` and `cycle_norm_floor` document common norm protection for both updates.
`orbital_max_accepted_step`, `orbital_rejected_trials`, and `spin_step_fraction` record update sizes and backtracking outcomes.
Orbital status codes are 0=gradient tolerance reached, 1=iteration limit, 2=no descent step, and 3=stop involving norm-protection rejection.
These describe the inner block, not complete-calculation convergence, which requires a full-gradient check.
`verbose=2` saves `orbital_progress` for inspecting inner updates.
Failure snapshots contain the state after the latest orbital update.
With closed-shell relaxation, the JSON history covers the final inner solve; save `output.verbose=2` diagnostics for histories of all solves.
Both NumPy/JAX and both energy evaluators are supported.
JAX determinant evaluation reuses fixed-orbital integrals and minors and processes spin states individually to avoid large intermediates.
Small spin eigensolves and BFGS orbital updates run on the CPU.

## Paths, precedence, and output

- Input-file XYZ and output paths are relative to the **directory containing the input file**.
- Omitting `output.path` writes a same-stem `.json` beside the input.
- Explicit CLI options override input settings. CLI paths are relative to the current working directory.
- `--no-optimize-core` / `--no-verbose` disable input values of `true`.
- Unknown sections or keys, incorrect types, and invalid choices are errors.
- JSON `input_file` stores the absolute input path and contents.
- Python 3.11 and later use standard-library TOML parsing; earlier Python uses the conditional `tomli` dependency in `requirements-base.txt`.

This input format applies to `calc_molecule.py` for single-point and explicitly requested geometry-optimization calculations.
Distance-scan settings for `calc_diatomic_mol.py` remain editable in `scgvb/cli/diatomic.py`.

## Atomic-charge output

After calculation, Mulliken, Löwdin, and meta-Löwdin charges and spin populations are displayed automatically and saved by method in JSON `population_analysis.methods`.
No additional input settings are required. See [MOLECULE_INTERFACE.md](MOLECULE_INTERFACE.md) for postprocessing existing JSON.

## Selecting active/inactive spaces from atomic orbitals

With `mode = "ao"`, specify whitespace-separated rows in `orbitals` containing **one-based atom indices** from XYZ, MINAO reference-orbital names, and a role.
Multiple orbitals on the same atom can share one row.

```toml
[active_space]
mode = "ao"
orbitals = [
    "1 2px 2py 2pz A", # Use atom 1's three p orbitals as active-space targets
    "1 1s F",         # Permanently frozen doubly occupied space
    "2 1s I",         # Doubly occupied space, relaxable with optimize_core=true
]
ao_threshold = 0.2
# ao_electrons = 6    # Check the selected active-electron count; do not force it
optimize_core = true
```

This illustrates syntax; adapt orbital names and electron counts to the molecule. A runnable allyl input is `examples/inputs/allyl_ao.toml`.

- `A`: reference AOs selecting the active space.
- `I`: reference AOs selecting doubly occupied inactive space, frozen when `optimize_core=false`.
- `F`: reference AOs selecting doubly occupied frozen space, fixed even with `optimize_core=true`.
- If `A` is specified, unselected doubly occupied HF orbitals are treated like `I` and unselected virtual orbitals become external. Without `A`, all HF orbitals outside `I/F` become active.
- All singly occupied HF orbitals must be active; excluding them is an error.

The specified AOs are not occupied directly. Instead, combinations of HF orbitals are selected by projection onto MINAO reference AOs, retaining eigenvalues at least `ao_threshold` (strictly between 0 and 1; default 0.2).
`F`, then `I`, are selected from the doubly occupied space; `A` is selected by occupation sector from the remainder.
Virtual components of `I/F` are not selected. The number of requested AOs need not equal the number of selected MOs.
Overlapping projected spaces requested by different roles are detected and stop processing.
Selection eigenvalues, AO requests, electron counts, and frozen-orbital counts are saved in active-space JSON metadata.

`2pz` follows the **Cartesian z axis of the input**, not an automatically detected molecular-plane normal.
Individual atoms of the same element can be selected separately. Element-wide selection and orbital-name wildcards are not supported.
Orbital names must exactly match MINAO labels such as `1s`, `2s`, `2px`, `2py`, and `2pz`.
Whole-shell shorthand is supported: `"1 2p A"` is equivalent to `"1 2px 2py 2pz A"`.
Shells such as `3d` expand to all components present in that atom's MINAO. Shorthand works for A/I/F and can be mixed, as in `"1 2s 2p A"`.
Overlapping requests such as `2p` plus `2pz` are errors.
This is not direct indexing of additional radial or polarization functions in calculation bases such as cc-pVDZ.
Missing reference-MINAO orbitals, out-of-range atoms, duplicates, and role conflicts report their reason and exit with code 1 before HF, regardless of calculation basis.
Projection-dependent conflicts are detected after HF.
Do not combine with `frozen_core`, `inactive_mos`, `active_mos`, or pi modes.

With `optimize_core=true`, inactive spaces other than F and the active space can relax, so final AO compositions may differ from their initial selection.
AO requests select an initial space; they do not constrain atom localization or p-orbital direction during optimization.

## Initial orbitals from saved JSON

```toml
[molecule]
xyz = "../thiophene.xyz"
basis = "cc-pvdz"
multiplicity = 1

[calculation]
backend = "jax"
device = "cpu"
energy_evaluation = "determinant"
initial_json = "thiophene_ao.json" # Relative to this TOML file

[active_space]
mode = "ao"
orbitals = ["1 2pz A", "2 2pz A", "3 2pz A", "4 2pz A", "5 3pz A"]
ao_electrons = 6
optimize_core = true

[output]
path = "thiophene_ccpvdz.json" # Use a name different from the source
```

`initial_json` overrides `initial_guess` and `initial_perturbation` and loads saved SCGVB orbitals, spin-coupling coefficients, inactive orbitals, and the active basis.
Atom order, charge, multiplicity, active-electron count, inactive-orbital count, and frozen-F-orbital count must match.
Changed geometry is allowed by default: the input coordinates are retained and the saved wavefunction is projected onto them.
With `allow_geometry_change = false`, coordinates must match within 1e-7 Bohr. Atom reordering is not supported.
See [Geometry optimization](GEOMETRY_OPTIMIZATION.md#wavefunction-transfer) for changed-geometry transfer.

For a different basis at the same geometry, cross-AO overlaps solve `S_new C_new = S_new,old C_old`, restoring core orthonormality and core–active orthogonality.
The F space is retained separately from I. If the active-basis dimension increases, an orthogonal complement from the new selection is added.
Dimension reduction or projection rank loss is an error. Cs is retained and the projected wavefunction norm is restored to one.
Basis projection is approximate: preservation of the original energy and subsequent convergence are not guaranteed.

At unchanged geometry, loaded core/active spaces define the initial spaces and can differ from a fresh HF selection in the new basis.
With `optimize_core=false`, projected core/active spaces are fixed while SCGVB coefficients are optimized within them; use true to relax the spaces themselves.
For changed geometries, the newly selected target spaces are retained and only the initial wavefunction is projected.
The CLI also accepts `--initial-json result.json`; `--check` validates input and projection.

## Automatic bond-order output

Both normal calculations and saved-JSON analysis with `scgvb.analysis.population` display Mayer bond orders and Wiberg–Löwdin bond indices after charge/spin analysis, without additional settings.
They share the normalized full SCGVB one-electron density, including inactive electrons, without reoptimization.

For `P=P_alpha+P_beta`, `R=P_alpha-P_beta`, and AO overlap `S`, summation over AO blocks of atoms A and B gives:

- Mayer: `sum [(PS)_mu,nu (PS)_nu,mu + (RS)_mu,nu (RS)_nu,mu]`, including the spin-density term for open shells.
- Wiberg–Löwdin: `sum (P_L)_mu,nu (P_L)_nu,mu`, with `P_L=S^(1/2) P S^(1/2)`. This uses total electron density without an additional spin-density term and is not the NBO/NAO variant.

These definitions correspond to equations (5.12) and (5.13) in the [ORCA population-analysis manual](https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/population.html).
They are analysis indices applied to a correlated wavefunction's one-electron density, not exact electron-pair sharing counts from the two-electron density.
They depend on basis, orthogonalization, and spin state and need not equal integer formal bond orders or bond energies.
Negative values are not clipped.

JSON `population_analysis.bond_orders` stores all atom `pairs`, symmetric `matrices` with keys `mayer` and `wiberg_lowdin`, and definitions.
Standalone analysis puts `bond_orders` at the top level.
Atom indices are one-based and matrix ordering follows input atom order.
Diagonals are zero; intra-atomic indices are not calculated.
Console output alone is restricted to pairs with either absolute index at least 0.05; all pair values remain in JSON.

## Optimization diagnostic output levels

`[output] verbose` accepts `false` / `true` or integers `0` / `1` / `2`.
`false` / `0` disables progress; `true` / `1` retains existing progress output; `2` also saves coefficients, integrals, and convergence indicators.
Use `--verbose 2` or `--verbose=2` in the CLI. Plain `--verbose` is level 1 and `--no-verbose` is level 0.

```toml
[output]
path = "thiophene_ccpvdz_verbose.json"
verbose = 2
```

Diagnostics are saved beside output JSON in `<output_stem>_diagnostics/run-*/`.
Each execution creates a new directory without overwriting previous diagnostics.
In Python, use `calculate(..., verbose=2, diagnostics_dir="diagnostics")`; omission uses `scwf_diagnostics/run-*/` in the working directory.
Levels 0/1 create no diagnostic files.

- `metadata.json`: PySCF molecule information (`molecule` can be restored with `pyscf.gto.loads`), AO labels, active-electron count, spin, evaluator, and device.
- `reference.npz`: AO overlap matrix and spin-coupling branch diagrams.
- `solve-NNNN.npz`: each inner SCGVB solve's overlap `S`, effective one-electron integrals `h1`, two-electron integrals `eri` in chemists' notation, AO active basis `basis`, inactive orbitals `inactive`, combined nuclear/inactive constant `ecore`, and outer orbital-optimization iteration index.
- `events.jsonl`: one event per line, recording inner solve `solve`, outer iteration `macro_iteration`, inner augmented Lagrangian `iteration` when applicable, energy numerator, norm, total energy, unit-orbital norm, maximum gradient, penalty coefficient, multiplier, and errors.
- `event-*.npz`: the starting state, candidates after penalty optimization or physical-energy refinement, accepted states, and the last saveable state on failure. It stores raw variables `x`, optimization-basis coefficients `C`, normalized spin coefficients `Cs`, AO coefficients `ao_C`, orbital overlap, unit-orbital-overlap eigenvalues, and physical gradients. Orbital-update events store updated `basis` and `inactive`; the previous state corresponds to the preceding solve/accepted event.

Read NPZ files with `numpy.load(path, allow_pickle=False)`.
Total energy is `numerator / norm + ecore`.
Diagnostics remain even if a failed calculation produces no result JSON.
They are analysis files, not restart files directly usable as `initial_json`.
Unavailable indicators record `evaluation_error`, while raw variables are retained.
Level 2 adds energy/gradient evaluations and file output, increasing runtime and storage.
Optimization algorithms and convergence criteria are unchanged.

For each fixed Hamiltonian, the inner optimizer retains the best normalizable physical state.
Comparison uses `numerator / norm`, not the penalty objective.
Candidates more than `1e-10 Eh` above the best value are not used for restarts; the retained state is restored instead.
Five consecutive restores without improvement stop as nonconverged, without relaxing criteria.
At level 2, `best` records a new best state and `best_restart` records restart from it.
Failure events also store the retained best state, or the latest state if none was obtained.
The best value resets for every inner solve; energies with different integrals are not compared.

Optional `[optimizer] joint_correction = true` adds a bounded correction of
orbital and spin coefficients together after each alternating cycle from
`joint_correction_start` onward (default 1). It requires
`method = "alternating"`, `spin_solver = "projected"`, and determinant energy
evaluation. The default is `false`. See `JAX_BACKEND.md` for the experimental
six-direction approximation, physical bounds and unchanged convergence tests.

## Independent spin initialization and joint-correction start

```toml
[calculation]
initial_guess = "hf"
initial_spin_guess = "eigen"
initial_spin_overlap_cutoff = 1e-12

[optimizer]
method = "alternating"
orbital_update = "single_orbital"
spin_solver = "projected"
spin_overlap_cutoff = 1e-12
orbital_micro_iterations = 3
max_inner_iterations = 12
max_outer_iterations = 50
joint_correction = true
joint_correction_start = 21
```

`initial_spin_guess = "eigen"` solves the generalized spin eigenproblem at fixed initial orbitals to obtain Cs for optimization.
It is independent of orbital `initial_guess` and optimizer `method`.
It supports HF/core/random orbitals, simultaneous/alternating optimization, single-orbital and wavefunction updates, and inactive-orbital relaxation.
Initialization leaves orbitals unchanged. Determinant evaluation uses the centered projected solver; permutation evaluation uses the existing spin matrices.
`initial_spin_overlap_cutoff` is separate from optimization `spin_overlap_cutoff` and defaults to 1e-10.
The solve uses the numerically retained metric subspace and incoming state, rejecting actual energy increases.
Small metric eigenvalues are not inverted unconditionally.

The default `initial_spin_guess` remains the compatible `"uniform"`.
New calculations start Cs at equal coefficients 1/sqrt(fs); `initial_json` restarts retain saved Cs.
Only explicitly selecting `"eigen"` on restart resolves Cs at the loaded orbitals.
Initialization results, Cs, metric rank, before/after active energies, and acceptance are recorded in JSON `calculation.spin_initialization`.
Detailed diagnostics also save an `initial_spin` event.
Alternating optimization still performs its first spin solve, and initialization/optimization can use different cutoffs.

The example performs alternating optimization alone for the first 20 sweeps, then adds joint correction after alternating updates at the end of every sweep from sweep 21 onward.
It does not replace alternating optimization with an independent correction-only solve.
Orbitals, Cs, and norm are not reset at the transition.
`joint_correction_start` is an integer at least 1; 1 retains correction from the first sweep.
Any other value requires `joint_correction = true`.
Sweep diagnostics record `optimization_phase` and `joint_correction_active`.

The start index is one-based within each `optimize` call. Restart and every inactive-relaxation inner solve begin again at 1.
A start beyond the iteration limit produces no corrections during that call.
If full-gradient and normalization criteria are satisfied before the transition, the solve converges without correction.
The finite preconditioning period awaiting correction does not stop on energy stagnation alone; it proceeds to the requested start.
The stagnation counter resets at the transition, then the existing five-stagnation nonconvergence rule applies.
Energy-increase checks, iteration limits, and full-gradient 1e-6/normalization convergence criteria remain active throughout.

Corresponding CLI options are `--initial-spin-guess eigen`, `--initial-spin-overlap-cutoff 1e-12`, and `--joint-correction --joint-correction-start 21`.
An input is provided at `examples/inputs/fe_co5_staged_joint.toml`.
Twenty sweeps is an illustrative choice, not a validated optimum for Fe(CO)5.

### Distinction from experimental correction APIs

Normal `joint_correction = true` invokes the correction with at most six directions in `scgvb.optimization.coupled.correction`.
No TOML key selects residual-augmented `adaptive_correction` or orbital-only-damped `scgvb.optimization.spin_response.response_correction`.
Invoke them explicitly through the diagnostic scripts or low-level APIs described in [HPC_README.md](HPC_README.md#comparing-joint-corrections-from-a-saved-state).
These scripts try only joint corrections from the same saved state without intervening alternating optimization.
That differs from adding corrections to alternating updates using `joint_correction_start` above.
These Fe(CO)5 trials did not converge and are not converged example configurations.
