# Spin Coupled Wave Function

This repository contains the program, usage documentation, molecular geometries, and example TOML inputs.
Test code, validation data, calculation results, and logs are excluded.
Tests and restarts from saved calculations described in the documentation require those files separately.

## Code organization

The library is organized as the `scgvb` package. NumPy and JAX backends and the optimization algorithms are available within the same package.

```text
scgvb/
  energy/        # NumPy/JAX energy evaluation, determinants, compound minors
  orbitals/      # Active spaces, AO/pi orbitals, initial guesses, restarts, molecular integrals
  optimization/  # Simultaneous/alternating/single-orbital/core optimization, initial spin, diagnostics
  spin/          # Spin functions, branch diagrams, electron arrangements, permutation representations
  io/            # TOML input, orbital export, orbital viewer
  analysis/      # Electron density, atomic charges, bond orders
  cli/           # Molecular calculations and diatomic-scan commands
  benchmarks/    # Performance measurements and evaluator comparisons
tools/           # HPC source distribution archives
hpc/             # Environment checks, job scripts, diagnostic commands
examples/        # Molecular geometries and TOML inputs
tests/           # Local regression tests (excluded from GitHub and distributions)
```

Existing commands `python calc_molecule.py`, `python calc_diatomic_mol.py`, `python generate_orbitals.py`, and `python build_hpc_bundle.py` remain available.
Use package-qualified names when importing from Python:

```python
from scgvb.cli.molecule import calculate, build_molecule
from scgvb.energy.numpy_backend import SCWF_Energy
from scgvb.orbitals.active_space import prepare_active_space
```

You can also run `python -m scgvb.cli.molecule` or `python -m scgvb.analysis.population`.
Install the package with `python -m pip install .`, or use `python -m pip install -e .` during development.
Installation also provides the `scgvb-calculate`, `scgvb-diatomic`, `scgvb-orbitals`, and `scgvb-population` commands.
Add JAX using `requirements-cpu.txt` or `requirements-gpu-cuda13.txt`.

Run the local test suite with `python -m unittest discover -s tests -t . -p 'test_*.py'`.

## Running from an input file

A TOML file can collect the molecule, basis, active space, and optimization settings.

```bash
python calc_molecule.py examples/inputs/allyl.toml
```

An allyl-radical example (cc-pVDZ, three pi electrons, core relaxation) is included.
See [Input file format](INPUT_FILE.md). The existing CLI is also available.

Geometry optimization supports energy-only BOBYQA and an internal-coordinate quasi-Newton method with numerical gradients.
Install `pip install -r requirements-geometry.txt` and select the method in the input's `[geometry]` section.
See [Geometry optimization](GEOMETRY_OPTIMIZATION.md) for automatic internal coordinates, orbital projection between geometries, convergence criteria, and examples.

Codes for "Augmented Lagrangian method for optimizing non-orthogonal orbitals"

# Dependency
- pyscf
- numpy
- scipy
- sympy
- matplotlib
- jax (optional, JAX backend only)

# Usage
General polyatomic single-point calculations (no Full CI; optional HF initial orbitals):

```bash
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --multiplicity 1 \
    --backend jax --device cpu --output h3_plus.json
python calc_molecule.py --xyz examples/h4.xyz --output h4.json
```

See [MOLECULE_INTERFACE.md](MOLECULE_INTERFACE.md) for geometry input, results,
the Python API and electron-count limitations. The existing diatomic test is
retained separately below.

Display the nonorthogonal orbitals saved in a calculation result:

```bash
python generate_orbitals.py h3_plus.json --output-dir h3_orbitals
```

Open `h3_orbitals/index.html` in a browser, or use the generated Cube files in
a molecular viewer. See [ORBITAL_VIEWER.md](ORBITAL_VIEWER.md).

Existing NumPy / complex-step augmented-Lagrangian version (default):

```bash
python calc_diatomic_mol.py
```

JAX float64 / automatic-differentiation version, using the same optimizer:

```bash
python calc_diatomic_mol.py --backend jax --device cpu
```

On a machine with a working NVIDIA/CUDA JAX installation:

```bash
python calc_diatomic_mol.py --backend jax --device gpu
```

Molecule, charge, spin, basis and distance range are configured in
`scgvb/cli/diatomic.py`. An explicitly requested unavailable GPU raises an error;
it does not silently fall back to CPU. See [JAX_BACKEND.md](JAX_BACKEND.md).

The diatomic scan defaults to `--energy-evaluation both`: permutation and
determinant SCGVB optimizations run independently, alongside a Full CI reference.
Both methods receive the same initial guesses and then each carries its own
converged `C` and `Cs` to the next distance. If one method fails, its curve stops
at its last converged point while the other method and Full CI continue. There
is no evaluator switching in a comparison. Single-method runs remain available
with `--energy-evaluation permutation|determinant`; `--no-evaluator-fallback`
is retained as a compatibility option (comparison runs never use a fallback).

```bash
python calc_diatomic_mol.py --backend jax --device cpu --output-dir runs/diatomic
```

`plot_energy.pdf` (also saved as PNG) compares **absolute total energies** in Eh for determinant,
permutation and Full CI on a common axis. `plot_Cs.pdf` (also PNG) shows the squared spin
coefficients of both SCGVB methods, using colors for coefficient indices and
line styles for methods. These squared coefficients are not generally physical
configuration probabilities for nonorthogonal spatial functions.
`diatomic_results.json` retains the raw converged energies, `C`, signed `Cs`,
Full CI references, and failure distances/messages. These files are updated
after every geometry, and completed points are saved on interruption as well.
The 10 Angstrom reference is recorded separately; its failure does not prevent
the absolute-energy scan. There is no automatic checkpoint loading between runs.
Determinant-space size limits still apply.

For JAX determinant evaluation with at most four electrons and more AO basis
functions than electrons, an exact moving orbital-span representation avoids
the expensive full AO determinant contraction. QR is only an internal change
of representation, not an orthogonality constraint on the optimized orbitals.
Rank-deficient orbital matrices use the original polynomial AO evaluator so
that derivatives remain defined. Larger electron counts retain the existing
memory-bounded evaluator.

Use `--initial-guess hf` for RHF/ROHF starting orbitals, or `--initial-guess core` for core-Hamiltonian orbitals. The default `random` retains the previous initialization. See `MOLECULE_INTERFACE.md`.

## Initial spin-coupling coefficients and optimization methods

Initial orbitals, initial spin-coupling coefficients `Cs`, and the optimization method can be selected independently.
`Cs` contains coefficients in the spin-coupled basis, distinct from the determinant-expansion coefficients constructed from orbitals and `Cs`.

| TOML setting | Default | Behavior |
|---|---|---|
| `[calculation] initial_spin_guess` | `"uniform"` | New calculations start with equal coefficients `1/sqrt(fs)`. `"eigen"` solves the generalized spin eigenproblem at fixed initial orbitals. |
| `[calculation] initial_spin_overlap_cutoff` | `1e-10` | Relative metric-eigenvalue cutoff for initial spin solving, independent of the optimization cutoff. |
| `[optimizer] method` | `"simultaneous"` | The existing simultaneous optimizer. `"alternating"` alternates spin solving at fixed orbitals with orbital updates at fixed coefficients. |
| `[optimizer] orbital_update` | `"bfgs"` | Orbital update for alternating optimization. `"wavefunction"` uses wavefunction-metric preconditioning; `"single_orbital"` solves generalized eigenproblems one orbital at a time. |
| `[optimizer] spin_solver` | `"canonical"` | Spin solver for alternating optimization. `"projected"` solves a centered Hamiltonian in the retained physical wavefunction subspace. |
| `[optimizer] orbital_micro_iterations` | `0` | Interval between spin solves during orbital updates: accepted steps for BFGS-based methods, attempted eigenproblem solves for single-orbital updates. `0` solves after the complete orbital block. |
| `[optimizer] joint_correction` | `false` | Add a simultaneous orbital/spin correction after alternating updates. |
| `[optimizer] joint_correction_start` | `1` | First sweep that includes joint correction. Alternating optimization continues afterward. |

Example settings for initializing spin coefficients from an eigenproblem and using alternating optimization follow.
Specify the molecule, basis, and active space in the input file.

```toml
[calculation]
energy_evaluation = "determinant"
initial_guess = "hf"
initial_spin_guess = "eigen"
initial_spin_overlap_cutoff = 1e-10

[optimizer]
method = "alternating"
orbital_update = "single_orbital"
spin_solver = "projected"
orbital_micro_iterations = 3
max_inner_iterations = 12
joint_correction = false
```

This example resolves spin coefficients after every three attempted single-orbital updates and limits each cycle to 12 orbital attempts.
`initial_spin_guess = "eigen"` also works with the existing `simultaneous` optimizer.
For `initial_json` restarts, `"uniform"` retains the saved `Cs`; explicitly selecting `"eigen"` recomputes coefficients at the loaded orbitals.

`wavefunction`, `single_orbital`, `projected`, and joint correction require determinant evaluation.
Joint correction additionally requires `method = "alternating"` and `spin_solver = "projected"`.
See [Alternating optimization](INPUT_FILE.md#simultaneous-and-alternating-optimization) and [Spin initialization and correction start](INPUT_FILE.md#independent-spin-initialization-and-joint-correction-start) for constraints, CLI/Python APIs, norm protection, and convergence criteria.

`scgvb.optimization.coupled.adaptive_correction`, which adds residual directions, and `scgvb.optimization.spin_response.response_correction`, which damps only orbital components, are experimental APIs invoked explicitly.
The standard `joint_correction = true` uses the existing correction with up to six directions and does not switch automatically to these experimental methods.
Diagnostic scripts comparing joint corrections from the same saved state are described in [HPC_README.md](HPC_README.md#comparing-joint-corrections-from-a-saved-state).
Fe(CO)5 trials did not converge; the included [staged-correction input](examples/inputs/fe_co5_staged_joint.toml) is a comparison configuration.

## Fixed inactive orbitals / active-space SCGVB

The existing simultaneous optimizer remains the default. Use
`--optimization-method alternating` (or `[optimizer] method = "alternating"`
in TOML) to solve spin coefficients at fixed orbitals and alternate with
bounded orbital updates with feasible backtracking. Both alternating blocks
share a relative wavefunction-norm guard based on the accepted state; it is
separate from the spin-metric eigencutoff. Both methods require the same full physical-gradient and
normalization checks. See [INPUT_FILE.md](INPUT_FILE.md) for the spin-overlap
cutoff, iteration limits, and diagnostic history.

Add `--orbital-micro-iterations 5` (or `orbital_micro_iterations = 5` under
`[optimizer]`) to re-solve spin coefficients after every short orbital block.
These blocks share one norm floor and one total orbital-step budget per outer
cycle; BFGS restarts when the spin objective changes. The default `0` retains
the original full orbital block. Intermediate spin acceptance fractions and
residuals are recorded in the micro-update history.

Add `--optimize-core` to relax the selected inactive orbitals while keeping them
doubly occupied. This alternates SCGVB optimization with orbital rotations;
it does not turn the calculation into FCI. Without this flag, inactive orbitals
remain fixed. See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for examples, tolerances,
saved diagnostics and the CPU/GPU division of work.

Full SCGVB remains the default. Fixed closed shells, manual HF MO selection,
a benzene 6-pi-electron preset, and orientation-independent planar hydrocarbon
`--active-space pi` selection (including COT with 8 pi electrons) are available. The determinant evaluator
works in NumPy and JAX without constructing electron-permutation tables.
See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for CH4/benzene commands, limitations, and validation.

## Single-GPU 12-pi-electron distribution

The determinant backend now supports 12e/12o singlets using bounded intermediate
arrays and a custom reverse-mode derivative. See [HPC_README.md](HPC_README.md)
for GB200/Linux aarch64 setup, validation, benchmarks and run scripts.
Build the portable source archive with `python build_hpc_bundle.py`.

Twisted sp2 hydrocarbons such as biphenyl can use `--active-space pi
--pi-normal-mode local --pi-electrons 12`. The included `examples/biphenyl_45deg.xyz`
is an idealized test geometry, not an optimized structure. See `ACTIVE_SPACE.md`.

### Cyclododecahexaene: 12-pi-electron ring benchmark

`examples/cyclododecahexaene.xyz` contains C12H12 ([12]annulene), with carbon
atoms 1--12 followed by hydrogens 13--24. This is an **idealized planar test
geometry, not an optimized equilibrium structure**. Ring C--C distances alternate
between 1.34 and 1.46 Angstrom; C--H distances are 1.08 Angstrom. The neutral
singlet input selects 12 pi electrons and uses cc-pVDZ, the JAX determinant
backend, and a GPU. Initially the 36 doubly occupied inactive orbitals are fixed.

```bash
python calc_molecule.py examples/inputs/cyclododecahexaene.toml --check
python calc_molecule.py examples/inputs/cyclododecahexaene.toml
```

Use `--device cpu` to override the device or `--optimize-core` to relax the
inactive/active spaces (additional CPU orbital-optimization cost). The check
performs HF and validates the active-space size; it does not establish SCGVB
convergence. For a nonplanar geometry, select local pi normals and validate its
coordination with `--check` before optimization.

For a 14-electron/14-orbital singlet, explicitly select determinant evaluation and `--determinant-size-profile large14`. The existing `standard` profile remains the default.
See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for TOML, Python API, restart, and inactive-orbital relaxation settings.
The phenazine input for the standard calculation code is `examples/inputs/phenazine14.toml`.
