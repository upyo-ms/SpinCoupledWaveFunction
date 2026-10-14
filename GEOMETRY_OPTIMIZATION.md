# Geometry optimization

Energy-only BOBYQA and an internal-coordinate quasi-Newton method with central-difference nuclear gradients are available.
Both are local optimizers that run the existing SCGVB wavefunction optimization at each trial geometry.
They minimize `total_energy_hartree`, including nuclear repulsion and inactive-space energy.
Electronic optimization methods, backends, and core-relaxation settings are shared with single-point calculations.

## Installation and execution

Add the following dependencies to an existing CPU/GPU environment.

```bash
python -m pip install -r requirements-geometry.txt
# To install as a package
python -m pip install -e '.[geometry]'

python calc_molecule.py examples/inputs/h2_geometry_bobyqa.toml
python calc_molecule.py examples/inputs/h2_geometry_quasi_newton.toml
# Check calculation size and generated coordinates without SCGVB optimization (HF still runs for active-space selection)
python calc_molecule.py examples/inputs/h2_geometry_bobyqa.toml --check
```

Add `[geometry]` to an existing input. Omitting it retains single-point calculations.

```toml
[geometry]
method = "bobyqa"            # "quasi-newton" or "none"
coordinates = "auto"         # "dlc", "tric", or "cartesian" also supported
max_iterations = 100
max_evaluations = 2000
difference_step = 0.001      # Bohr; quasi-Newton only
gradient_tolerance = 0.0003  # Eh/Bohr; RMS nuclear gradient; quasi-Newton only
energy_tolerance = 0.000001  # Eh; quasi-Newton only
step_tolerance = 0.0012      # Angstrom; RMS displacement; quasi-Newton only
trust_radius = 0.1           # Angstrom
max_radius = 0.3             # Angstrom
final_radius = 0.0001        # Angstrom; BOBYQA only
# bonds = ["1-2", "2-3"]     # Complete bond list overriding automatic detection; one-based atom indices
```

CLI options prefix each key with `--geometry-`, for example `--geometry-method bobyqa` and `--geometry-coordinates auto`.
Use `--geometry-method none` to return to a single-point calculation.
`max_evaluations` limits actual electronic-optimization calls, including failed trials; cached results do not count.
For quasi-Newton, `max_iterations` counts geometry updates. For BOBYQA, it counts searches in fixed local coordinate charts, including chart restarts.

## Automatic internal coordinates

geomeTRIC estimates bonds from covalent radii and distances, then generates bond lengths, angles, dihedrals, out-of-plane displacements, and linear-bending coordinates.
`auto` selects DLC for a single fragment and TRIC, including translations and rotations, for multiple fragments. TRIC also handles isolated-atom positions.
Specify `bonds` when automatic detection is unsuitable, for example for metal complexes.
Geometries are updated without changing atom order.

For BOBYQA, the primitive-coordinate B matrix is restricted to the Cartesian subspace with overall translation and rotation removed. SVD then produces independent combinations: normally 3N−6 variables, or 3N−5 for a linear geometry.
The reference Jacobian is scaled so that the search-variable norm locally matches the Cartesian RMS displacement in Bohr. Specified radii are converted from Angstrom to Bohr.
Redundant primitive coordinates in rings are not optimized directly as independent variables.

Coordinate back-transformation is iterative and its residual is checked. Within each BOBYQA run, the reference geometry, coordinate basis, scaling, and initial wavefunction remain fixed so that the same variables always represent the same geometry.
When the search reaches the boundary of the local chart, coordinates are rebuilt at the best geometry and the search restarts.
Back-transformation failure also triggers a restart with a smaller radius rather than returning an artificial penalty energy.
A fixed Cartesian gauge removes overall translation and rotation; geomeTRIC handles periodic dihedral differences.
An error is raised if internal coordinates do not span the required degrees of freedom. Explicit `cartesian` coordinates are also supported.

The quasi-Newton method uses geomeTRIC DLC/TRIC coordinates, a trust region, and BFGS Hessian updates.
Gradients use central differences of every Cartesian component, so each gradient normally requires 6N electronic optimizations.
Positive and negative displacements start from the same central-geometry solution.
Finite-difference evaluations currently run sequentially; parallel evaluation of independent displacements is not implemented.

## Wavefunction transfer

Initial orbitals are projected by solving `S_new C_new = S_new,old C_old`, using the AO overlap at the new geometry and the cross-overlap between new and old AOs.
Atom types and order, charge, multiplicity, active-electron count, and spin basis must match.
Nonorthogonal orbitals are not orthogonalized against one another; repeated use of the same spatial orbital is allowed.
The actual many-electron wavefunction norm is checked and SCGVB is reoptimized.

At a changed geometry, the selected active space and frozen core are redefined for that geometry; only the initial orbitals are projected into the new space.
This avoids transporting the old active subspace along the trajectory and thereby changing the energy surface.
When core relaxation is enabled, the existing core optimizer starts from the new geometry's initial core.
Numerical differences fully recompute the calculation, including the coordinate dependence of frozen cores and HF-based selection.
Discontinuous active-space selection does not guarantee a smooth energy surface.
`benzene-pi` is restricted to the xy plane. For general geometry optimization including out-of-plane displacements, use a selection suitable for the geometry, such as local `pi` selection.

Explicitly allow a changed geometry when restarting a single-point calculation:

```toml
[calculation]
initial_json = "previous.json"
allow_geometry_change = true
```

Use `--allow-geometry-change` in the CLI, or `calculate(..., initial_json=path_or_result_dict, allow_geometry_change=True)` in Python.
Geometry optimization enables this automatically for internal evaluations.
Existing same-geometry restarts and basis changes retain their default behavior.

## Results, convergence, and interruption

The result JSON adds `geometry_optimization` to the usual single-point result.
It records the method, coordinate definitions, settings, convergence status, evaluation count, compact evaluation history, and best-geometry history.
Quasi-Newton results also contain the final unprojected Cartesian nuclear gradient in Eh/Bohr.
Convergence requires the energy-change, RMS/maximum nuclear-gradient, and RMS/maximum displacement criteria to be satisfied.
BOBYQA convergence is based on the final search radius; it does not prove that nuclear gradients are below tolerance.
Providing quasi-Newton-only settings such as `gradient_tolerance` to BOBYQA does not add a force check.

The search stops if the electronic calculation fails normalization or stationarity checks.
Smaller finite-difference steps amplify electronic errors. Tighten electronic-optimization conditions as needed and check nuclear-gradient stability at different difference steps.
Discontinuities from local electronic solutions or active-space changes cannot be resolved by the geometry optimizer alone.

When `output` is specified, the best geometry is saved atomically to JSON after each successful electronic evaluation.
`<output_stem>_geometry.xyz` stores updates to the best geometry, which are distinct from the accepted-step sequence.
On evaluation limits, nonconvergence, or electronic failure, the best valid electronic state is saved and the CLI returns exit code 1.
Interruption also preserves the best state. If the first electronic evaluation fails before any valid state exists, an exception is raised.
The saved result can be supplied as `initial_json` for another run, but the BOBYQA interpolation model and BFGS history are not resumed.

Python API:

```python
from scgvb.optimization.geometry import optimize_geometry

result = optimize_geometry(
    "H 0 0 0; H 0 0 1.0",
    method="bobyqa", coordinates="auto", max_evaluations=200,
    calculation_options={"basis": "sto-3g", "energy_evaluation": "determinant", "initial_guess": "hf"},
    output="h2_optimized.json",
)
assert result["geometry_optimization"]["converged"]
```

References: [geomeTRIC coordinate generation and optimization](https://geometric.readthedocs.io/en/latest/how-it-works.html),
[Py-BOBYQA](https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html).
