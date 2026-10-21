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

Restart a single-point calculation at a changed geometry by supplying the new
coordinates and the old electronic result separately:

```toml
[molecule]
xyz = "slightly_changed.xyz" # A result JSON is also accepted

[calculation]
initial_json = "previous.json"
```

Changed-geometry projection defaults to enabled in TOML, the CLI, and
`calculate(..., initial_json=path_or_result_dict)` in Python. Set
`allow_geometry_change = false` or `--no-allow-geometry-change` to require matching
coordinates for a single-point restart. Geometry optimization always enables
projection for its internal evaluations.

`xyz = "result.json"` reads the top-level `molecule.atoms`, with its stored
coordinate unit. It imports coordinates only; use `initial_json` as well to load
the wavefunction. Histories and checkpoints are not selected by `xyz`.
Explicit `geometry.restart_json` continues the geometry optimizer from its saved
best checkpoint structure, overriding the supplied input coordinates. To start
at a deliberately modified structure, use `initial_json` without
`geometry.restart_json`. See [JSON geometry input](INPUT_FILE.md#coordinates-from-result-json).

## Fixed positions, distances, angles, and dihedrals

Both existing solvers support hard geometry constraints. Add `freeze` to hold
values from the input geometry, or `set` to impose explicit target values:

```toml
[geometry]
method = "bobyqa" # or "quasi-newton"
coordinates = "auto"
freeze = ["xyz 1", "distance 2 3"]
set = ["angle 2 1 3 105.0"]
constraint_tolerance = 0.0000001
```

The types and syntax are:

| Type | Freeze at input value | Set an explicit value |
|---|---|---|
| Cartesian components | `"x 1"`, `"yz 2"`, `"xyz 3"` | `"x 1 -0.2"`, `"xyz 3 0 0 1.5"` |
| Distance | `"distance 1 2"` | `"distance 1 2 1.5"` |
| Angle (vertex is the middle atom) | `"angle 1 2 3"` | `"angle 1 2 3 110"` |
| Signed dihedral | `"dihedral 1 2 3 4"` | `"dihedral 1 2 3 4 -170"` |

Atom indices are one-based. Write one constraint per string; atom ranges are
not supported. Cartesian selectors are `x`, `y`, `z`, `xy`, `xz`, `yz`, or
`xyz`. Positions and distances in `set` are always in **Angstrom**, and angles
and dihedrals in **degrees**, including when the molecular input uses Bohr.
`freeze` uses the physical input values after conversion. Dihedral differences
wrap at ±180 degrees. Angles must be strictly between 0 and 180 degrees;
undefined linear-angle/dihedral constraints are rejected.

`constraint_tolerance` is the maximum absolute error in the native units of
each constraint (Angstrom or degrees); its default is 1e-7. Explicit targets
are satisfied before the first electronic evaluation, while frozen quantities
remain at their original input values. Conflicting targets or unsatisfiable
combinations fail before electronic optimization. Duplicate identical
constraints are deduplicated. All fixed degrees of freedom cause a single
electronic evaluation and a `no_free_variables` result.

CLI examples:

```bash
python calc_molecule.py examples/inputs/h3_plus_geometry_fixed.toml
python calc_molecule.py examples/inputs/h3_plus_geometry_fixed.toml --geometry-method quasi-newton
python calc_molecule.py examples/inputs/h3_plus_geometry_fixed.toml --check
# Repeat these options to specify multiple constraints:
python calc_molecule.py --atom 'O 0 0 0; H 0 0 1; H .9 0 -.2' \
  --energy-evaluation determinant --geometry-method bobyqa \
  --geometry-freeze 'xyz 1' --geometry-set 'distance 1 2 1.1'
```

Python uses `optimize_geometry(..., freeze=["xyz 1"],
set_values=["distance 1 2 1.1"], constraint_tolerance=1e-7)`.

BOBYQA retains its existing trust-region algorithm. Its local chart keeps only
independent free variables, and back-transformation solves both the requested
free displacement and the fixed physical quantities. Fixing an arbitrary DLC
component is not used as a substitute for fixing a physical bond or angle.
The fixed Cartesian gauge removes only rigid motions compatible with Cartesian
anchors, so frozen atoms do not accidentally remove additional internal
relaxation directions. Constraint targets persist across chart changes.
The existing symmetry mode can be combined with compatible constraints;
incompatible symmetry and fixed targets are rejected.

Quasi-Newton retains geomeTRIC's trust region and BFGS updates, using its
constrained DLC/HDLC coordinates. Each trial step is corrected to the requested
constraint tolerance. Cartesian finite-difference probes may temporarily leave
the constraint surface to compute the complete gradient, but these infeasible
probes are **never** used as best-geometry or restart states. This method still
normally needs 6N electronic solves per numerical gradient.

Constraint forces need not vanish. Quasi-Newton convergence uses the Cartesian
gradient projected onto the allowed tangent space, together with energy,
displacement, and constraint-error checks. Results retain the full nuclear
gradient and additionally report `projected_nuclear_gradient_hartree_per_bohr`,
`projected_nuclear_gradient_rms`, and `projected_nuclear_gradient_max_abs`.
Both methods report normalized `constraints`, a per-constraint
`constraint_report`, and `constraint_error_max`. BOBYQA still uses the final
search radius for convergence rather than a nuclear-force certification.

Geometry checkpoints preserve the original frozen targets. Resume with the
same `freeze`, `set`, and `constraint_tolerance` settings; changing them is
rejected. As with other implementation changes, checkpoints from an older
source version cannot be replayed. To start a new constrained search from such
a result, use it as `initial_json` instead of `geometry.restart_json`.

## Point-symmetry constraints

BOBYQA supports `coordinates = "symmetry"`. Supply generators of the desired
finite point group, in the Cartesian frame of the input geometry:

```toml
[geometry]
method = "bobyqa"
coordinates = "symmetry"
symmetry_operations = ["-x,y,z", "x,-y,z", "x,y,-z"]
trust_radius = 0.03
max_radius = 0.15
final_radius = 0.0001
max_evaluations = 300
max_seconds = 342000 # 95 hours, checked between electronic calculations
```

These three mirror generators preserve D2h for the supplied diborane geometry
and automatically produce four independent variables. The implementation has
no molecule-specific atom numbers or bond definitions. It infers same-element
atom permutations, intersects the invariant displacement space with the space
excluding rigid motion, and uses an orthonormal Cartesian basis for BOBYQA.
Every finite trial displacement preserves all supplied operations. Input
geometries that violate the operations are rejected. The symmetry origin is
the arithmetic centroid of the input atoms, and the frame is kept fixed.

Each operation is either three signed axes (`"y,x,-z"`) or nine row-major
entries of an orthogonal matrix (`"1 0 0; 0 -1 0; 0 0 1"`). Numeric matrices
also support rotated frames and other finite point groups. Generators suffice;
listing the complete group is unnecessary. This mode currently requires
BOBYQA. In the CLI repeat `--geometry-symmetry-operations=-x,y,z`, using the
equals sign when an operation starts with a minus sign.

The variables are orthonormal combinations of allowed displacements, with the
same RMS Cartesian radius convention as other BOBYQA coordinates. For D2h
diborane they span changes in B-B, bridge B-H, terminal B-H and terminal H-B-H
angle. A symmetry-constrained search certifies convergence within that space.

## Geometry restart and time limits

Each completed electronic evaluation is written atomically to an on-disk
cache beside the output; the main output also checkpoints the best geometry,
its converged wavefunction, accumulated history, and the active BOBYQA chart.
Use the same input and source release to resume:

```bash
python calc_molecule.py input.toml \
  --geometry-restart-json result.json --output result.json
```

Or set `geometry.restart_json = "result.json"`, relative to the TOML file.
The saved best geometry replaces the input geometry. Atom order, electronic
settings, source implementation, method, symmetry generators, coordinate type,
and initial/maximum chart radii must remain compatible. Evaluation and wall-time
budgets apply to each invocation; cumulative counts are also retained.

The active coordinate chart, starting radius, and reference wavefunction are
restored. BOBYQA rebuilds its interpolation model by replaying completed
evaluations from the disk cache. Native library optimizer objects are not
serialized. Replayed points require no new electronic solve; an unfinished
electronic calculation is recomputed. Retain all `*_geometry_cache` directories
referenced by the checkpoint. A missing cache makes those points require
recalculation, while the saved best wavefunction remains available.
Reusing the same library versions improves replay reproducibility.

`max_seconds` stops before starting another uncached electronic solve once its
budget is exhausted. A calculation already in progress may run beyond that
budget. Use a margin below the scheduler limit. The main output survives a hard
kill, with the last complete electronic state. `hpc/run_geometry.py` additionally
handles Slurm's USR1 and TERM signals as interruptions; submit with
`#SBATCH --signal=USR1@300` to request a checkpoint before the wall-time limit.
Use a separate log for each submission and a shared checkpoint/cache location.

## Results, convergence, and interruption

The result JSON adds `geometry_optimization` to the usual single-point result.
It records the method, coordinate definitions, settings, convergence status, evaluation count, compact evaluation history, and best-geometry history.
Quasi-Newton results also contain the final unprojected Cartesian nuclear gradient in Eh/Bohr.
Convergence requires the energy-change, RMS/maximum nuclear-gradient, and RMS/maximum displacement criteria to be satisfied. With fixed degrees of freedom, the gradient criterion applies to allowed directions and the fixed quantities must also satisfy `constraint_tolerance`.
BOBYQA convergence is based on the final search radius; it does not prove that nuclear gradients are below tolerance.
Providing quasi-Newton-only settings such as `gradient_tolerance` to BOBYQA does not add a force check.

The search stops if the electronic calculation fails normalization or stationarity checks.
Smaller finite-difference steps amplify electronic errors. Tighten electronic-optimization conditions as needed and check nuclear-gradient stability at different difference steps.
Discontinuities from local electronic solutions or active-space changes cannot be resolved by the geometry optimizer alone.

When `output` is specified, the best geometry is saved atomically to JSON after each successful electronic evaluation.
`<output_stem>_geometry.xyz` stores updates to the best geometry, which are distinct from the accepted-step sequence.
On evaluation limits, nonconvergence, or electronic failure, the best valid electronic state is saved and the CLI returns exit code 1.
Interruption also preserves the best state. If the first electronic evaluation fails before any valid state exists, an exception is raised.
The saved result can be supplied as `initial_json` for wavefunction transfer.
Use `geometry.restart_json` for geometry continuation with cached evaluations
and chart replay. Quasi-Newton continuation starts a new BFGS history.

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
