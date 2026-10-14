# Displaying nonorthogonal orbitals

`generate_orbitals.py` reads result JSON saved by `calc_molecule.py` and exports each nonorthogonal orbital as a Gaussian Cube file and browser HTML.
The legacy example with hard-coded coefficients is now `examples/legacy_generate_cube.py`.
The new program does not require molecules or coefficients to be embedded in source code.
It requires neither rerunning SCF, Full CI, or orbital optimization nor installing JAX.

## Basic usage

```bash
python calc_molecule.py --xyz examples/h3_plus.xyz --charge 1 --output result.json
python generate_orbitals.py result.json --output-dir orbitals
```

Open the generated `orbitals/index.html` in a browser.
You can select orbitals, rotate and zoom with the mouse, and adjust the isovalue and opacity.
Positive amplitudes are red and negative amplitudes are blue. Click "Update view" after changing inputs.
Cube data is embedded in the HTML, so no Python server is required for display.

By default the HTML loads [3Dmol.js](https://3dmol.org/doc/GLViewer.html#addVolumetricData) from its official website and requires an internet connection.
To use a locally saved library, specify:

```bash
python generate_orbitals.py result.json --viewer-js /path/to/3Dmol-min.js
```

The library is copied into the output folder and referenced as a local file.
For offline use, keep the HTML and library in the same folder.
To omit HTML and display Cube files in an external molecular viewer, specify `--no-html`.

## Selecting orbitals and grids

```bash
# Export only the first and third orbitals (one-based indices)
python generate_orbitals.py result.json --orbitals 1 3 --output-dir selected

# Use a finer grid and larger box margin
python generate_orbitals.py result.json --grid 80 80 80 --margin 4 --isovalue 0.03

# Normalize each orbital separately for display (without orthogonalizing)
python generate_orbitals.py result.json --normalize
```

Defaults are all orbitals, a 64×64×64 grid, a 3 Bohr margin, and an absolute isovalue of 0.04.
The grid box is a rectangular volume enclosing the molecule; coordinates and margins are in Bohr.
Isosurfaces represent orbital amplitudes (Bohr^(-3/2)), not electron density.
With the same grid dimensions, larger molecules have wider grid spacing.
The default limit is two million grid points; adjust it with `--max-grid-points` if needed.

## Coefficient handling

The program evaluates `phi_i(r) = sum_mu C[mu,i] chi_mu(r)`.
By default it preserves the signs and magnitudes of saved coefficients and does not orthogonalize orbitals.
`--normalize` only divides each column by `sqrt(C_i.T @ S @ C_i)`, preserving orbital shape, phase, and nonorthogonality.
Because the overall orbital scale is arbitrary, this option is useful when comparing shapes at a common isovalue.
The program does not compute many-electron densities or occupations from spin coefficients.

AOs are reconstructed from the molecule, charge, multiplicity, basis, and coordinate units in the result JSON.
AO-label ordering and the overlap matrix are checked; mismatches stop processing.
This prevents coefficients from a different basis or atom order from being displayed incorrectly.
Input JSON is left unchanged; existing output files with the same names are overwritten.

## Output files

- `orbital_001.cube`, etc.: signed amplitudes for each orbital index.
- `molecule.xyz`: molecular geometry in Angstrom.
- `index.html`: an orbital viewer with selection and rotation controls.
- `manifest.json`: grid settings, exported orbitals, original orbital norms, coefficient scale factors, and the inter-orbital overlap matrix used for display.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_generate_orbitals -v
```

Five tests checked grid agreement with PySCF's `cubegen.orbital` within Cube-output rounding precision, selected-orbital ordering, individual normalization, preservation of nonorthogonality, invalid-input detection, and local-library selection.
Isosurface display was also checked in a browser using an H3+ result.

## Active-space results

Active SCGVB orbitals are displayed first, followed by each frozen inactive-space orbital.
Inactive orbitals are labeled "inactive, doubly occupied" in the HTML selector, and their roles are stored in the manifest.
Loading checks electron counts, closed-shell orthonormality, and active/inactive orthogonality.
Legacy full-space JSON is also supported. See [ACTIVE_SPACE.md](ACTIVE_SPACE.md) for details.
