# SCGVB with 12 pi electrons: deployment on a supercomputer

This distribution targets **one process, one GPU, FP64**. It retains full SCGVB and handles frozen closed shells and active spaces with the same evaluators.
It does not automatically distribute work across GPUs or nodes.
A planar 12-pi-electron polyene converged on rikyu's GB200 (job 125026; orbital optimization took 2 minutes 46 seconds).
Nonplanar pi-space selection using local normals has CPU energy/gradient validation, but optimization convergence and GPU execution of the included biphenyl example remain unverified.
Measure performance and peak memory for each target system.

## 1. Files to transfer

Transfer `dist/scgvb-12pi.tar.gz` and `dist/scgvb-12pi.tar.gz.sha256`.
The archive includes source, CLI entry points, full-space support, XYZ and TOML examples, execution scripts, documentation, and dependency lists, including geometry-optimization support.
Tests, reference data, CPU validation reports, Mac virtual environments, caches, large Cube files, running-calculation outputs, and private settings are excluded.
Install binary dependencies separately.

```bash
sha256sum -c scgvb-12pi.tar.gz.sha256
tar -xzf scgvb-12pi.tar.gz
cd scgvb-12pi
python hpc/verify_manifest.py
```

Run `python build_hpc_bundle.py` to redistribute changes.
MANIFEST.json records the SHA256 of every file at distribution time.

## 2. Python environment for GB200

Grace CPUs use Linux aarch64: **do not copy the Mac environment**.
First check facility rules for Python/CUDA modules and containers.
The following example creates an independent environment with Python 3.12 and pip.
Prefer facility-provided JAX when available, record its version, and run validation tests supplied separately.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cuda13.txt
```

This requirements list pins CUDA 13 packages to the development JAX version, 0.11.2.
GPU dependency installation itself has not been validated on the development machine.
The official JAX instructions provide Linux aarch64/x86_64 GPU wheels; CUDA 13 requires Linux NVIDIA driver 580 or newer. Select an environment supported by the facility.
For a facility environment such as CUDA 12, confirm GB200 compatibility with administrators and test the adjusted dependencies.
These scripts do not modify drivers.
[Official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)

Use PySCF from pip or conda-forge ARM64 packages. Environments without a wheel may require a source build; facility-provided PySCF is another option.
[Official PySCF installation instructions](https://pyscf.org/user/install.html)

For offline compute systems, download dependency wheels on a connected Linux aarch64 system with the same Python version, then transfer them together.
Running `pip download` on a login node with a different CPU architecture can retrieve incompatible wheels.

```bash
# Run in the same Linux aarch64/Python environment as the target
python -m pip download --only-binary=:all: -r requirements-gpu-cuda13.txt -d wheelhouse
# After transferring wheelhouse, run in the target environment
python -m pip install --no-index --find-links wheelhouse -r requirements-gpu-cuda13.txt
```

Use `requirements-cpu.txt` for validation environments without a GPU.

## 3. Checks on an allocated compute node

Allocate one GPU first. Respect scheduler-controlled GPU visibility; do not override `CUDA_VISIBLE_DEVICES` to use unallocated GPUs.
Match CPU thread counts to the allocation. The following check uses one thread.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python hpc/check_environment.py --device gpu
```

The check reports device name, architecture, package versions, and an FP64 automatic-differentiation test.
If no GPU is available, it exits with code 1 rather than silently falling back to the CPU.
Run it on the allocated GPU compute node, not the login node.

## 4. Validation, benchmarking, and calculations

```bash
# Short GPU validation (compare 12e energy and directional derivatives with independent NumPy/PySCF)
# Requires separately supplied local tests, which are excluded from the archive
bash hpc/run_12pi.sh verify

# Benchmark synthetic 12e/12o and a molecular 12-pi-electron polyene
bash hpc/run_12pi.sh benchmark

# SCGVB optimization of the 12-pi-electron polyene
bash hpc/run_12pi.sh optimize
```

Each run creates `results/12pi-<timestamp>-<PID>/`; override the destination with `SCGVB_OUTPUT_DIR`.
It saves `environment.json`, `pip-freeze.txt`, logs, and result JSON.
The example polyene is an idealized planar C12H14 geometry, not a geometry-optimized molecule.
Energy/gradient evaluation was validated on the development machine; its 12-pi-electron orbital-optimization convergence was not established there.
Optimization needs many evaluations; estimate job duration from benchmark timings.

The full regression suite can also be run when local tests are supplied separately (mostly CPU tests, with some JAX tests depending on the environment):

```bash
SCGVB_RUN_12E=1 SCGVB_TEST_DEVICE=gpu python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

`hpc/slurm_1gpu.example.sh` is a generic template for Slurm facilities.
rikyu-specific batches (`hpc/rikyu*`, `hpc/*.sbatch`) remain local and are excluded from GitHub and distribution archives.
Adapt the account, partition, GPU request syntax, environment activation, time, and memory to the facility.
The actual scheduler has not been checked, so jobs are not submitted automatically.
Even with four allocated GPUs, this implementation uses only one. Run independent initial guesses or geometries as separate processes, each allocated one GPU by the scheduler.

### Other pi systems

```bash
python calc_molecule.py --xyz examples/COT.xyz --active-space pi --pi-electrons 8 \
  --energy-evaluation determinant --backend jax --device gpu --initial-guess hf \
  --output cot_pi.json
```

Automatic pi selection targets hydrocarbons. It defaults to planar molecules; add `--pi-normal-mode local` for nonplanar sp2 systems.
See the biphenyl example in `ACTIVE_SPACE.md`.
Electron count, spin, and size limits are checked.
Pi subspaces include occupied and virtual orbitals; sigma and core electrons form frozen closed shells.
See `ACTIVE_SPACE.md` for selection details and manual MO options.

## 5. Evaluator changes

- Determinant-expansion minors use Laplace recursion from order k−1 to k. No enumeration of k! permutations or inverses is needed; singular submatrices remain polynomially differentiable. JAX checkpoints recompute each order to reduce reverse-mode storage.
- The symmetric orbital-pair space of chemists' notation ERIs is diagonalized, expressing the Hamiltonian as a sum of squared one-body operators. Eigenvalues are not truncated, so no additional theoretical approximation is introduced; ordinary rounding errors remain.
- One-body actions use gather/contraction tiles of at most 32 rows, avoiding an `n_orbital**2 × n_det_alpha × n_det_beta` intermediate.
- For a fixed real symmetric Hamiltonian, the custom VJP of `(c.T H c, c.T c)` is `2 * (energy_cotangent * Hc + norm_cotangent * c)`. It returns gradients from c and Hc without storing Hamiltonian internals for reverse mode, then differentiates through nonorthogonal orbitals and spin coefficients.
- Integrals, operators, and coupling tables are transferred to the selected device at initialization. SciPy optimization runs on the CPU; each evaluation exchanges parameters, energies, and gradients rather than large expansion arrays.

Closed shells, the active-space basis, and integrals remain fixed during an electronic optimization.
The evaluator provides first derivatives for real orbitals and spin coefficients, not nuclear-coordinate derivatives or simultaneous optimization of inactive orbitals.
The default `standard` profile allows at most six electrons of each spin and two million elements each in determinant coefficients, minor arrays of every order, primitive spin coefficients, and transformed two-electron integrals.
These bounds are not checks of free GPU memory or guarantees of peak memory usage.

## 6. Interpreting benchmarks

`calc_molecule.py --optimize-core` also relaxes inactive orbitals while preserving double occupation.
This differs from the frozen-core 12-pi-electron benchmark. Outer integral transformations, density matrices, and orbital rotations currently run on the CPU; the requested GPU evaluates inner SCGVB calculations.
The evaluator is rebuilt each macroiteration. Performance and convergence of 12-pi-electron relaxation have not been validated by the frozen-core benchmark.
See `ACTIVE_SPACE.md` for settings and small-molecule examples.

`scgvb/benchmarks/active_space.py` evaluates energy and gradients at fixed parameters without optimization.
It bypasses the final Python evaluation cache and waits for GPU completion for each timing.

- `setup_seconds`: initialization, including HF, integrals, and evaluator preparation.
- `compile_seconds`: JAX/XLA compilation.
- `first_execution_seconds`: first execution after compilation.
- `warm_execution_seconds`: individual warm-run durations.
- `compiled_memory_analysis`: XLA buffer estimates for the compiled execution, not total peak device usage.
- `device_allocator_memory_stats`: allocator statistics when provided by the backend; normally null on the CPU. Reserved GPU memory differs from actual usage.
- `process_peak_rss_bytes`: peak host-process RSS, not GPU memory.
- `verification`: energy and directional-derivative errors against independent NumPy/PySCF evaluation.

CPU validation results are stored locally in `hpc/validation/` and excluded from GitHub and distribution archives.
Estimated compilation temporary storage was about 389 MiB for 12e/12o; GPU compiler choices and temporary storage require separate measurement.
No GB200 speed estimates extrapolated from these measurements are reported.

## Three, four, and six active electrons: permutation vs determinant performance

```bash
python -m scgvb.benchmarks.evaluators --case all --basis cc-pvdz --device gpu \
  --repeats 30 --output results/evaluator-comparison
```

Pi active spaces for allyl (3e, doublet), cyclobutadiene (4e, singlet), and benzene (6e, singlet) are constructed from the same HF reference.
For each molecule, identical perturbed-HF orbitals and spin coefficients are saved to `fixture.npz` and passed to separate evaluator subprocesses.
Core and active spaces remain fixed; no orbital optimization is performed.

Measurements separate initialization, the first energy/norm call, the subsequent first Rayleigh-energy/gradient call, and 30 calls following three warmups.
First calls include JIT and are not pure compilation timings. Median, minimum, maximum, and all samples are saved.
Argument transfer from the host and result retrieval are included; conversion to NumPy synchronizes GPU work.
Evaluation caches are disabled for every call. The compared gradient is that of `E_active=E_numerator/norm`, independent of overall wavefunction normalization and constant energy terms.

`summary.json` records energy, norm, and gradient agreement between evaluators and timing ratios.
Peak CPU RSS and JAX device memory statistics are also recorded for each subprocess.
Memory peaks describe the whole process, including initialization and compilation, not incremental cost per evaluation.
Results and logs are saved in molecule-specific folders; existing output directories are not overwritten.

## Comparing joint corrections from a saved state

`hpc/probe_joint_correction.py` compares update widths, direction counts, and direction-admission thresholds for the existing six-direction joint correction from the same saved singlet state, without intervening alternating optimization.
`hpc/probe_adaptive_joint.py` compares the existing method against one that adds up to four spin-residual directions from its candidate and resolves from the original state with at most ten directions.
Both read fixed integrals (`S/h1/eri/ecore`) from `solve-*.npz` and saved `C/Cs`.
They perform no HF initialization, active-space reconstruction, or separate orbital/spin updates.
Fixed-orbital spin solving inside direction construction is used only to generate candidate correction directions.

```bash
python hpc/probe_adaptive_joint.py --problem /path/to/solve-0001.npz \
  --state /path/to/joint_before.npz --problem-sha256 <SHA256> --state-sha256 <SHA256> \
  --expected-energy=<total-energy-hartree> --norm-floor=<fixed-unit-column-norm-floor> \
  --device gpu --steps 10 --output /path/to/new-diagnostic-directory
```

Run on a compute node. Input hashes and starting energy are checked.
The first six directions are saved in common; matching metric-array shapes are used to verify agreement between the first existing correction and the initial adaptive candidate.
Comparisons use trust radii 0.02 and 0.001, with at most ten corrections per method.
Adaptive correction preserves the same norm floor and trust radius and reevaluates reduced metric/curvature at the original state.
It does not move to a candidate and stack a further correction there.

In addition to the existing actual-energy decrease, norm floor, and orbital-displacement/wavefunction-distance conditions, candidates require the normalized full physical-gradient L2 norm to increase by no more than 1e-8 and the spin-gradient L2 norm to remain at most `max(1e-6, 2 * incoming_spin_gradient_l2)`.
These are experimental candidate-acceptance conditions, not changes to the existing convergence criteria of maximum physical gradient 1e-6 and normalization error 1e-8.
If no candidate satisfies them, the original state is retained and the process stops.
Additional directions are projected onto the tangent to the original Cs and orthogonalized twice against the existing basis.
Duplicate directions with relative independence at most 1e-10 are not added. The relative metric cutoff remains 1e-12.

Each addition stage saves its state, gradients, reduced metric/curvature, residuals, and acceptance decision.
Read `diagnostic_completed` and `scientific_convergence` in `summary.json` separately.
The low-level `scgvb.optimization.coupled.adaptive_correction` API is an explicitly invoked experimental addition; normal `joint_correction`, alternating optimization, and start-point defaults are unchanged.

## Joint correction with orbital-only damping and spin response

`scgvb.optimization.spin_response.response_correction` is a separate experimental low-level API.
The existing `correction` / `adaptive_correction` and normal alternating-optimization defaults remain available.
It derives `ds = -Hss^{-1} gs - Hss^{-1} Hso do` from reduced-Hessian blocks and substitutes this into the orbital equation.
A level shift is applied only to the orbital Schur complement; the spin level shift is zero.
The spin metric is resolved at relative cutoff 1e-12. Rank deficiency or nonpositive spin curvature stops without applying the correction.
The spin-curvature admission threshold is `1e-12 * max(1, max(abs(curvature)))`.
The method assumes proximity to a fixed-orbital spin-stationary point.

An earlier mixed joint displacement is split by SVD into the spaces spanned by its orbital and spin components.
Singular values above `max(1e-10, 1e-10 * largest_singular_value)` are retained for each component.
Splitting can increase the initial direction count. At most four post-correction spin-residual directions are added, with at most ten directions total.
Every augmented model is solved relative to the original pre-correction state.

The trust radius limits the physical distance of orbital changes and their induced spin response.
The spin correction present even for `do=0` is excluded from this radius.
Every candidate must satisfy actual total-wavefunction distance 0.05, each orbital's S-distance 0.1, actual energy decrease, and the fixed norm floor.
Full physical-gradient L2 and spin-gradient L2 acceptance conditions match the adaptive method above.
If augmented directions still fail acceptance, only orbital coordinates are halved and the spin response is recomputed from the equation each time, for at most 30 candidates.
The full joint displacement is not halved.
Eliminating the local-model spin residual does not guarantee zero full spin residual after a finite update.
Candidates, local models, predicted spin residuals, and orbital-halving histories are saved for inspection.

```bash
python hpc/probe_spin_response.py --problem /path/to/solve-0001.npz \
  --state /path/to/joint_before.npz --problem-sha256 <SHA256> --state-sha256 <SHA256> \
  --expected-energy=<total-energy-hartree> --norm-floor=<fixed-unit-column-norm-floor> \
  --device gpu --steps 10 --output /path/to/new-diagnostic-directory
```

From the same saved state and initial six directions, compare the common-damping adaptive method and the spin-response method with trust radii 0.02 / 0.001.
The new radius is measured with the post-response metric above, so its definition differs from the old full-joint-displacement radius.
No HF reinitialization or alternating optimization is performed.
The distinction between `diagnostic_completed` and `scientific_convergence`, and convergence criteria, are the same as above.
Internal gradient-evaluation counts exclude additional evaluations by diagnostic-output callbacks.

## Running 14-electron/14-orbital calculations

Explicitly select `--determinant-size-profile large14` in the standard code.
The existing `standard` profile remains the default.
See [Determinant-size profiles: 14-electron support](ACTIVE_SPACE.md#determinant-size-profiles-14-electron-support) for TOML, Python API, restart, core-relaxation, and density-analysis settings and limits.

```bash
python -m scgvb.benchmarks.active_space --device gpu --synthetic-orbitals 14 \
  --determinant-size-profile large14 --verify --output benchmark14.json
```

Phenazine 14e/14o with cc-pVDZ converged with frozen inactive orbitals (job 165725) and with all 40 inactive orbitals relaxed (job 166080).
Results are stored locally in `runs/rikyu_phenazine14_20261004/` and `runs/rikyu_phenazine14_relaxed_20261004/`.
