# JAX backend

The NumPy implementation with complex-step differentiation remains in `scgvb/energy/numpy_backend.py`.
`scgvb/energy/jax_backend.py` subclasses it, replacing energy evaluation and gradients with double-precision JAX JIT compilation and automatic differentiation.
The augmented Lagrangian method, normalized-energy fallback for stagnation, and convergence criteria are shared with the existing implementation.

```python
from scgvb.energy.jax_backend import SCWF_Energy

scwf = SCWF_Energy(mol, device='cpu')  # Use 'gpu' on NVIDIA systems
energy, C, Cs = scwf.augmentedLagrangian(C, Cs)
```

Use `from scgvb.energy.numpy_backend import SCWF_Energy` for the original backend.
`calc_diatomic_mol.py` selects between `--backend numpy` (default) and `--backend jax`.
Omitting JAX `--device` uses the default JAX device.

## Execution scope and precision

- Module import sets `jax_enable_x64=True`; integrals and coefficients use float64.
- PySCF integral generation and reference FCI, and SciPy optimizer control, run on the CPU.
- The JAX backend transfers integrals and permutation data to the device when an instance is created and retains them there. Recreate the instance when the molecule or integrals change.
- Energy and gradient evaluations are JIT-compiled, with automatic differentiation for all variables. Objective and gradient evaluations at the same point share a single calculation.
- Coefficients and results are transferred between SciPy and JAX; the complete optimization does not run entirely on the GPU.
- This backend is for real orbitals. Use `target_gradient` / `rayleigh_gradient` instead of complex-step inputs. Physical complex orbitals are not implemented.
- JIT recompiles when array shapes change; a distance scan with unchanged shapes can reuse compiled functions.
- The default permutation method still stores `(N!)²` configuration pairs. See `ACTIVE_SPACE.md` for determinant evaluation and active spaces with frozen closed shells.

## Validation and timing

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
    tests.test_jax_energy tests.test_physical_energy tests.test_optimizer \
    tests.test_energy_regression tests.test_spin_function -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scgvb.benchmarks.jax --device cpu
```

JAX tests are skipped if JAX is not installed. The original backend does not import JAX.
Tests compare fixed-parameter energies and gradients for H₂ singlets and triplets, HeH⁺, and LiH.
They also check zero overlap factors, multiple spin coefficients, and cache updates.
Optimized energies and norms for H₂, HeH⁺, and LiH are compared with an independent determinant expansion.

On 2026-09-18, the current LiH script completed the 10 Å calculation and all 21 distance points from 0.5 to 2.5 Å on an Apple M4 CPU with Python 3.12.2 and JAX 0.11.2; plot output was checked.
The JAX energy at 10 Å was `-7.782107830840677 Eh`.
Different arithmetic ordering means that optimization trajectories and local solutions are not guaranteed to agree bit for bit.

Example timings for energy plus gradient evaluation in LiH / STO-3G / 1.6 Å (averages over 20 runs):

| Evaluation | Time |
|---|---:|
| First JAX call, including compilation | 0.163 s |
| NumPy plus complex-step differentiation | 12.368 ms |
| JAX automatic differentiation after compilation | 0.266 ms |

This evaluation is about 46× faster; that is not a speedup for the complete calculation.
Measurements clear the same-point cache before every run and include host retrieval of results, ensuring synchronization on GPUs as well.
NVIDIA GPU timings were not measured on this Mac.

## Bounded determinant spaces: stable expectation values

The optional `--energy-evaluation determinant` (NumPy and JAX) evaluates the same
SCWF through determinant amplitudes in an
orthonormal AO basis (`scgvb/energy/stable_jax.py`). These amplitudes are derived from
nonorthogonal orbital coefficients and spin coefficients; they are not additional
optimization variables. No Full CI eigenvalue problem is solved. The original
nonorthogonal orbitals remain the output. Permutation evaluation remains the default.
The optional evaluator supports at most six electrons in each spin sector and
at most 2,000,000 entries in each determinant, compound-minor, primitive-spin
coefficient, or transformed two-electron integral array. It supports 12e/12o singlets. Polynomial compound matrices,
checkpointed reverse mode, tiled Hamiltonian application and an analytic energy
VJP replace factorial minor expansion and dense transition tensors.
See [HPC_README.md](HPC_README.md) for the single-GPU distribution and benchmarks.

The norm is a sum of squared amplitudes, which avoids cancellation between
positive and negative permutation contributions. Polynomial minors also allow
derivatives at singular subblocks. This reduces one source of numerical error,
but does not guarantee convergence near a vanishing wavefunction or a singular
parameterization. `--verbose` and result metadata identify the evaluator.
The augmented Lagrangian algorithm and NumPy backend remain available.

This is an experimental comparison option, not a demonstrated Li2 convergence fix.
Initialization no longer builds permutation data in determinant mode.
Fixed inactive orbitals reduce the working electron count and orbital dimension;
see [ACTIVE_SPACE.md](ACTIVE_SPACE.md).
Python API: `SCWF_Energy(mol, compute_fci=False, energy_evaluation="determinant")`.

## Optional joint orbital/spin correction

`--joint-correction` (TOML `[optimizer] joint_correction = true`) adds one bounded
joint correction after each complete alternating cycle. It requires determinant
energy evaluation, `method = "alternating"`, and `spin_solver = "projected"`.
It is disabled by default; `--no-joint-correction` preserves previous behavior.
Existing simultaneous, BFGS, wavefunction and single-orbital methods remain available.

`scgvb/optimization/coupled.py` selects at most six tangent directions from orbital
and spin gradients, the projected spin response, and an earlier accepted joint
step. It forms a small physical wavefunction tangent Gram matrix, including
orbital/spin cross terms. A central difference of retracted physical gradients
approximates the reduced energy Hessian: at most twelve perturbed gradient
calls, without constructing the full parameter Hessian. The JAX metric uses
JVPs of derived normalized determinant amplitudes, not independent CI variables.
NumPy uses complex-step derivatives of those amplitudes.

Metric whitening, a positive curvature shift, and a physical trust radius
produce a descent proposal. Actual energy decrease, the unchanged cycle norm
floor, orbital displacement <= 0.1, and wavefunction distance <= 0.05 are checked
before accepting. An inaccurate curvature estimate or failed line search retains
the incoming state. The full normalized physical gradient <= 1e-6 and the existing
normalization tolerance remain required for convergence. Rank in the small
correction space is not a convergence criterion.

This is a finite-difference subspace Newton approximation, not the complete
orbital/spin Newton solve. Timing, accepted corrections, metric spectrum,
curvature spectrum, finite-difference asymmetry, trial rejections and before/after
states are recorded in the optional optimization diagnostics. Fe(CO)5 performance
and convergence must be assessed by the controlled comparison, not inferred
from the small-system validation.

Fixed-orbital spin initialization is independently selectable with
`--initial-spin-guess eigen` and `--initial-spin-overlap-cutoff` (default 1e-10).
It applies before every chosen optimizer and before inactive-orbital relaxation.
The default `uniform` preserves legacy initialization and saved restart Cs.
Explicit `eigen` also re-solves Cs on restored orbitals. Determinant evaluation
uses the centered projected solver; permutation evaluation uses spin matrices.
The initialization and optimizer spin cutoffs are independent.

`--joint-correction --joint-correction-start 21` uses ordinary alternating
updates for sweeps 1–20 and adds the existing joint correction at the end of
sweep 21 and every subsequent sweep. This preserves the current state at
switching. The default start is 1. Numbering is local to each optimizer call,
including restarts and inner inactive-orbital solves. Physical convergence in
warmup ends the solve without waiting for the correction stage. Bounded warmup
can continue across an energy plateau until the scheduled stage; the five-cycle
stall counter restarts at the switch. Energy increases, iteration limits and
full gradient/normalization tests remain enforced. This schedule is not yet
validated for Fe(CO)5 convergence; see `INPUT_FILE.md` for details.
