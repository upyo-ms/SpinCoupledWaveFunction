# Optional second-order SCGVB corrections

`second_order` controls internal and outer corrections with one option. Its
default is `false`. It never changes the nuclear coordinates or the active
electron count.

```toml
[active_space]
optimize_core = false

[optimizer]
second_order = true
```

With `optimize_core=false`, safeguarded Newton corrections update the SCGVB orbital
coefficients C and spin-coupling coefficients Cs within the selected active
basis. With `optimize_core=true`, the same internal corrections run at every
macro iteration, and their response corrects the outer orbital Hessian too.
Inactive occupations remain two; AO F orbitals remain frozen.

CLI: `--second-order` / `--no-second-order`. Python:
`calculate(..., second_order=True, optimize_core=False)` or
`calculate(..., second_order=True, optimize_core=True)`.

`core_response` and `--[no-]core-response` remain aliases for the common option.
This preserves old input spelling, **not the old outer-only trajectory**:
enabled old inputs now apply internal corrections as well. Contradictory
explicit API aliases or TOML keys are errors. A CLI override wins over file
settings even when it uses the other spelling. The default, disabled algorithm
has no second-derivative tensor allocations.

A small runnable example is [LiH](examples/inputs/lih_second_order.toml).

## JAX curvature option

The existing analytic implementation remains the default:
`second_order_backend = "analytic"`. To use JAX automatic differentiation:

Install JAX using `python -m pip install -r requirements-cpu.txt` for CPU use,
or `requirements-gpu-cuda13.txt` on a compatible NVIDIA system. The default
analytic curvature with NumPy energy evaluation does not require JAX.

```toml
[calculation]
backend = "jax"
device = "cpu"  # "gpu" with an installed, working GPU JAX backend
energy_evaluation = "determinant"

[optimizer]
second_order = true
second_order_backend = "jax"
```

CLI: `--second-order-backend jax`. Python:
`calculate(..., second_order=True, second_order_backend="jax", backend="jax", device="cpu")`.
The curvature choice is independent of the energy evaluator: with NumPy
energy evaluation, JAX curvature runs on CPU. Selecting `"jax"` alone does not
enable corrections, import JAX, or allocate its derivative arrays. It also
selects the curvature used by `second_order_fallback = true` upon activation.

JAX differentiates the full normalized Rayleigh energy with respect to C and
Cs, including their mixed derivatives, with `jax.hessian`. Derived determinant
amplitudes remain functions of C/Cs, never independent variables. The FP64
polynomial-minor path avoids QR rank decisions and inverse-minor derivatives;
it does not reuse the energy evaluator's first-derivative custom VJP. A constant
energy reference is subtracted before contraction to reduce cancellation.

The Hessian and normalized-wavefunction Jacobian run on the selected JAX
device. Integral transformation/factorization, physical-metric SVD, shifts,
step control, and the surrounding SciPy/PySCF solvers remain on CPU. In an outer
response, the internal curvature A uses the selected backend; mixed orbital
derivatives B and the outer action D retain their existing analytic PySCF
implementation. Compiled derivatives accept the current Hamiltonian as data
and are reused across internal/outer calls in a core solve. They do not freeze
the Hamiltonian at the initial orbitals.

This implementation forms the dense internal Hessian; it is not an HVP-only
iterative solver. The existing derivative-size limit remains conservative and
does not bound JAX compiler workspaces or peak device memory. GPU execution
does not guarantee a speedup; compilation and transfers can dominate small
systems. GPU hardware performance has not yet been validated locally.

Both backends use the same redundancy cutoff, safeguarded step, independent
gradient agreement check and convergence gates. No silent switch to analytic
curvature occurs when JAX is selected. An unreliable correction stage can end
and let the ordinary optimizer continue, as with the analytic option.
Result records include `calculation.second_order_backend`, and correction and
outer-response histories record the selected backend and curvature device.
A runnable example is [LiH with JAX curvature](examples/inputs/lih_second_order_jax.toml).

```sh
python calc_molecule.py examples/inputs/lih_second_order_jax.toml
python calc_molecule.py examples/inputs/lih_second_order.toml
python calc_molecule.py examples/inputs/lih_second_order.toml --optimize-core
python calc_molecule.py examples/inputs/vinylthiol_pi_ccpvdz.toml --second-order
python calc_molecule.py examples/inputs/vinylthiol_pi_ccpvdz.toml --second-order --optimize-core
```

## Automatic retry after nonconvergence

Use `second_order_fallback` to start with the existing optimizer and switch only
after electronic nonconvergence. Its default is `false`.

```toml
[optimizer]
second_order = false
second_order_fallback = true
```

CLI: `--second-order-fallback` / `--no-second-order-fallback`.
Python: `calculate(..., second_order_fallback=True)`.
`second_order=true` takes precedence and enables corrections from the start;
there is then no extra automatic retry. A runnable input is
[LiH](examples/inputs/lih_second_order_fallback.toml).

The automatic path makes **one retry per electronic calculation**. Stalling,
iteration exhaustion, and failure to satisfy physical stationarity or
normalization are eligible. Input validation, unrelated runtime errors, and
integral-reconstruction inconsistencies are not converted into convergence
failures. Eligible errors use `ElectronicConvergenceError`, a `RuntimeError`
subclass. The same limits apply independently to each attempt; tolerances are
unchanged. If the second attempt fails, the calculation still fails and retains
both errors in the exception chain.

With fixed inactive orbitals, finite retained candidates and the incoming state
are reevaluated in the same Hamiltonian, and the lowest-energy normalized seed
is used. Alternating optimization also supplies its last physical iterate.
With `optimize_core=true`, both inner and outer failures can trigger the retry.
The rotated inactive/active basis is captured before each inner solve; AO
integrals are rebuilt in that basis before selecting a retained seed. Outer
iteration exhaustion uses the last returned basis and corresponding C/Cs.
The second attempt starts a fresh outer optimization with internal corrections
and outer response enabled. If a consistent stopped state cannot be rebuilt,
the original input core/active state is used and the reason is recorded.
AO F constraints and the active electron/spin space are preserved.

Successful primary attempts do not allocate second-derivative tensors. Unlike
explicit `second_order=true`, the automatic path checks the derivative size
limit only when activating the retry, so a larger first-order calculation can
still succeed. If retry tensors exceed the limit, the original convergence
error is reported with the unavailability reason. `--check` reports the requested
retry option without allocating tensors or attempting a solve.

Geometry optimization passes the policy to every electronic evaluation.
Each new geometry tries the ordinary method first; switching is local to that
evaluation and does not change later evaluations' requested policy. The two
attempts form one electronic evaluation for the geometry evaluation budget.
Evaluation history records successful or failed retry metadata.

`second_order_fallback` in result JSON records `enabled`, `triggered`, `status`,
`first_order_error`, and `restart_source`. `calculation.second_order_requested`
records the original explicit mode, and `calculation.second_order` records the
effective mode used for the accepted result. The input provenance retains both
requested settings. Failed retries carry `second_order_fallback_info` on the
Python exception; the CLI reports the retry status and the first failure.
Automatic switching provides another recovery attempt, not guaranteed convergence.

## Internal correction

All variations stay within the SCGVB parameterization `x = [C.ravel(), Cs]`.
Slater determinant amplitudes are derived exterior products, never independent
optimization variables. Hamiltonian actions evaluate this parameterized state;
no FCI optimizer is called. The default analytic derivatives run on CPU and
work with NumPy or JAX energy/gradient evaluation. The optional JAX curvature
backend described above computes its derivatives on the selected JAX device.

In the analytic backend, polynomial exterior products provide the amplitude, Jacobian and exact second
derivative, including derivatives of individual zero minors. The physical
Rayleigh Hessian includes the contraction of the second amplitude derivative
with the Hilbert-space residual. C/Cs mixed derivatives are retained. AO overlap
is accounted for by transforming amplitudes and integrals consistently to an
orthonormal basis.

The normalized-wavefunction Jacobian defines the metric. SVD removes null and
weak directions, including coefficient scale freedoms. The squared relative
singular-value cutoff is `1e-10`, with an absolute singular-value floor of
`1e-12`. The local Newton equation is solved in the retained physical tangent
space. Eigenvalue shifts keep its minimum curvature at least `1e-6 Eh` and
bound the proposed physical step. Initial wavefunction radius is `0.1`;
individual orbital changes in the AO metric and the spin-vector change are
limited to `0.2`. Backtracking verifies the actual state distance and the
selected evaluator's energy/full normalized gradient. Energy ties at roundoff
are accepted only with a gradient improvement. In alternating optimization the
existing cycle norm floor also applies.

These are **bounded corrections around the existing optimizer**, not a new
unbounded full-Newton solver. Each internal solve receives up to eight initial
Newton steps, then the requested simultaneous or alternating optimizer runs.
A stalled simultaneous BFGS refinement can receive up to sixteen additional
steps from each of its failed and incoming states. Alternating cycles receive
up to two corrections per sweep. The configured inner iteration limit also
bounds each correction stage. Existing BFGS/alternating success checks still
apply. A higher stationary result cannot override a lower retained feasible
energy beyond the existing `1e-10 Eh` gate.

Large cancellation (ratio above `1e8`), inconsistent gradients, or unreliable
curvature ends the correction stage and leaves the existing optimizer to
continue. Such a fallback is recorded. It is not reported as convergence or
precision recovery, and does not enable the earlier compensated-arithmetic
prototype.

## Outer response

When inactive orbitals are relaxed, let A denote the internal physical Hessian,
B its mixed derivative with the outer rotations, and D the outer Hessian. The
outer action receives `D - B.T inverse(A) B`, using shifted internal curvature.
Active differential RDMs provide B. Inactive-only constants are omitted
analytically rather than subtracted from large orbital gradients. PySCF's
half-gradient/half-Hessian convention is handled explicitly. The outer Hessian
is applied as an operator, not stored densely.

Outer AH limits, final orbital-gradient checks, normalization checks and
independent integral reconstruction are unchanged. This option does not
guarantee convergence, especially with nearly cancelling SCGVB representations
or a residual final outer gradient. The historical outer-only comparisons are
in [CORE_RESPONSE.md](CORE_RESPONSE.md).

## Limits and records

Derivative blocks are limited to `2,000,000` scalar entries, including their
value, first and second derivatives. Peak memory is larger than one block.
This is separate from energy-evaluator allocation limits. Unsupported sizes
are rejected before derivative tensors are created, including with `--check`.
The 5-dimensional, 6-electron thiophene model fits this limit.

Results store:

- `calculation.second_order`: the effective common option; it becomes true after
  a successful automatic switch. `second_order_requested` retains the original
  explicit choice and `second_order_fallback` the requested retry policy.
- `second_order_fallback`: whether a retry was triggered, its reason, seed
  source, and outcome.
- `second_order_optimization`: internal attempts, accepted steps, curvature
  shifts, gradient disagreement, rejection/fallback status and histories.
- `core_optimization.second_order`, `response_history`: common enablement and
  outer response diagnostics when inactive orbitals are relaxed.
- `calculation.core_response`: compatibility field indicating whether the
  **outer** response is active; it is false for fixed inactive orbitals.

Detailed diagnostics include `second_order_start` and accepted
`second_order_correction` snapshots with the matching saved Hamiltonian.
Internal failures carry `second_order_info` and retained candidates on the
Python exception. Existing outer failure response histories remain available.
Geometry optimization passes the option through to each electronic solve and
its existing checkpoint signature; enabling it does not request geometry
optimization.

## Verification

`tests/test_second_order.py` checks nonorthogonal AO Hessians against gradient
differences, both inactive settings, aliases and overrides, diagnostics,
alternating/JAX integration, retained-energy gates and allocation bounds.
`tests/test_core_response.py` also checks analytic second amplitude derivatives,
the internal Hessian, both orders of mixed orbital derivatives and the outer
factor of two. Molecular benchmarks are recorded in
`runs/local_second_order_20261021/` separately from the historical experiments.
`tests/test_second_order_fallback.py` checks real nonconvergence/recovery with
fixed and relaxed inactive orbitals, alternating and JAX solves, consistent
rotated-basis reconstruction, bounded failed retries, strict error eligibility,
deferred size limits, CLI/TOML overrides, and geometry evaluation records.
`tests/test_jax_second_order.py` compares AD and analytic Hessians (including
nonorthogonal AO overlap, zero minors, six active electrons and open shells),
checks Hessian actions against gradient differences, changed Hamiltonians,
both optimization methods and inactive settings, automatic retry, and option
parsing. CPU results are recorded in `runs/local_jax_second_order_20261022/`.
That verification includes a matched thiophene/STO-3G convergence, and a
furan/STO-3G case where analytic curvature converges but JAX curvature stalls.
The optional backend is therefore an experimental alternative, not a claim of
uniformly improved convergence.
