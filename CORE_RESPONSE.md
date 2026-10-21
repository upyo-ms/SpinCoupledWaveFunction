# Optional SCGVB response during inactive-orbital relaxation

**The common option is now `second_order`.** It enables internal SCGVB
corrections even when `optimize_core=false`, and also the outer response when
`optimize_core=true`. `core_response` remains an input/API/CLI alias with the
expanded behavior. See [the current interface and algorithm](SECOND_ORDER.md).
The description and benchmark results below document the earlier outer-only
implementation; they are historical comparisons, not current option limits.

The existing two-step orbital optimization remains the default. To include
the response of the SCGVB orbital and spin-coupling coefficients in the outer
orbital Hessian, explicitly enable:

```toml
[active_space]
optimize_core = true

[optimizer]
core_response = true
```

CLI equivalents are `--optimize-core --core-response`; `--no-core-response`
overrides an input file. The Python API accepts
`calculate(..., optimize_core=True, core_response=True)`. The flag requires
nonempty inactive orbitals and `optimize_core=true`. `--check` validates the
combination and derivative allocation bound. A complete small example is
[lih_core_response.toml](examples/inputs/lih_core_response.toml).

This is an experimental CPU response calculation. It works with the NumPy or
JAX inner solver, and does not require JAX itself. The inner solver still varies
only SCGVB orbital coefficients C and spin-coupling coefficients Cs. Slater
determinant expansion amplitudes are derived quantities; no FCI minimization or
independent determinant-coefficient response is used.

## What the option changes

At each converged inner state, the code differentiates the physical Rayleigh
energy, not its normalization penalty. Polynomial exterior products yield first
and second derivatives of the amplitudes. The second-amplitude contraction with
the full Hilbert-space residual is retained, so this is not just a projected
linear-wavefunction Hessian. Individual zero minors require no matrix inverse.

The Jacobian of the normalized physical state supplies its tangent metric.
SVD removes null directions and retains squared singular values above a relative
cutoff of 1e-10, with an absolute singular-value floor of 1e-12. This also removes
coefficient scale freedoms. The retained tangent coordinates have unit physical
metric. The cutoff can discard weakly represented physical directions; this is
a regularized local response, not an exact inverse on an arbitrarily singular
SCGVB parameterization.

Let A be the physical internal Hessian in these coordinates and B the mixed
derivative with external orbital rotations. Differential active RDMs provide B.
Inactive-only constants are omitted analytically in this derivative rather than
subtracted from large total orbital gradients. The orbital Hessian action gets
the correction `-B.T @ inverse(A) @ B`. Negative or small internal eigenvalues
are shifted together until the minimum is at least 1e-6 Eh. Such a shift makes
the correction an approximate, stabilized response. Its size is recorded.

PySCF's packed orbital gradient and Hessian are half the physical derivatives;
the implementation applies that factor to the correction and its diagonal.
The correction is applied as a matrix-vector product; a dense outer orbital
Hessian is not formed. Existing augmented-Hessian step limits and trust checks
remain active. Internal coefficients are subsequently reoptimized by the
existing inner SCGVB solver, using the same convergence thresholds.

This remains a two-step method with a response-corrected orbital Hessian. It is
not a full simultaneous Newton solver for an unconverged internal state, and
does not provide recovery when the initial inner solve fails before the first
orbital update. It also does not replace the energy/gradient evaluator with the
earlier compensated-arithmetic prototype.

## Limits and result records

Exact second amplitude derivatives have an additional allocation limit:
each derivative block is at most 2,000,000 scalar entries, including value,
Jacobian and Hessian entries. Temporary arrays make peak memory larger than
one block. This bound is separate from the energy evaluator's size profile;
larger active spaces are rejected explicitly. The 5-orbital/6-electron model is
within the bound. Strong amplitude cancellation (ratio above 1e8), disagreement
with the inner solver's state, or nonfinite curvature is rejected rather than
silently switching methods. FP64 roundoff remains a limitation.

The result records `calculation.core_response`,
`core_optimization.core_response`, `response_method`, and `response_history`.
The history includes tangent rank, metric cutoff, minimum internal curvature,
curvature shift, mixed derivative norm, cancellation ratio, internal gradient,
and timing. Errors raised during the macro solve carry `core_response_history`
as a Python exception attribute. The final physical internal gradient, orbital
gradient, normalization and independent integral reconstruction checks are
unchanged. Geometry calculations pass the option to each electronic solve, and
the option participates in the existing geometry checkpoint/cache signature.

## Initial validation

Analytic amplitude first/second derivatives were checked at zero individual
minors, and the physical Hessian against finite differences of an independent
gradient. This includes the 5-orbital/6-electron SCGVB model with 28 independent
physical tangent directions. For LiH, both orders of the mixed derivative were
checked with fresh rotated molecular integrals.

Small molecular comparisons with the same starts and unchanged thresholds:

| System / active model | Default macro iterations | Response macro iterations |
| --- | ---: | ---: |
| LiH, STO-3G, 2e/2o | 22 | 16 |
| LiH, cc-pVDZ, 2e/2o | 25 | 18 |
| H2O, STO-3G, 4e/3o | 15 | 13 |

Both methods converged in these cases; their energies agreed within 1e-12 Eh.
These are small validation examples, not evidence of universal convergence or
speedup. With the current thiophene cc-pVDZ fixed-space restart and full inactive
relaxation, both methods stopped in the inner SCGVB solve following the first
outer orbital update. The response option did not resolve that failure. It is
therefore an experimental comparison option, not a verified thiophene recovery.
