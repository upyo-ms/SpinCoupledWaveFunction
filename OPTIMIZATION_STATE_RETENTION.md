# Retained optimization candidates

Simultaneous optimization separately retains the lowest-energy point satisfying normalization and the lowest-energy stationary candidate satisfying both normalization and physical-gradient criteria.

`SCWF_Energy.retained_optimization_states` contains:

- `lowest_feasible`: the lowest-energy normalizable point, without a guarantee of gradient convergence.
- `stationary_candidate`: a candidate satisfying normalization and the physical-gradient threshold `1e-6` with the current evaluator. This does not establish a local or global minimum.

Each candidate contains a copy of the raw parameter vector `x`, `active_energy_hartree`, and `norm`. Stationary candidates also contain `physical_gradient_max_abs`. A missing candidate is `None`. The energy is the active-Hamiltonian Rayleigh quotient, excluding nuclear repulsion and inactive-orbital constants; it is not the molecular total energy.

`x` is `[C.ravel(), Cs]` in the optimization basis. During inactive-orbital relaxation, consult the diagnostic snapshot for the corresponding inner electronic solve to recover its basis and integrals. This information is not itself restart JSON.

When optimization raises an exception, the same information is available in its `retained_optimization_states` attribute. Exception types and existing acceptance conditions are unchanged. A retained higher-energy stationary candidate is not automatically returned as the converged energy.

With diagnostic snapshots enabled, stationary candidates are written as `best_stationary_candidate` and retained candidates on failure as `failure_stationary_candidate`. The existing `failure` event continues to contain the lowest-energy point.

Retention uses energy and gradient checks that have already been computed. It changes neither search directions, restart states, gradient thresholds, nor energy-acceptance conditions. For representations that are ill-conditioned in double precision, evaluator errors also affect candidate classification; retention is distinct from high-precision reevaluation.

Local thiophene experiments with compensated arithmetic and candidate-dependent restart choices are stored in `runs/local_thiophene_state_retention_20261020/`. That experimental evaluator targets a saved five-orbital/six-electron model and has not replaced the standard evaluator.
