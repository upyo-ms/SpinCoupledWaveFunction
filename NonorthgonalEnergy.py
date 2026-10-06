import math
import warnings

import numpy as np
from pyscf import gto
from scipy.optimize import minimize

import GetMoleculeInfor
import Spin_Function as sf


class SCWF_Energy:
    def __init__(self, mol, compute_fci=True, energy_evaluation='permutation', active_space=None, determinant_size_profile='standard'):
        from DeterminantEnergy import determinant_size_profile as validate_size_profile
        validate_size_profile(determinant_size_profile)
        self.determinant_size_profile = determinant_size_profile
        self.mol = mol
        if energy_evaluation not in ('permutation', 'determinant'):
            raise ValueError('energy_evaluation must be permutation or determinant')
        self.energy_evaluation = energy_evaluation
        self.active_space = active_space
        self.inactive_energy = 0.
        # Calculating the electron arrangement and matrix representation of permutations
        # Sm : spin multiplicity
        self.Ne = mol.nelectron if active_space is None else active_space.electrons
        self.Sm = mol.spin + 1
        if energy_evaluation == 'determinant':
            from DeterminantEnergy import check_determinant_size
            n = mol.nao_nr() if active_space is None else active_space.basis.shape[1]
            check_determinant_size(n, self.Ne, mol.spin, profile=self.determinant_size_profile)
        if active_space is None:
            self.energy_nuc, self.Ne, self.Sm, self.S, self.Hcore, self.JK, self.e_fci = GetMoleculeInfor.GetMolecule(mol, compute_fci=compute_fci)
        else:
            if compute_fci:
                raise ValueError('Use compute_fci=False with an active space; full FCI is not a like-for-like reference')
            self.energy_nuc, self.e_fci = mol.energy_nuc(), None
            self.S = np.eye(active_space.basis.shape[1])
            self.Hcore, self.JK, self.inactive_energy = active_space.integrals(mol)
        # fs : the number of spin function
        # DN : matrix representations of permutations
        # IndexElectron: permutation
        if energy_evaluation == 'permutation':
            self.fs, self.DN, self.IndexElectron = sf.GenerateElecArrangement_SymRe(self.Ne, self.Sm)
        else:
            from Spin_Function.SpinFuncDege import spin_function_dim
            self.fs = spin_function_dim(self.Ne, self.Sm, 0)
        self._determinant_evaluator = None

    def orth2nonorth_twoIntegral(self, a, b, c, d, C):
        return np.einsum(
            'ijkl,i,j,k,l->', self.JK, C[:, a], C[:, b], C[:, c],
            C[:, d], optimize=True,
        )

    def transform_two_integrals(self, C):
        """Transform all orbital indices once per energy evaluation.

        Sequential contractions avoid a Python loop over AO indices and keep
        the intermediates to four dimensions. Do not cache across evaluations:
        the optimizer changes C on each call.
        """
        tmp = self.JK
        # Contract the leading AO index and append the new orbital index:
        # ijkl -> jkla -> klab -> labc -> abcd.
        for _ in range(4):
            tmp = np.tensordot(tmp, C, axes=(0, 0))
        return tmp

    def Normalization(self, C):
        S_tmp = (C.T)@self.S@C
        for i in range(self.Ne):
            C[:, i] = C[:, i]/np.sqrt(S_tmp[i, i])
        return C

    def calcEnergy(self, C, Cs):
        if getattr(self, 'energy_evaluation', 'permutation') == 'determinant':
            if self._determinant_evaluator is None:
                from DeterminantEnergy import make_evaluator
                self._determinant_evaluator = make_evaluator(self)
            return self._determinant_evaluator(C, Cs)
        # Normalize coefficients
#        C = self.Normalization(C)
        # The physical parameters are real. For complex-step differentiation,
        # use the analytic continuation of the real norm (no conjugation).
        # This branch does not implement complex-valued physical orbitals.
        spin_norm = np.sqrt(np.dot(Cs, Cs)) if np.iscomplexobj(Cs) else np.linalg.norm(Cs)
        Cs = Cs/spin_norm

        # Transform to non-orthogonal orbital
        S_non = (C.T)@self.S@C
        Hcore_non = (C.T)@self.Hcore@C
        JK_non_all = self.transform_two_integrals(C)

        # Each pair is the overlap of two signed, permuted spin functions.
        representations = np.asarray(self.DN)
        weights = np.einsum('i,pij,j->p', Cs, representations, Cs)
        indices = np.asarray(self.IndexElectron)
        left, right = indices[:, 0], indices[:, 1]
        factors = S_non[left, right]
        # The first N! entries fix the bra ordering to identity.
        count = math.factorial(self.Ne)
        Delta_s = np.sum(weights[:count] * np.prod(factors[:count], axis=1))
        H_one = np.sum(weights * Hcore_non[left[:, 0], right[:, 0]]
                       * np.prod(factors[:, 1:], axis=1)) / math.factorial(self.Ne - 1)
        # Chemists' ERI order pairs bra/ket for the SAME electron:
        # (left[0] right[0] | left[1] right[1]).
        G_two = np.sum(weights * JK_non_all[left[:, 0], right[:, 0],
                                           left[:, 1], right[:, 1]]
                       * np.prod(factors[:, 2:], axis=1)) / (2 * math.factorial(self.Ne - 2))
        return H_one + G_two, Delta_s

    def gen_target(self, mu, lam):
        def target(x):
            if not np.isfinite(x).all():
                raise FloatingPointError("Optimizer supplied non-finite coefficients")
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                C, Cs = np.split(x, [len(self.S)*self.Ne])
                C = C.reshape(len(self.S), self.Ne)
                Ee, Delta_s = self.calcEnergy(C, Cs)
                c = Delta_s - 1.0
                value = Ee + mu * c**2.0 + lam * c
            if not np.isfinite(value):
                raise FloatingPointError("Augmented Lagrangian is non-finite")
            return value
        return target

    def split_x(self, x):
        C, Cs = np.split(x, [len(self.S)*self.Ne])
        C = np.reshape(C, (len(self.S), self.Ne))
#        C = self.Normalization(C)
        # np.split returns views; never modify the optimizer's input vector.
        norm = np.linalg.norm(Cs)
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("Spin coefficients must have a finite nonzero norm")
        Cs = Cs / norm
        return C, Cs

    def target_gradient(self, x, mu, lam):
        """Differentiate the real objective without subtractive cancellation.

        A complex step evaluates Im(f(x + i*h*e_j))/h. Unlike forward
        differences this remains accurate as the penalty term grows.
        """
        target = self.gen_target(mu, lam)
        step = 1e-20
        gradient = np.empty(len(x))
        trial = np.asarray(x, dtype=complex).copy()
        for j in range(len(x)):
            trial[j] += 1j * step
            gradient[j] = target(trial).imag / step
            trial[j] = x[j]
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Objective gradient is non-finite")
        return gradient

    def delta_constraint(self, x):
        C, Cs = self.split_x(x)
        Ee, Delta_s = self.calcEnergy(C, Cs)
        return Delta_s - 1.0

    def normalized_candidate(self, x):
        """Remove orbital scale freedoms without changing the physical state."""
        C, Cs = self.split_x(x)
        orbital_norms = np.einsum('ia,ij,ja->a', C, self.S, C)
        if np.any(orbital_norms <= 0) or not np.isfinite(orbital_norms).all():
            raise FloatingPointError("Invalid orbital norm")
        C = C / np.sqrt(orbital_norms)
        _, norm = self.calcEnergy(C, Cs)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Wavefunction has non-positive norm")
        # Every orbital occurs once in every term of the wavefunction.
        # Multiplying all columns by t multiplies its norm by t**(2*Ne).
        C = C * norm ** (-1.0 / (2 * self.Ne))
        # Re-evaluate after scaling: cancellation in nonorthogonal overlaps
        # can leave a measurable floating-point residual. Keep the best scale.
        best_C = C.copy()
        best_error = abs(self.calcEnergy(C, Cs)[1] - 1.)
        for _ in range(3):
            if best_error < 1e-12:
                break
            _, norm = self.calcEnergy(C, Cs)
            if not np.isfinite(norm) or norm <= 0:
                break
            C = C * norm ** (-1.0 / (2 * self.Ne))
            error = abs(self.calcEnergy(C, Cs)[1] - 1.)
            if error < best_error:
                best_C, best_error = C.copy(), error
            if best_error < 1e-12:
                break
        C = best_C
        return np.r_[C.ravel(), Cs]

    def rayleigh_gradient(self, x):
        """Gradient of the physical energy Ee/Delta (nuclear term is constant)."""
        gradient = np.empty(len(x))
        for j in range(len(x)):
            trial = np.asarray(x, dtype=complex).copy()
            trial[j] += 1e-20j
            C = trial[:-self.fs].reshape(len(self.S), self.Ne)
            energy, norm = self.calcEnergy(C, trial[-self.fs:])
            gradient[j] = (energy / norm).imag / 1e-20
        return gradient

    def unit_orbital_wavefunction_norm(self, x):
        """Expose cancellations hidden by overall wavefunction rescaling."""
        C, Cs = self.split_x(x)
        norms = np.einsum('ia,ij,ja->a', C, self.S, C)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise FloatingPointError('Invalid orbital norm')
        return self.calcEnergy(C / np.sqrt(norms), Cs)[1]

    def spin_matrices(self, C):
        """Electronic Hamiltonian and overlap in the fixed-orbital spin basis."""
        stable = getattr(self, '_stable_energy', None)
        if stable is not None:
            import jax
            result = stable.spin_matrices(jax.device_put(np.asarray(C), self.device))
            return tuple(np.asarray(a) for a in result)
        if self.energy_evaluation == 'determinant':
            if self._determinant_evaluator is None:
                from DeterminantEnergy import make_evaluator
                self._determinant_evaluator = make_evaluator(self)
            return self._determinant_evaluator.spin_matrices(C)
        # Permutation weights are quadratic in the spin coefficients.
        indices = np.asarray(self.IndexElectron)
        left, right = indices[:, 0], indices[:, 1]
        overlap = C.T @ self.S @ C
        one = C.T @ self.Hcore @ C
        two = self.transform_two_integrals(C)
        factors = overlap[left, right]
        count = math.factorial(self.Ne)
        sspin = np.einsum('p,pij->ij', np.prod(factors[:count], axis=1),
                          np.asarray(self.DN)[:count])
        weights = (one[left[:, 0], right[:, 0]] * np.prod(factors[:, 1:], axis=1)
                   / math.factorial(self.Ne - 1))
        weights += (two[left[:, 0], right[:, 0], left[:, 1], right[:, 1]]
                    * np.prod(factors[:, 2:], axis=1) / (2 * math.factorial(self.Ne - 2)))
        hspin = np.einsum('p,pij->ij', weights, np.asarray(self.DN))
        return (hspin + hspin.T) * .5, (sspin + sspin.T) * .5

    def _wavefunction_evaluator(self):
        if self.energy_evaluation != 'determinant':
            raise ValueError('Wavefunction geometry requires determinant energy evaluation')
        if self._determinant_evaluator is None:
            from DeterminantEnergy import make_evaluator
            self._determinant_evaluator = make_evaluator(self)
        return self._determinant_evaluator

    def orbital_wavefunction_metric(self, C, Cs):
        return self._wavefunction_evaluator().orbital_metric(C, Cs)

    def wavefunction_overlap(self, C, Cs, other_C, other_Cs):
        return self._wavefunction_evaluator().wavefunction_overlap(C, Cs, other_C, other_Cs)

    def spin_overlap_matrix(self, C):
        return self._wavefunction_evaluator().overlap_matrix(C)

    def projected_spin_matrices(self, C, basis, reference):
        return self._wavefunction_evaluator().projected_matrices(C, basis, reference)

    def single_orbital_overlap(self, C, Cs, index):
        return self._wavefunction_evaluator().orbital_overlap(C, Cs, index)

    def projected_orbital_matrices(self, C, Cs, index, basis, reference):
        return self._wavefunction_evaluator().orbital_projected(C, Cs, index, basis, reference)

    def joint_wavefunction_metric(self, C, Cs, basis):
        """Small Gram matrix of normalized SCGVB tangents, including cross terms."""
        if self.energy_evaluation != 'determinant':
            raise ValueError('Joint metric requires determinant energy evaluation')
        from DeterminantEnergy import make_ci_builder
        build = make_ci_builder(self)
        tangents = []
        for vector in np.asarray(basis).T:
            x = np.r_[C.ravel(), Cs].astype(complex) + 1e-20j*vector
            ci = build(x[:C.size].reshape(C.shape), x[C.size:])
            tangents.append((ci/np.sqrt(np.sum(ci*ci))).imag/1e-20)
        return np.einsum('iab,jab->ij', np.asarray(tangents), np.asarray(tangents))

    def optimize(self, C, Cs, *, optimization_method='simultaneous',
                 spin_overlap_cutoff=1e-10, orbital_micro_iterations=0,
                 orbital_update='bfgs', spin_solver='canonical', orbital_overlap_cutoff=1e-10,
                 joint_correction=False, joint_correction_start=1, **options):
        """Explicit method selection; the historical simultaneous solver is unchanged."""
        from AlternatingOptimization import validate_options
        validate_options(optimization_method, spin_overlap_cutoff, orbital_micro_iterations,
                         orbital_update, spin_solver, orbital_overlap_cutoff, joint_correction, joint_correction_start)
        if optimization_method == 'simultaneous':
            return self.augmentedLagrangian(C, Cs, **options)
        if optimization_method != 'alternating':
            raise ValueError('optimization_method must be simultaneous or alternating')
        from AlternatingOptimization import optimize
        return optimize(self, C, Cs, spin_overlap_cutoff=spin_overlap_cutoff,
                        orbital_micro_iterations=orbital_micro_iterations,
                        orbital_update=orbital_update, spin_solver=spin_solver,
                        orbital_overlap_cutoff=orbital_overlap_cutoff,
                        joint_correction=joint_correction, joint_correction_start=joint_correction_start, **options)

    def refine_candidate(self, x, max_iterations):
        """Refine the same physical state without a growing norm penalty.

        The Rayleigh quotient is equivalent to the normalized constrained
        objective. Acceptance is checked separately after exact normalization.
        """
        start = self.normalized_candidate(x)

        def objective(values):
            C, Cs = self.split_x(values)
            energy, norm = self.calcEnergy(C, Cs)
            if norm <= 0 or not np.isfinite(norm):
                raise FloatingPointError("Non-positive norm in energy refinement")
            value = energy / norm
            if not np.isfinite(value):
                raise FloatingPointError("Non-finite energy in refinement")
            return value

        result = minimize(objective, start, jac=self.rayleigh_gradient,
                          options={'maxiter': max_iterations, 'gtol': 1e-8})
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise FloatingPointError("Non-finite refinement result")
        # A line-search status alone cannot certify or reject stationarity.
        return self.normalized_candidate(result.x)


    def augmentedLagrangian(self, C, Cs, verbose=False, epsilon=1e-8,
                            max_outer_iterations=50, max_inner_iterations=1000):
        options = dict(verbose=verbose, epsilon=epsilon,
                       max_outer_iterations=max_outer_iterations, max_inner_iterations=max_inner_iterations)
        diagnostics = getattr(self, 'diagnostics', None)
        if diagnostics is None:
            return self._augmentedLagrangian(C, Cs, **options)
        diagnostics.start(self)
        self._diagnostic_x = np.r_[C.ravel(), Cs].copy()
        self._diagnostic_best_x = None
        diagnostics.snapshot(self, 'start', self._diagnostic_x, optimizer_options=options)
        try:
            result = self._augmentedLagrangian(C, Cs, **options)
        except Exception as exc:
            diagnostics.snapshot(self, 'failure',
                                 self._diagnostic_best_x if self._diagnostic_best_x is not None
                                 else self._diagnostic_x, error=str(exc))
            raise
        diagnostics.snapshot(self, 'accepted', np.r_[result[1].ravel(), result[2]])
        return result

    def _augmentedLagrangian(self, C, Cs, verbose=False, epsilon=1e-8,
                            max_outer_iterations=50, max_inner_iterations=1000):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        for limit in (max_outer_iterations, max_inner_iterations):
            if not isinstance(limit, (int, np.integer)) or limit < 1:
                raise ValueError("Iteration limits must be positive integers")
        mu = 1.0
        lam = 0.0
        x = np.concatenate([C.flatten(), Cs.flatten()])
        if not np.isfinite(x).all():
            raise ValueError("Initial coefficients must be finite")
        previous_constraint = np.inf
        stalled_iterations = 0
        energy_tolerance = 1e-10
        best_x = None
        best_energy = np.inf
        rejected_iterations = 0

        def physical_energy(values):
            numerator, norm = self.calcEnergy(*self.split_x(values))
            if not np.isfinite(norm) or norm <= 0:
                raise FloatingPointError('Invalid norm in restart energy comparison')
            energy = numerator / norm
            if not np.isfinite(energy):
                raise FloatingPointError('Non-finite restart energy')
            return float(energy)

        def retain(values):
            nonlocal best_x, best_energy
            normalized = self.normalized_candidate(values)
            norm = self.calcEnergy(*self.split_x(normalized))[1]
            energy = physical_energy(normalized)
            if abs(norm - 1.) < epsilon and energy < best_energy:
                best_x, best_energy = normalized.copy(), energy
                if getattr(self, 'diagnostics', None) is not None:
                    self._diagnostic_best_x = best_x.copy()
                    self.diagnostics.snapshot(self, 'best', best_x)
            return energy

        retain(x)
        for iteration in range(1, max_outer_iterations + 1):
            target = self.gen_target(mu, lam)
            try:
                res = minimize(
                    target, x, jac=lambda values: self.target_gradient(values, mu, lam),
                    options={'maxiter': max_inner_iterations, 'gtol': 1e-8},
                )
                if not np.isfinite(res.x).all() or not np.isfinite(res.fun):
                    raise FloatingPointError("Inner optimizer returned non-finite values")
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    c = self.delta_constraint(res.x)
                if not np.isfinite(c):
                    raise FloatingPointError("Wavefunction norm is non-finite")
            except (FloatingPointError, OverflowError) as exc:
                raise RuntimeError(
                    f"Energy optimization diverged at outer iteration {iteration} "
                    f"(mu={mu:g}): {exc}. No converged energy is available."
                ) from exc
            if verbose:
                print(f"Iteration {iteration}, mu={mu:g}, constraint={c:g}: {res.message}")
            diagnostics = getattr(self, 'diagnostics', None)
            if diagnostics is not None:
                self._diagnostic_x = res.x.copy()
                diagnostics.snapshot(self, 'penalty', res.x, iteration=iteration, mu=mu, lam=lam,
                                     optimizer_status=int(res.status))
            previous_best_energy = best_energy
            refinement_start = best_x.copy() if best_x is not None else x.copy()
            restart_candidate = None
            gradient_max = np.inf
            candidate_error = np.inf
            unit_norm = np.nan
            # An inner stationary state can have Delta != 1 simply because
            # the finite penalty allows an overall amplitude error. Project
            # that redundant scale exactly, and independently check physical
            # stationarity before accepting it. This avoids growing mu merely
            # to correct an amplitude that has no physical significance.
            try:
                candidate = self.normalized_candidate(res.x)
                retain(candidate)
                physical_gradient = self.rayleigh_gradient(candidate)
                candidate_error = self.calcEnergy(*self.split_x(candidate))[1] - 1.
                stationary = (np.isfinite(physical_gradient).all()
                              and np.max(np.abs(physical_gradient)) <= 1e-6)
                if not stationary and res.status in (1, 2):
                    # Increasing mu cannot cure a stalled search in physical
                    # orbital directions. Refine Ee/Delta instead, then check
                    # both physical stationarity and normalization again.
                    # Restart from the incoming state: a failed penalty solve
                    # may have driven its orbitals close to linear dependence.
                    candidate = self.refine_candidate(refinement_start, max_inner_iterations)
                    physical_gradient = self.rayleigh_gradient(candidate)
                    candidate_error = self.calcEnergy(*self.split_x(candidate))[1] - 1.
                    stationary = (np.isfinite(physical_gradient).all()
                                  and np.max(np.abs(physical_gradient)) <= 1e-6)
                    # Do not discard useful physical progress just because a
                    # single refinement did not reach the gradient tolerance.
                    before_e, before_n = self.calcEnergy(*self.split_x(x))
                    after_e, after_n = self.calcEnergy(*self.split_x(candidate))
                    if (before_n > 0 and after_n > 0
                            and abs(candidate_error) < epsilon
                            and after_e / after_n < before_e / before_n - 1e-10):
                        restart_candidate = candidate
                candidate_energy = retain(candidate)
                gradient_max = np.max(np.abs(physical_gradient))
                unit_norm = self.unit_orbital_wavefunction_norm(candidate)
                if diagnostics is not None:
                    self._diagnostic_x = candidate.copy()
                    diagnostics.snapshot(self, 'candidate', candidate, iteration=iteration, mu=mu, lam=lam)
                if verbose:
                    print(f"Normalized candidate: constraint={candidate_error:g}, "
                          f"physical_gradient={gradient_max:g}, "
                          f"unit_orbital_norm={unit_norm:g}")
                if (stationary and abs(candidate_error) < epsilon
                        and candidate_energy <= best_energy + energy_tolerance):
                    x = candidate
                    if verbose:
                        print("Accepted normalized state: physical gradient <= 1e-6, "
                              f"constraint={candidate_error:g}")
                    break
            except (FloatingPointError, OverflowError) as exc:
                # A singular trial is not accepted; retain the bounded AL path.
                if verbose:
                    print(f"Physical refinement failed: {exc}")
            acceptable_inner = res.success
            if not res.success:
                # Line searches may narrowly miss BFGS's requested gradient
                # tolerance at a feasible point. Preserve that usable result,
                # but explicitly report the weaker stationarity check.
                gradient = np.asarray(getattr(res, 'jac', np.inf))
                near_stationary = (np.isfinite(gradient).all()
                                   and np.max(np.abs(gradient)) <= 1e-4)
                if res.status == 2 and abs(c) < epsilon and near_stationary:
                    acceptable_inner = True
                    warnings.warn(
                        "BFGS reported precision loss at a feasible point; "
                        "gradient infinity norm is <= 1e-4, but the requested "
                        "1e-8 stationarity tolerance was not certified.",
                        RuntimeWarning, stacklevel=2,
                    )
                elif res.status in (1, 2) and np.isfinite(gradient).all():
                    # An inexact inner solve is not a failure of the outer
                    # augmented-Lagrangian iteration. Update the multipliers
                    # and retry, but do not declare convergence on norm alone.
                    if verbose:
                        print("Continuing outer iteration after an inexact inner solve.")
                else:
                    raise RuntimeError(
                        f"Inner energy optimization failed at outer iteration {iteration}: "
                        f"{res.message} (constraint={c:g}, mu={mu:g}). "
                        "No converged energy is available."
                    )
            # Compare physical energies, never the penalty objective. Keep the
            # best normalized state even when a later penalty/refinement fails.
            try:
                raw_energy = physical_energy(res.x)
            except (FloatingPointError, OverflowError):
                raw_energy = np.inf
            if best_x is not None and (raw_energy > best_energy + energy_tolerance
                                       or restart_candidate is not None):
                improved = best_energy < previous_best_energy - energy_tolerance
                rejected_iterations = 0 if improved else rejected_iterations + 1
                stalled_iterations = 0
                x = best_x.copy()
                self._diagnostic_x = x.copy()
                if diagnostics is not None:
                    diagnostics.snapshot(self, 'best_restart', x, iteration=iteration,
                                         rejected_energy=raw_energy, best_energy=best_energy)
                if verbose:
                    print(f'Restarting from best physical state: energy={best_energy:.12g}, '
                          f'rejected_energy={raw_energy:.12g}')
                mu, lam = 1.0, 0.0
                previous_constraint = np.inf
                if rejected_iterations >= 5:
                    raise RuntimeError(
                        f'Energy optimization stalled at outer iteration {iteration}: '
                        f'no improvement over retained best state (physical_gradient={gradient_max:g}). '
                        'No converged energy is available.')
                continue
            # The original delta_constraint normalized the spin slice of
            # res.x in place. Preserve that restart scale explicitly, without
            # restoring the unintended mutation in split_x.
            rejected_iterations = 0
            next_C, next_Cs = self.split_x(res.x)
            x = np.concatenate([next_C.ravel(), next_Cs])
            if np.abs(c) < epsilon and acceptable_inner:
                break
            stalled_iterations = (stalled_iterations + 1
                                  if abs(c) >= 0.95 * previous_constraint else 0)
            if stalled_iterations >= 5 or mu >= 1e8:
                raise RuntimeError(
                    f"Energy optimization stalled at outer iteration {iteration} "
                    f"(constraint={c:g}, normalized_constraint={candidate_error:g}, "
                    f"physical_gradient={gradient_max:g}, unit_orbital_norm={unit_norm:g}, "
                    f"mu={mu:g}). "
                    "Increasing the outer iteration limit will not resolve this stall. "
                    "No converged energy is available."
                )
            lam += 2.0 * mu * c
            # Increase the penalty only when feasibility is not improving.
            if abs(c) > 0.25 * previous_constraint:
                mu *= 2
            previous_constraint = abs(c)
            if not np.isfinite(mu) or not np.isfinite(lam):
                raise RuntimeError("Non-finite augmented Lagrangian multipliers")
        else:
            raise RuntimeError(
                f"Energy optimization did not converge in {max_outer_iterations} "
                f"outer iterations (constraint={c:g}, "
                f"normalized_constraint={candidate_error:g}, "
                f"physical_gradient={gradient_max:g}, unit_orbital_norm={unit_norm:g}). "
                "No converged energy is available."
            )
        C, Cs = self.split_x(x)
        Ee, Delta_s = self.calcEnergy(C, Cs)
        Total_Energy = Ee / Delta_s + self.energy_nuc + getattr(self, 'inactive_energy', 0.)
        return Total_Energy, C, Cs
