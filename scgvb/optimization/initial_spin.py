"""Optional fixed-orbital spin initialization, independent of the optimizer."""
import numpy as np


def validate_initial_spin(method, overlap_cutoff):
    if method not in ('uniform', 'eigen'):
        raise ValueError('initial_spin_guess must be uniform or eigen')
    if not np.isfinite(overlap_cutoff) or not 0 < overlap_cutoff < 1:
        raise ValueError('initial_spin_overlap_cutoff must be finite and between 0 and 1')


def initial_spin_coefficients(system, C, Cs, *, method='uniform', overlap_cutoff=1e-10,
                              source='uniform'):
    """Return Cs and diagnostics without changing the supplied spatial orbitals.

    uniform preserves the supplied seed, including restored restart coefficients.
    eigen solves in the resolved overlap support plus the incoming physical state.
    Determinant evaluation uses the centered projected solver; the permutation
    evaluator uses the existing small generalized spin-matrix solver.
    """
    validate_initial_spin(method, overlap_cutoff)
    C, Cs = np.asarray(C), np.asarray(Cs, dtype=float)
    if C.shape != (len(system.S), system.Ne) or Cs.shape != (system.fs,):
        raise ValueError('Spin initialization coefficient shapes do not match the system')
    seed = Cs.copy()
    C, Cs = system.split_x(np.r_[np.asarray(C).ravel(), np.asarray(Cs)])
    Cs = np.asarray(Cs, dtype=float).copy()
    if (not np.isfinite(C).all() or not np.isfinite(Cs).all()
            or np.linalg.norm(Cs) <= 0):
        raise ValueError('Spin initialization requires finite orbitals and nonzero coefficients')
    info = dict(method=method, seed_source=source, performed=method == 'eigen',
                overlap_cutoff=overlap_cutoff if method == 'eigen' else None)
    if method == 'uniform':
        return seed, info
    from scgvb.optimization.alternating import normalized_columns, solve_spin, guarded_spin_update
    c, _ = normalized_columns(C, system.S)
    incoming = Cs / np.linalg.norm(Cs)
    num, norm = system.calcEnergy(c, incoming)
    if not np.isfinite(num) or not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Spin initialization starts from an invalid physical state')
    before = float(num / norm)
    if system.energy_evaluation == 'determinant':
        from scgvb.optimization.wavefunction import projected_spin_solve
        proposed, solver_info = projected_spin_solve(system, c, incoming, overlap_cutoff)
        solver = 'projected'
    else:
        h, metric = system.spin_matrices(c)
        proposed, solver_info = solve_spin(h, metric, overlap_cutoff, incoming=incoming)
        solver = 'canonical'
    cs, guard = guarded_spin_update(system, c, incoming, proposed)
    num, norm = system.calcEnergy(c, cs)
    after = float(num / norm)
    if not np.isfinite(after) or after > before + 1e-10:
        raise FloatingPointError('Spin initialization increased the physical energy')
    info.update(solver=solver, solver_info=solver_info, guard_info=guard,
                energy_before_hartree=before, energy_after_hartree=after,
                unit_orbital_norm=float(norm), coefficients=cs.tolist())
    return cs, info
