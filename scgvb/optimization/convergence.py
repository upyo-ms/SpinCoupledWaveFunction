"""Typed electronic nonconvergence and a single optional second-order retry."""
import numpy as np


class ElectronicConvergenceError(RuntimeError):
    """A valid electronic optimization could not certify a converged state."""


def restart_candidate(system, C, Cs, error):
    """Choose a finite normalized seed in this exact Hamiltonian and basis."""
    candidates = [('incoming', np.r_[C.ravel(), Cs])]
    for name, state in (getattr(error, 'retained_optimization_states', None) or {}).items():
        if state is not None:
            candidates.append((name, state['x']))
    best = None
    for name, values in candidates:
        try:
            values = system.normalized_candidate(np.asarray(values, dtype=float).copy())
            c, cs = system.split_x(values)
            numerator, norm = system.calcEnergy(c, cs)
            energy = float(numerator/norm)
            if norm <= 0 or not np.isfinite(energy) or not np.isfinite(norm) or abs(norm-1.) > 1e-6:
                continue
        except (ValueError, FloatingPointError, OverflowError):
            continue
        if best is None or energy < best[0]:
            best = energy, c.copy(), cs.copy(), name
    if best is None:
        raise ElectronicConvergenceError('No valid wavefunction is available for the second-order retry') from error
    return best[1], best[2], best[3]


def optimize_electronic(system, C, Cs, *, optimize_core, second_order, second_order_fallback,
                        core_max_iterations, core_gradient_tolerance, optimizer_options, verbose):
    """Return the usual electronic result, effective mode and retry record.

    A core failure may carry a reconstructed system whose orbital basis matches
    its coefficients. Coefficients from different orbital charts are never mixed.
    """
    info = dict(enabled=second_order_fallback, triggered=False, status='not_needed',
                first_order_error=None, restart_source=None)
    if not second_order_fallback:
        info['status'] = 'disabled'
    elif second_order:
        info['status'] = 'second_order_requested'

    def run(s, c, cs, enabled, capture=False):
        options = dict(optimizer_options, second_order=enabled)
        if optimize_core:
            from scgvb.optimization.core import optimize_core as relax_core
            return relax_core(s, c.copy(), cs.copy(), max_iterations=core_max_iterations,
                gradient_tolerance=core_gradient_tolerance, optimizer_options=options,
                verbose=verbose, second_order=enabled, capture_failure_state=capture)
        total, c, cs = s.optimize(c.copy(), cs.copy(), **options)
        if second_order_fallback:
            norm = float(s.calcEnergy(c, cs)[1])
            gradient = float(np.max(np.abs(s.rayleigh_gradient(np.r_[c.ravel(), cs]))))
            if (not np.isfinite([norm, gradient]).all() or gradient > 1e-6
                    or abs(norm-1.) >= options.get('epsilon', 1e-8)):
                error = ElectronicConvergenceError('Electronic solve did not satisfy physical stationarity '
                    f'and normalization (gradient={gradient:g}, norm_error={norm-1.:g})')
                error.retained_optimization_states = dict(
                    getattr(s, 'retained_optimization_states', None) or {},
                    last_iterate=dict(x=np.r_[c.ravel(), cs]))
                raise error
        return total, c, cs, s, s.active_space, None

    try:
        result = run(system, C, Cs, second_order, capture=second_order_fallback and not second_order)
        return result, second_order, info
    except ElectronicConvergenceError as first:
        if second_order or not second_order_fallback:
            raise
        info.update(triggered=True, status='retrying', first_order_error=str(first))
        from scgvb.optimization.core_response import check_response_size
        try:
            check_response_size(len(system.S), system.Ne, system.mol.spin)
        except ValueError as limit:
            info.update(status='unavailable', unavailable_reason=str(limit))
            first.second_order_fallback_info = info
            first.args = (f'{first} Automatic second-order retry unavailable: {limit}',)
            raise first from limit
        try:
            s, c, cs = system, C, Cs
            if optimize_core:
                state = getattr(first, 'core_restart_state', None)
                if state is not None:
                    s, c, cs, source = state
                    info['restart_source'] = source
                else:
                    info['restart_source'] = 'incoming_core_state'
                    info['restart_note'] = getattr(first, 'core_restart_unavailable', None)
            else:
                c, cs, source = restart_candidate(s, c, cs, first)
                info['restart_source'] = source
            if verbose:
                print('Electronic optimization did not converge; retrying once with second_order=true.')
                print('Second-order retry source:', info['restart_source'])
            result = run(s, c, cs, True)
        except Exception as final:
            info.update(status='failed', second_order_error=str(final))
            final.second_order_fallback_info = info
            raise final from first
        info['status'] = 'converged'
        return result, True, info
