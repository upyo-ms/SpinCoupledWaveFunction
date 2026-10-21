"""Opt-in analytic Newton corrections for SCGVB C and spin-coupling coefficients.

The physical wavefunction metric removes redundant directions. Corrections are
bounded and checked with the selected energy evaluator; the existing optimizer
and its stationarity/normalization/retained-energy gates still decide success.
"""
import numpy as np
from pyscf import ao2mo, fci

from scgvb.energy.determinants import make_ci_builder
from scgvb.optimization.core_response import AmplitudeDerivatives, internal_curvature


def resolve_option(second_order=None, core_response=None):
    """The old public name is an alias, not a separately enabled algorithm."""
    for name, value in (('second_order', second_order), ('core_response', core_response)):
        if value is not None and type(value) is not bool:
            raise ValueError(f'{name} must be a boolean')
    if second_order is not None and core_response is not None and second_order != core_response:
        raise ValueError('second_order and its legacy alias core_response disagree')
    return bool(second_order if second_order is not None else core_response)


class InternalCorrections:
    """One fixed Hamiltonian; no independent determinant coefficients are varied."""
    def __init__(self, system, epsilon=1e-8):
        self.system, self.epsilon = system, epsilon
        self.amplitudes = AmplitudeDerivatives(system)
        self.build = make_ci_builder(system)
        w, v = np.linalg.eigh(system.S)
        inverse = np.einsum('ik,k,jk->ij', v, 1/np.sqrt(w), v, optimize=False)
        if np.array_equal(system.S, np.eye(len(w))):
            one, two = system.Hcore, system.JK
        else:
            intermediate = np.einsum('ip,ij->pj', inverse, system.Hcore, optimize=False)
            one = np.einsum('pi,iq->pq', intermediate, inverse, optimize=False)
            two = ao2mo.incore.full(system.JK, inverse, compact=False).reshape((len(w),)*4)
        n, ne, spin = len(w), system.Ne, system.Sm-1
        self.electrons = (ne+spin)//2, (ne-spin)//2
        self.h2 = fci.direct_spin1.absorb_h1e(one, two, n, self.electrons, .5)
        self.lowest = self.stationary = None
        self.info = dict(enabled=True, method='analytic_safeguarded_newton',
                         initial_step_limit=8, refinement_step_limit=16,
                         accepted_steps=0, attempts=0, history=[])

    def apply_h(self, ci):
        return fci.direct_spin1.contract_2e(self.h2, np.ascontiguousarray(ci),
                                          len(self.system.S), self.electrons)

    def canonical(self, C, Cs):
        norms = np.einsum('ia,ij,ja->a', C, self.system.S, C, optimize=False)
        sn = np.linalg.norm(Cs)
        if not np.isfinite(norms).all() or np.any(norms <= 0) or not np.isfinite(sn) or sn <= 0:
            raise FloatingPointError('Invalid coefficients in second-order correction')
        return C/np.sqrt(norms), Cs/sn

    def offer(self, C, Cs):
        s = self.system
        x = s.normalized_candidate(np.r_[C.ravel(), Cs])
        numerator, norm = s.calcEnergy(*s.split_x(x))
        gradient = s.rayleigh_gradient(x)
        energy = float(numerator/norm)
        if not np.isfinite(energy) or not np.isfinite(gradient).all() or abs(norm-1) >= self.epsilon:
            raise FloatingPointError('Invalid normalized second-order trial')
        state = dict(x=x.copy(), active_energy_hartree=energy, norm=float(norm),
                     physical_gradient_max_abs=float(np.max(np.abs(gradient))))
        if self.lowest is None or energy < self.lowest['active_energy_hartree']:
            self.lowest = state
        if state['physical_gradient_max_abs'] <= 1e-6 and (self.stationary is None or energy < self.stationary['active_energy_hartree']):
            self.stationary = state
        return state

    def correct(self, C, Cs, *, max_steps, stage, norm_floor=0.):
        C, Cs = self.canonical(np.asarray(C, dtype=float), np.asarray(Cs, dtype=float))
        state = self.offer(C, Cs)
        radius = .1
        for iteration in range(max_steps):
            if state['physical_gradient_max_abs'] <= 1e-6:
                break
            record = dict(stage=stage, iteration=iteration+1, accepted=False)
            self.info['attempts'] += 1
            self.info['history'].append(record)
            try:
                data = internal_curvature(self.amplitudes, C, Cs, self.apply_h)
                if not np.isfinite(data['gradient']).all():
                    raise FloatingPointError('Nonfinite analytic second-order gradient')
                ordinary = self.system.rayleigh_gradient(np.r_[C.ravel(), Cs])
                error = float(np.max(abs(ordinary-data['gradient'])))
                record.update(cancellation_ratio=data['cancellation_ratio'], gradient_disagreement=error)
                if error > max(1e-9, 1e-5*float(np.max(abs(ordinary)))):
                    record['status'] = 'unreliable_curvature'
                    break
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status='unreliable_curvature', reason=str(exc))
                break
            tangent, A = data['tangent'], data['reduced_hessian']
            if not A.size:
                record['status'] = 'zero_tangent_rank'
                break
            g = np.einsum('ia,i->a', tangent, data['gradient'], optimize=False)
            try:
                w, v = np.linalg.eigh(A)
            except np.linalg.LinAlgError as exc:
                record.update(status='unreliable_curvature', reason=str(exc))
                break
            rotated = np.einsum('ia,i->a', v, g, optimize=False)
            shift = max(0., 1e-6-float(w[0]))
            for _ in range(64):
                step = -np.einsum('ia,a->i', v, rotated/(w+shift), optimize=False)
                if np.linalg.norm(step) <= radius:
                    break
                shift = max(2*shift, 1e-6)
            direction = np.einsum('ia,a->i', tangent, step, optimize=False)
            dc, ds = direction[:C.size].reshape(C.shape), direction[C.size:]
            largest = max(float(np.max(np.sqrt(np.maximum(0., np.einsum('ia,ij,ja->a', dc, self.system.S, dc, optimize=False))))),
                          float(np.linalg.norm(ds)))
            if largest > .2:
                direction *= .2/largest
                step *= .2/largest
            predicted = float(-g@step-.5*step@A@step)
            record.update(tangent_rank=len(w), curvature_min=float(w[0]), curvature_shift=shift,
                          energy_before=state['active_energy_hartree'],
                          gradient_before=state['physical_gradient_max_abs'])
            if not predicted > 0 or not np.isfinite(direction).all():
                record['status'] = 'no_descent'
                break
            for exponent in range(32):
                fraction = 2.**(-exponent)
                try:
                    trial = np.r_[C.ravel(), Cs]+fraction*direction
                    tc, ts = self.canonical(trial[:C.size].reshape(C.shape), trial[C.size:])
                    if self.system.calcEnergy(tc, ts)[1] < norm_floor:
                        continue
                    candidate = self.offer(tc, ts)
                    ci = self.build(tc, ts).ravel()
                    ci_norm = np.linalg.norm(ci)
                    if not np.isfinite(ci_norm) or ci_norm <= 0:
                        raise FloatingPointError('Invalid trial wavefunction in second-order correction')
                    ci /= ci_norm
                    distance = float(np.sqrt(max(0., 1-float(ci@data['ci'].ravel())**2)))
                    drop = state['active_energy_hartree']-candidate['active_energy_hartree']
                    improved = drop > 1e-12 or (drop >= -1e-12 and candidate['physical_gradient_max_abs'] < .99*state['physical_gradient_max_abs'])
                    if distance <= radius*(1+1e-6) and improved:
                        break
                except (FloatingPointError, OverflowError):
                    pass
            else:
                record['status'] = 'line_search_rejected'
                break
            record.update(accepted=True, status='accepted', fraction=fraction, wavefunction_distance=distance,
                          energy_after=candidate['active_energy_hartree'],
                          gradient_after=candidate['physical_gradient_max_abs'])
            self.info['accepted_steps'] += 1
            C, Cs, state = tc, ts, candidate
            diagnostics = getattr(self.system, 'diagnostics', None)
            if diagnostics is not None:
                details = {k: v for k, v in record.items() if k != 'stage'}
                diagnostics.snapshot(self.system, 'second_order_correction', state['x'],
                                     second_order_phase=stage, **details)
        result_C, result_Cs = self.system.split_x(state['x'])
        return result_C.copy(), result_Cs.copy()

    def merge_retained(self):
        s = self.system
        states = getattr(s, 'retained_optimization_states', {}) or {}
        for value in list(states.values()):
            if value is not None:
                self.offer(*s.split_x(value['x']))
        s.retained_optimization_states = dict(lowest_feasible=self.lowest, stationary_candidate=self.stationary)


def optimize(system, C, Cs, run, options):
    """Apply bounded analytic corrections, then the requested existing solver."""
    epsilon = options.get('epsilon', 1e-8)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    for name in ('max_inner_iterations', 'max_outer_iterations'):
        count = options.get(name, 1000 if name == 'max_inner_iterations' else 50)
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('Iteration limits must be positive integers')
    context = InternalCorrections(system, epsilon)
    system.second_order_info = context.info
    previous = getattr(system, '_second_order_context', None)
    system._second_order_context = context
    try:
        diagnostics = getattr(system, 'diagnostics', None)
        if diagnostics is not None:
            diagnostics.start(system)
            context.diagnostics_started = True
            diagnostics.snapshot(system, 'second_order_start', np.r_[C.ravel(), Cs])
        C, Cs = context.correct(C, Cs, max_steps=min(8, options.get('max_inner_iterations', 1000)), stage='initial')
        result = run(C, Cs)
        final = context.offer(result[1], result[2])
        context.merge_retained()
        if final['active_energy_hartree'] > context.lowest['active_energy_hartree']+1e-10:
            raise RuntimeError('Second-order solve retained a lower feasible state; no converged energy is available')
        if final['physical_gradient_max_abs'] > 1e-6:
            raise RuntimeError('Second-order solve did not satisfy physical stationarity')
        return result
    except Exception as exc:
        context.merge_retained()
        exc.retained_optimization_states = system.retained_optimization_states
        exc.second_order_info = context.info
        diagnostics = getattr(system, 'diagnostics', None)
        if diagnostics is not None:
            diagnostics.snapshot(system, 'second_order_failure',
                context.lowest['x'] if context.lowest is not None else None, error=str(exc))
        raise
    finally:
        if previous is None:
            del system._second_order_context
        else:
            system._second_order_context = previous
