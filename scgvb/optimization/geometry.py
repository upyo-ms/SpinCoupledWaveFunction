"""SCGVB geometry optimization using energy-only BOBYQA or numerical forces.

Nuclear coordinates/difference steps are in Bohr; physical energies are in Eh.
Every energy is a converged electronic solve at the trial geometry, including
nuclear repulsion and inactive-space energy. Failed solves are never penalties.
"""
from collections import OrderedDict
from pathlib import Path
import json
import tempfile
import time

import numpy as np

from .coordinates import BOHR, CoordinateError, GeometryCoordinates


class EvaluationLimit(RuntimeError):
    """The budget counts actual electronic solves, including unsuccessful ones."""


def validate_geometry_options(*, method, coordinates, max_iterations, max_evaluations,
                              difference_step, gradient_tolerance, energy_tolerance,
                              step_tolerance, trust_radius, max_radius, final_radius):
    if method not in ('bobyqa', 'quasi-newton'):
        raise ValueError('geometry method must be bobyqa or quasi-newton')
    if coordinates not in ('auto', 'dlc', 'tric', 'cartesian'):
        raise ValueError('Invalid geometry coordinate system')
    for name, value in [('max_iterations', max_iterations), ('max_evaluations', max_evaluations)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'geometry {name} must be a positive integer')
    for name, value in [('difference_step', difference_step), ('gradient_tolerance', gradient_tolerance),
                        ('energy_tolerance', energy_tolerance), ('step_tolerance', step_tolerance),
                        ('trust_radius', trust_radius), ('max_radius', max_radius), ('final_radius', final_radius)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'geometry {name} must be finite and positive')
    if not final_radius < trust_radius <= max_radius:
        raise ValueError('Geometry radii must satisfy 0 < final_radius < trust_radius <= max_radius')


class GeometryEvaluator:
    """Converged electronic evaluations with bounded cache and live checkpoints."""
    def __init__(self, mol, calculation_options, max_evaluations, *, calculator=None,
                 output=None, provenance=None):
        if calculator is None:
            from scgvb.cli.molecule import calculate
            calculator = calculate
        self.calculator = calculator
        self.symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        self.options = dict(calculation_options)
        self.options['unit'] = 'Bohr'
        self.options['allow_geometry_change'] = True
        self.initial_state = self.options.pop('initial_json', None)
        # Per-trial diagnostics otherwise overwrite one another.
        self.options.pop('diagnostics_dir', None)
        if self.options.get('verbose', 0) == 2:
            self.options['verbose'] = 1
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.cache = OrderedDict()
        self.best = None
        self.best_coords = None
        self.history = []
        self.structures = []
        self.info = {}
        self.output = None if output is None else Path(output)
        self.provenance = provenance
        self.started = time.monotonic()

    def ensure_budget(self, count=1):
        if self.evaluations + count > self.max_evaluations:
            raise EvaluationLimit(f'Electronic evaluation budget ({self.max_evaluations}) exhausted')

    def evaluate(self, coords, *, reference=None, role='trial'):
        x = np.asarray(coords, dtype=float).ravel()
        if x.shape != (3 * len(self.symbols),) or not np.isfinite(x).all():
            raise ValueError('Invalid nuclear coordinates')
        key = x.tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self.ensure_budget()
        seed = reference if reference is not None else self.best
        if seed is None:
            seed = self.initial_state
        self.evaluations += 1
        start = time.monotonic()
        record = dict(evaluation=self.evaluations, role=role, success=False)
        try:
            result = self.calculator(
                list(zip(self.symbols, x.reshape(-1, 3).tolist())),
                initial_json=seed, **self.options)
            energy = float(result['total_energy_hartree'])
            norm_error = abs(float(result['normalization_error']))
            electronic_gradient = float(result['physical_gradient_max_abs'])
            if (not np.isfinite([energy, norm_error, electronic_gradient]).all()
                    or norm_error > max(self.options.get('epsilon', 1e-8), 1e-8)
                    or electronic_gradient > 1.01e-6):
                raise RuntimeError('Electronic solve did not satisfy normalization/stationarity tolerances')
        except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
            record.update(message=str(exc), seconds=time.monotonic() - start)
            self.history.append(record)
            self.save('failed', str(exc))
            raise
        record.update(success=True, energy_hartree=energy, seconds=time.monotonic() - start,
                      electronic_gradient_max_abs=electronic_gradient)
        self.history.append(record)
        self.cache[key] = result
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        if self.best is None or energy < self.best['total_energy_hartree']:
            self.best, self.best_coords = result, x.copy()
            self.structures.append(dict(atoms=result['molecule']['atoms'], energy_hartree=energy))
        self.save('running', 'Optimization in progress')
        return result

    def snapshot(self, status, message, *, result=None):
        if result is None:
            result = self.best
        if result is None:
            return None
        result = dict(result)
        result['geometry_optimization'] = dict(
            self.info, status=status, converged=status == 'converged', message=message,
            energy_evaluations=self.evaluations, max_energy_evaluations=self.max_evaluations,
            elapsed_seconds=time.monotonic() - self.started, evaluation_history=list(self.history),
            best_structure_history=list(self.structures))
        if self.provenance is not None:
            result['input_file'] = self.provenance
        return result

    def save(self, status, message, *, result=None):
        snapshot = self.snapshot(status, message, result=result)
        if snapshot is not None and self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output.with_name(self.output.name + '.tmp')
            temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
            temporary.replace(self.output)
            xyz = self.output.with_name(self.output.stem + '_geometry.xyz')
            frames = []
            for structure in self.structures:
                atoms = structure['atoms']
                frames.extend([str(len(atoms)), f"Best energy {structure['energy_hartree']:.12f} Eh"])
                frames.extend(f'{symbol} {x:.12f} {y:.12f} {z:.12f}' for symbol, (x, y, z) in atoms)
            xyz.write_text('\n'.join(frames) + '\n')
        return snapshot


def central_gradient(evaluator, coords, step, *, reference=None):
    """Central differences; every +h/-h pair uses the same center state.

    The cache can avoid repeated solves, but the preflight budget conservatively
    reserves 6N evaluations so an incomplete gradient is never returned.
    """
    x = np.asarray(coords, dtype=float).ravel()
    if not np.isfinite(step) or step <= 0:
        raise ValueError('difference step must be finite and positive')
    center = evaluator.evaluate(x, reference=reference, role='gradient_center')
    evaluator.ensure_budget(2 * x.size)
    gradient = np.empty_like(x)
    for i in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        ep = evaluator.evaluate(plus, reference=center, role='difference_plus')['total_energy_hartree']
        em = evaluator.evaluate(minus, reference=center, role='difference_minus')['total_energy_hartree']
        gradient[i] = (ep - em) / (2 * step)
    return center, gradient


def _quasi_newton(evaluator, chart, *, max_iterations, difference_step,
                  gradient_tolerance, energy_tolerance, step_tolerance,
                  trust_radius, max_radius, verbose):
    from geometric.engine import Engine
    from geometric.optimize import Optimizer
    from geometric.params import OptParams

    class NumericalEngine(Engine):
        def __init__(self):
            super().__init__(chart.molecule)
            self.last_coords, self.last_result, self.last_gradient = None, None, None

        def calc_new(self, coords, dirname):
            result, gradient = central_gradient(evaluator, coords, difference_step)
            self.last_coords = np.asarray(coords).copy()
            self.last_result, self.last_gradient = result, gradient.copy()
            if verbose:
                print(f"Geometry E={result['total_energy_hartree']:.12f} Eh "
                      f"RMS gradient={np.sqrt(np.mean(gradient**2)):.3e} Eh/Bohr "
                      f"evaluations={evaluator.evaluations}")
            return dict(energy=result['total_energy_hartree'], gradient=gradient.copy())

    engine = NumericalEngine()
    params = OptParams(
        maxiter=max_iterations, trust=trust_radius, tmax=max_radius, hessian='never',
        convergence_grms=gradient_tolerance, convergence_gmax=1.5 * gradient_tolerance,
        convergence_energy=energy_tolerance, convergence_drms=step_tolerance,
        convergence_dmax=1.5 * step_tolerance, subfrctor=2, verbose=0)
    with tempfile.TemporaryDirectory(prefix='scgvb-geometry-') as directory:
        optimizer = Optimizer(chart.reference.copy(), chart.molecule,
                              chart.optimizer_coordinates(), engine, directory, params, print_info=False)
        try:
            optimizer.optimizeGeometry()
        finally:
            evaluator.info['iterations'] = optimizer.Iteration
        # Optimizer subtracts net force/torque; retain the original numeric gradient
        # for an independent final convergence check and unit-explicit output.
        if engine.last_coords is not None and np.array_equal(engine.last_coords, optimizer.X):
            result, gradient = engine.last_result, engine.last_gradient
        else:
            result, gradient = central_gradient(evaluator, optimizer.X, difference_step)
        rms, largest = float(np.sqrt(np.mean(gradient**2))), float(np.max(np.abs(gradient)))
        evaluator.info.update(nuclear_gradient_hartree_per_bohr=gradient.reshape(-1, 3).tolist(),
                              nuclear_gradient_rms=rms, nuclear_gradient_max_abs=largest)
        if rms > gradient_tolerance or largest > 1.5 * gradient_tolerance:
            return result, False, 'Unprojected numerical nuclear gradient exceeds tolerance'
        return result, True, 'Energy, numerical gradient and displacement criteria satisfied'


def _bobyqa(evaluator, chart, *, max_iterations, trust_radius, max_radius, final_radius, bonds, verbose):
    import pybobyqa
    from scgvb.cli.molecule import build_molecule

    radius = max_radius / BOHR
    beginning, ending = trust_radius / BOHR, final_radius / BOHR
    scale = np.sqrt(len(evaluator.symbols))
    center = evaluator.evaluate(chart.reference, role='initial')
    for stage in range(max_iterations):
        evaluator.info.update(iterations=stage + 1, coordinate_chart=chart.metadata,
                              chart_cartesian_scale=float(scale))
        reference = center

        def objective(y):
            coords = chart.to_cartesian(np.asarray(y) * scale)
            result = evaluator.evaluate(coords, reference=reference, role='bobyqa')
            if verbose:
                print(f"Geometry E={result['total_energy_hartree']:.12f} Eh evaluations={evaluator.evaluations}")
            return result['total_energy_hartree']

        def project_ball(y):
            return y * min(1., radius / max(np.linalg.norm(y), 1e-30))

        evaluator.ensure_budget()
        try:
            solution = pybobyqa.solve(
                objective, np.zeros(chart.dimension), projections=[project_ball],
                npt=2 * chart.dimension + 1, rhobeg=min(beginning, radius * .5), rhoend=ending,
                maxfun=evaluator.max_evaluations - evaluator.evaluations + 1,
                objfun_has_noise=False, user_params={'restarts.use_restarts': False},
                do_logging=False, print_progress=False)
        except CoordinateError as exc:
            radius *= .5
            evaluator.info['coordinate_restarts'] = evaluator.info.get('coordinate_restarts', 0) + 1
            if radius <= 2 * ending:
                return evaluator.best, False, f'Internal-coordinate conversion failed: {exc}'
        else:
            evaluator.info['optimizer_exit_flag'] = int(solution.flag)
            if solution.x is None:
                return evaluator.best, False, str(solution.msg)
            x = chart.to_cartesian(solution.x * scale)
            center = evaluator.evaluate(x, reference=reference, role='bobyqa_solution')
            on_boundary = np.linalg.norm(solution.x) >= .8 * radius
            if solution.flag == 0 and not on_boundary:
                return center, True, str(solution.msg)
            if not on_boundary or center['total_energy_hartree'] >= reference['total_energy_hartree']:
                return evaluator.best, False, str(solution.msg)
        # Recenter outside the solve, so existing interpolation points are never
        # silently interpreted in a different coordinate basis. Also handles
        # nonlinear inverse failures without substituting a fictitious energy.
        center = evaluator.best
        mol = build_molecule(center['molecule']['atoms'], basis=evaluator.options.get('basis', 'sto-3g'),
                             charge=evaluator.options.get('charge', 0),
                             multiplicity=evaluator.options.get('multiplicity', 1), unit='Angstrom')
        chart = GeometryCoordinates(mol, chart.kind, bonds=bonds)
        evaluator.info['chart_restarts'] = evaluator.info.get('chart_restarts', 0) + 1
    return evaluator.best, False, 'Maximum number of BOBYQA coordinate charts reached'


def optimize_geometry(atom, *, calculation_options=None, method='quasi-newton', coordinates='auto',
                      bonds=None, max_iterations=100, max_evaluations=2000, difference_step=1e-3,
                      gradient_tolerance=3e-4, energy_tolerance=1e-6, step_tolerance=1.2e-3,
                      trust_radius=.1, max_radius=.3, final_radius=1e-4,
                      output=None, provenance=None, calculator=None):
    """Return a calculation result plus ``geometry_optimization`` status/history.

    ``output`` is checkpointed after each successful electronic solve. Budget,
    convergence or electronic failures return the best valid state marked as
    nonconverged; if no valid state exists, the exception is propagated.
    BOBYQA convergence is a trust-radius criterion, not a force certification.
    ``max_iterations`` limits quasi-Newton steps or BOBYQA chart stages.
    """
    validate_geometry_options(
        method=method, coordinates=coordinates, max_iterations=max_iterations,
        max_evaluations=max_evaluations, difference_step=difference_step,
        gradient_tolerance=gradient_tolerance, energy_tolerance=energy_tolerance,
        step_tolerance=step_tolerance, trust_radius=trust_radius,
        max_radius=max_radius, final_radius=final_radius)
    if method == 'bobyqa':
        try:
            import pybobyqa  # noqa: F401; fail before expensive electronic solves
        except ImportError as exc:
            raise ImportError('BOBYQA requires Py-BOBYQA; install scgvb[geometry]') from exc
    try:
        from geometric.errors import Error as GeometricError, GeomOptNotConvergedError
    except ImportError as exc:
        raise ImportError('Geometry optimization requires geometric; install scgvb[geometry]') from exc
    from scgvb.cli.molecule import build_molecule
    options = {} if calculation_options is None else dict(calculation_options)
    if output is not None and isinstance(options.get('initial_json'), (str, Path)):
        if Path(output).resolve() == Path(options['initial_json']).resolve():
            raise ValueError('Restart source and geometry output must be different files')
    mol = build_molecule(atom, basis=options.get('basis', 'sto-3g'), charge=options.get('charge', 0),
                         multiplicity=options.get('multiplicity', 1), unit=options.get('unit', 'Angstrom'))
    chart = GeometryCoordinates(mol, coordinates, bonds=bonds)
    evaluator = GeometryEvaluator(mol, options, max_evaluations, calculator=calculator,
                                  output=output, provenance=provenance)
    evaluator.info = dict(
        method=method, coordinate_chart=chart.metadata, max_iterations=max_iterations,
        difference_step_bohr=difference_step, gradient_tolerance_hartree_per_bohr=gradient_tolerance,
        energy_tolerance_hartree=energy_tolerance, step_tolerance_angstrom=step_tolerance,
        trust_radius_angstrom=trust_radius, max_radius_angstrom=max_radius,
        final_radius_angstrom=final_radius,
        convergence_criterion='internal_trust_radius' if method == 'bobyqa' else 'energy_gradient_displacement')
    try:
        if method == 'bobyqa':
            result, converged, message = _bobyqa(
                evaluator, chart, max_iterations=max_iterations, trust_radius=trust_radius,
                max_radius=max_radius, final_radius=final_radius, bonds=bonds, verbose=options.get('verbose', 0))
        else:
            result, converged, message = _quasi_newton(
                evaluator, chart, max_iterations=max_iterations, difference_step=difference_step,
                gradient_tolerance=gradient_tolerance, energy_tolerance=energy_tolerance,
                step_tolerance=step_tolerance, trust_radius=trust_radius, max_radius=max_radius,
                verbose=options.get('verbose', 0))
        status = 'converged' if converged else 'not_converged'
    except EvaluationLimit as exc:
        result, status, message = evaluator.best, 'evaluation_limit', str(exc)
    except GeomOptNotConvergedError as exc:
        result, status, message = evaluator.best, 'not_converged', str(exc)
    except (ValueError, RuntimeError, FloatingPointError, GeometricError) as exc:
        if evaluator.best is None:
            raise
        result, status, message = evaluator.best, 'failed', str(exc)
    except KeyboardInterrupt:
        evaluator.save('interrupted', 'Interrupted; best converged electronic state retained')
        raise
    return evaluator.save(status, message, result=result)
