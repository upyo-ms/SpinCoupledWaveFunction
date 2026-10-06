"""Compare float64 energy+gradient time; compilation and cache hits are separated."""
import argparse
import time
import numpy as np
from pyscf import gto
from NonorthgonalEnergy import SCWF_Energy as NumPyEnergy
from NonorthgonalEnergyJAX import SCWF_Energy as JAXEnergy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    reference, accelerated = NumPyEnergy(mol), JAXEnergy(mol, device=args.device)
    C = np.random.RandomState(40).rand(len(reference.S), reference.Ne)
    x = reference.normalized_candidate(np.r_[C.ravel(), np.ones(reference.fs)])

    def evaluate(system):
        # Avoid measuring the last-point cache across repeated calls. Within
        # each value/gradient pair the shared evaluation is deliberately reused.
        if isinstance(system, JAXEnergy):
            system._last_evaluation = None
        value = system.gen_target(1., 0.)(x)
        gradient = system.target_gradient(x, 1., 0.)
        # Public methods materialize NumPy values, synchronizing GPU execution.
        return value, gradient

    start = time.perf_counter()
    value, gradient = evaluate(accelerated)
    first_call = time.perf_counter() - start
    expected_value, expected_gradient = evaluate(reference)
    np.testing.assert_allclose(value, expected_value, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-8, rtol=1e-9)
    print('Device:', accelerated.device)
    print(f'First JAX value+gradient call (includes compilation): {first_call:.3f} s')
    times = []
    for label, system in [('NumPy complex step', reference), ('JAX autodiff', accelerated)]:
        start = time.perf_counter()
        for _ in range(args.repeats):
            evaluate(system)
        elapsed = (time.perf_counter() - start) / args.repeats
        times.append(elapsed)
        print(f'{label}: {elapsed * 1000:.3f} ms per value+gradient')
    print(f'Warm value+gradient speedup: {times[0] / times[1]:.2f}x')


if __name__ == '__main__':
    main()
