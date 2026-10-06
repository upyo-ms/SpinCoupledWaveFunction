#!/sw/bin/python3.4
import math


def _integer(value, name):
    """Accept integral floats used by legacy callers without truncation."""
    integer = int(value)
    if integer != value:
        raise ValueError(f"{name} must be an integer")
    return integer


def combination(a, b):
    a = _integer(a, "a")
    b = _integer(b, "b")
    if a < 0:
        raise ValueError("a must be non-negative")
    if b < 0 or b > a:
        return 0
    return math.comb(a, b)


def spin_function_dim(N, Sm, option):
    # if Sm is given by spin multiplicity, option == 0.
    # if Sm is given by total spin number, option should have a value.

    N = _integer(N, "N")
    twice_spin = _integer(Sm - 1 if option == 0 else 2 * Sm, "2*S")
    if N < 0 or twice_spin < 0 or twice_spin > N:
        raise ValueError("Require N >= 0 and 0 <= 2*S <= N")
    if (N - twice_spin) % 2:
        raise ValueError("Electron number and spin multiplicity are inconsistent")

    N_beta = (N - twice_spin) // 2
    return combination(N, N_beta) - combination(N, N_beta - 1)
