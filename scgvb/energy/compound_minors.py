"""Polynomial compound matrices with no factorial permutation expansion.

Laplace expansion along the last row builds all k-by-k minors from order k-1.
There are no divisions, so JAX derivatives remain valid at singular submatrices.
Each checkpointed stage recomputes intermediates during reverse mode.
"""
from itertools import combinations
import numpy as np
import jax
import jax.numpy as jnp
from pyscf.fci import cistring


def occupation_rows(n, count):
    strings = cistring.make_strings(range(n), count)
    return np.asarray([[i for i in range(n) if int(s) & (1 << i)]
                       for s in strings], dtype=np.int32).reshape(len(strings), count)


def make_compounds(n, ne, orders, device):
    stages = []
    previous_rows = {(): 0}
    previous_columns = {(): 0}
    column_maps = {0: previous_columns}
    for k in range(1, max(orders) + 1):
        rows = occupation_rows(n, k)
        columns = list(combinations(range(ne), k))
        row_prefix = np.asarray([previous_rows[tuple(r[:-1])] for r in rows], dtype=np.int32)
        remove = np.asarray([[previous_columns[c[:j]+c[j+1:]] for c in columns]
                             for j in range(k)], dtype=np.int32)
        values = (row_prefix, rows[:, -1], remove, np.asarray(columns, dtype=np.int32).T)
        stages.append(tuple(jax.device_put(v, device) for v in values))
        previous_rows = {tuple(r): i for i, r in enumerate(rows)}
        previous_columns = {c: i for i, c in enumerate(columns)}
        column_maps[k] = previous_columns

    def compounds(D):
        previous = jnp.ones((1, 1), dtype=D.dtype)
        output = {0: previous}
        for k, (row_prefix, last_row, remove, columns) in enumerate(stages, 1):
            # Default arguments bind each stage's indices for checkpoint tracing.
            def stage(prev, orbital, rp=row_prefix, lr=last_row, rm=remove, cols=columns, degree=k):
                def term(j, total):
                    factor = orbital[lr[:, None], cols[j][None, :]]
                    minor = prev[rp[:, None], rm[j][None, :]]
                    sign = jnp.where((degree-1+j) % 2 == 0, 1., -1.)
                    return total + sign * factor * minor
                return jax.lax.fori_loop(0, degree, term,
                                        jnp.zeros((lr.size, cols.shape[1]), dtype=D.dtype))
            previous = jax.checkpoint(stage)(previous, D)
            if k in orders:
                output[k] = previous
        return tuple(output[k] for k in orders)

    return compounds, column_maps
