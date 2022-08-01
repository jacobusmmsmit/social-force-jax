import jax.numpy as jnp
import jax.random as jrd


def two_groups_datum(key, N, D=5, V=2):
    A = N // 2
    B = N - A
    x1 = V * (2 * jrd.uniform(key, (A, 1)) - 1) - D
    x2 = V * (2 * jrd.uniform(key, (B, 1)) - 1) + D
    x = jnp.row_stack((x1, x2))
    y = V * (2 * jrd.uniform(key, (N, 1)) - 1)
    v1 = jnp.ones((A, 1))
    v2 = -jnp.ones((B, 1))
    v = jnp.row_stack((v1, v2))
    w = jnp.zeros((N, 1))
    xs = jnp.column_stack((x, y, v, w))
    return xs
