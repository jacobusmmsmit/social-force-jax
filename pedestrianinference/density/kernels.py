import jax
import jax.numpy as jnp

from jax import jit
from functools import partial


@partial(jit, static_argnums=(0,))
def stationary_kernel(f, x, y):
    return jnp.prod(f(x - y))


@jit
def gaussian(x, y, shape=jnp.diag(jnp.ones(2))):
    return jax.scipy.stats.multivariate_normal.pdf(x, y, shape)


@jit
def _tricube(x):
    return (70 / 81) * jnp.power(1 - jnp.power(jnp.abs(x), 3), 3)


@jit
def tricube(x, shape=1):
    t = x / shape
    conditions = [jnp.abs(t) > 1, jnp.abs(t) <= 1]
    return jnp.piecewise(t, conditions, [jnp.zeros_like, _tricube])
