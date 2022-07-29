import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def stationary_kernel(f, x, y):
    return jnp.prod(f(x - y))


@jax.jit
def gaussian_kernel(x, y, shape=jnp.diag(jnp.ones(2))):
    return jax.scipy.stats.multivariate_normal.pdf(x, y, shape)


@jax.jit
def _tricube(x):
    return (70 / 81) * jnp.power(1 - jnp.power(jnp.abs(x), 3), 3)


@jax.jit
def tricube(x, shape=1):
    t = x / shape
    conditions = [jnp.abs(t) > 1, jnp.abs(t) <= 1]
    return jnp.piecewise(t, conditions, [jnp.zeros_like, _tricube])


@partial(jax.jit, static_argnums=(0,))
def total_density(kernel, gridposition, agentpositions):
    return jnp.sum(
        vmap(stationary_kernel, (None, None, 0))(kernel, gridposition, agentpositions)
    )


@partial(jax.jit, static_argnums=(0,))
def grid_density(kernel, gridpositions, xs):
    return vmap(vmap(total_density, (None, 0, None)), (None, 0, None))(
        kernel, gridpositions, xs
    )
