import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from .kernels import stationary_kernel


@partial(jax.jit, static_argnums=(0,))
def total(kernel, gridposition, agentpositions):
    return jnp.sum(
        vmap(stationary_kernel, (None, None, 0))(kernel, gridposition, agentpositions)
    )


@partial(jax.jit, static_argnums=(0,))
def grid(kernel, gridpositions, xs):
    return vmap(vmap(total, (None, 0, None)), (None, 0, None))(
        kernel, gridpositions, xs
    )
