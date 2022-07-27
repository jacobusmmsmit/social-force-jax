import jax
import jax.numpy as jnp
from jax import vmap


def total_density(gridposition, agentpositions, shape=jnp.diag(jnp.ones(2))):
    return jnp.sum(
        vmap(jax.scipy.stats.multivariate_normal.pdf, (None, 0, None))(
            gridposition, agentpositions, shape
        )
    )


def grid_density(gridpositions, xs):
    return vmap(vmap(total_density, (0, None)), (0, None))(gridpositions, xs)
