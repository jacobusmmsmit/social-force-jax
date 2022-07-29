import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import numpy as np
import numpyro
import numpyro.distributions as dist
import seaborn as sns

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from diffrax import (
        diffeqsolve,
        Tsit5,
        ODETerm,
        SaveAt,
        PIDController,
    )

from jax import random, lax
from jax import vmap
from jax.config import config
from matplotlib.animation import FuncAnimation, PillowWriter
from numpyro.infer import MCMC, NUTS

import src.socialforce as sf
from src import transformations

config.update("jax_debug_nans", True)

key = random.PRNGKey(42)

if __name__ == "__main__":
    ### Initialise parameters of simulation
    N = 20
    width = 5
    height = 3
    x0 = sf.two_groups_datum(key, N, width, 0.66 * height)

    # Model parameters
    pedestrian_strength = 2.1
    pedestrian_shape = 1.0
    wall_strength = 10.0
    wall_shape = 2.5
    relaxation_time = 0.5
    desired_velocities = x0[:, 2:4]

    # Walls
    top_wall = jnp.array([-100, height, 100, height])
    bottom_wall = jnp.array([-100, -height, 100, -height])
    middle_top_wall = jnp.array([0, height, 0, (3 / 4) * height])
    middle_bottom_wall = jnp.array([0, -height, 0, -(3 / 4) * height])
    walls = jnp.row_stack((top_wall, bottom_wall, middle_top_wall, middle_bottom_wall))

    args = (
        pedestrian_strength,
        pedestrian_shape,
        wall_strength,
        wall_shape,
        relaxation_time,
        desired_velocities,
        walls,
    )

    ### Set-up and solve model
    term = ODETerm(sf.step)
    solver = Tsit5()
    tspan = (0.0, 50.0)
    nsaves = 120
    saveat = SaveAt(ts=jnp.linspace(tspan[0], tspan[1], nsaves))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    sol = diffeqsolve(
        term,
        solver,
        t0=tspan[0],
        t1=tspan[1],
        dt0=0.1,
        y0=x0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    ### Animated plot of solution
    fig, ax = plt.subplots(1, 1)

    def animate(i):
        ax.clear()
        for wall in walls:
            ax.plot((wall[0], wall[2]), (wall[1], wall[3]), color="black")
        datat = sol.ys[i]
        ax.set_xlim([-10, 10])
        ax.set_ylim([-4, 4])
        ax.scatter(datat[:, 0], datat[:, 1])
        ax.quiver(datat[:, 0], datat[:, 1], datat[:, 2], datat[:, 3])

    ani = FuncAnimation(fig, animate, frames=nsaves, interval=10, repeat=False)
    ani.save("plots/animation.gif", dpi=300, writer=PillowWriter(fps=60))

    ### Set up density calculations
    resolution = 0.5
    width_grid = jnp.arange(-width, width, resolution) + resolution / 2
    height_grid = jnp.arange(-height, height, resolution) + resolution / 2
    xgrid, ygrid = jnp.meshgrid(width_grid, height_grid)
    gridpositions = jnp.dstack(jnp.meshgrid(width_grid, height_grid)).reshape(
        len(height_grid), len(width_grid), 2
    )

    # Define indices of gridpositions used in "middle box" density calculation
    width_box = 2.0  # metres either side of centre
    area_box = width_box * height
    box_grid_size = (
        len(height_grid),
        len(width_grid[jnp.abs(width_grid) <= width_box]),
    )
    box_grid_start = (
        len(width_grid) - len(width_grid[jnp.abs(width_grid) <= width_box])
    ) // 2
    box_grid_end = box_grid_start + box_grid_size[1]

    kernel = lambda x: transformations.tricube(x, shape=1.5)

    true_data = vmap(transformations.grid_density, (None, None, 0))(
        kernel, gridpositions, sol.ys[:, :, 0:2]
    )

    def get_ts(sol, tspan):
        return vmap(jnp.logical_and)(tspan[0] <= sol.ts, sol.ts <= tspan[1])

    def sol_to_box_density(sol, ts):
        density_data = vmap(transformations.grid_density, (None, None, 0))(
            transformations.tricube, gridpositions, sol.ys[ts, :, 0:2]
        )
        box_density = vmap(lambda x: x[:, box_grid_start:box_grid_end])(density_data)
        return jnp.mean(box_density)

    steady_state_ts = get_ts(sol, (4, 6))
    sol_to_box_density(sol, steady_state_ts)

    # TODO: Write a function that takes in parameters and outputs a box density,
    # then vmap this function over some values to product different density/speed values.
    # TODO: Determine what conditions result in a steady state including length of
    # simulation time, length of box, etc.
    # TODO: Use GPJax to create a GP fundamental diagram
    # TODO: Recreate the paper or discuss in meeting how best to do so.
