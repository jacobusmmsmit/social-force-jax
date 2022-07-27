import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

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

import src.socialforce as sf
from src import transformations

config.update("jax_debug_nans", True)

key = random.PRNGKey(42)

if __name__ == "__main__":
    N = 40
    width = 5
    height = 3
    x0 = sf.two_groups_datum(key, N, width, 0.66 * height)
    pedestrian_strength = 2.1
    pedestrian_shape = 1.0
    wall_strength = 10.0
    wall_shape = 2.5
    relaxation_time = 0.5
    desired_velocities = x0[:, 2:4]
    top_wall = jnp.array([-100, height, 100, height])
    bottom_wall = jnp.array([-100, -height, 100, -height])
    walls = jnp.row_stack((top_wall, bottom_wall))

    args = (
        pedestrian_strength,
        pedestrian_shape,
        wall_strength,
        wall_shape,
        relaxation_time,
        desired_velocities,
        walls,
    )

    sf.step(0.0, x0, args)

    term = ODETerm(sf.step)
    solver = Tsit5()
    tspan = (0.0, 15.0)
    nsaves = 50
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

    fig, ax = plt.subplots(1, 1)

    def animate(i):
        ax.clear()
        for wall in walls:
            ax.plot(wall[0], wall[1], wall[2], wall[3], color="black")
        datat = sol.ys[i]
        ax.set_xlim([-10, 10])
        ax.set_ylim([-4, 4])
        ax.scatter(datat[:, 0], datat[:, 1])
        ax.quiver(datat[:, 0], datat[:, 1], datat[:, 2], datat[:, 3])

    ani = FuncAnimation(fig, animate, frames=nsaves, interval=10, repeat=False)
    ani.save("plots/animation.gif", dpi=300, writer=PillowWriter(fps=60))

    resolution = 1.0
    width_grid = jnp.arange(-width, width, resolution) + resolution / 2
    height_grid = jnp.arange(-height, height, resolution) + resolution / 2
    xgrid, ygrid = jnp.meshgrid(width_grid, height_grid)
    gridpositions = jnp.dstack(jnp.meshgrid(width_grid, height_grid)).reshape(
        len(width_grid), len(height_grid), 2
    )

    # xs = sol.ys[0][:, 0:2]
    # a = grid_density(gridpositions, xs)
    # plt.clf()
    # plt.imshow(jnp.transpose(a), cmap='hot', interpolation='nearest')
    # ax.scatter(xs[:, 0], xs[:, 1])
    # ax.quiver(xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3])
    # plt.savefig("plots/heatmap3.png")

    true_data = vmap(transformations.grid_density, (None, 0))(
        gridpositions, sol.ys[:, :, 0:2]
    )

    learning_rate = 0.1
    optimizer = optax.adam(learning_rate)
    params = jnp.array([1.0, 0.5])
    opt_state = optimizer.init(params)

    def predict(pedestrian_strength, pedestrian_shape):
        new_args = (
            pedestrian_strength,
            pedestrian_shape,
            wall_strength,
            wall_shape,
            relaxation_time,
            desired_velocities,
            walls,
        )
        sol = diffeqsolve(
            term,
            solver,
            t0=tspan[0],
            t1=tspan[1],
            dt0=0.1,
            y0=x0,
            args=new_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        return vmap(transformations.grid_density, (None, 0))(
            gridpositions, sol.ys[:, :, 0:2]
        )

    def compute_loss(params, y):
        pedestrian_strength, pedestrian_shape = params
        y_pred = predict(pedestrian_strength, pedestrian_shape)
        loss = jnp.mean(optax.l2_loss(y_pred, y))
        return loss

    # jax.grad(compute_loss)(params, true_data)
    jax.grad(compute_loss)(jnp.array([2.1, 1.0]), true_data)

    # Train:
    for _ in range(200):
        grads = jax.grad(compute_loss)(params, true_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Printing the loss is very inefficient, speed up the training by removing this line
        print(f"Parameters: {params}, Loss: {compute_loss(params, true_data)}")

    # Final parameters:
    print(f"Final parameters, {params}")
