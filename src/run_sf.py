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
    warnings.filterwarnings("ignore",category=DeprecationWarning)
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

    ### Set-up and solve model
    term = ODETerm(sf.step)
    solver = Tsit5()
    tspan = (0.0, 10.0)
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
            ax.plot(wall[0], wall[1], wall[2], wall[3], color="black")
        datat = sol.ys[i]
        ax.set_xlim([-10, 10])
        ax.set_ylim([-4, 4])
        ax.scatter(datat[:, 0], datat[:, 1])
        ax.quiver(datat[:, 0], datat[:, 1], datat[:, 2], datat[:, 3])

    ani = FuncAnimation(fig, animate, frames=nsaves, interval=10, repeat=False)
    ani.save("plots/animation.gif", dpi=300, writer=PillowWriter(fps=60))

    ### Set up density calculations
    resolution = 0.125
    width_grid = jnp.arange(-width, width, resolution) + resolution / 2
    height_grid = jnp.arange(-height, height, resolution) + resolution / 2
    xgrid, ygrid = jnp.meshgrid(width_grid, height_grid)
    gridpositions = jnp.dstack(jnp.meshgrid(width_grid, height_grid)).reshape(
        len(height_grid), len(width_grid), 2
    )

    kernel = lambda x: transformations.tricube(x, shape=1.5)

    # Plot a heatmap of the initial conditions (broken, need to transpose)
    # xs = sol.ys[0][:, 0:2]
    # a = transformations.grid_density(transformations.tricube, gridpositions, xs)
    # plt.clf()
    # plt.imshow(jnp.transpose(a), cmap="hot", interpolation="nearest")

    true_data = vmap(transformations.grid_density, (None, None, 0))(
        kernel, gridpositions, sol.ys[:, :, 0:2]
    )

    ### Save solution to csv
    sol_numpy = np.asarray(true_data)
    print(f"True dimensions: {np.shape(sol_numpy)}")
    reshaped_density = sol_numpy.reshape(sol_numpy.shape[0], -1)
    np.savetxt(f"data/reshaped_density.csv", reshaped_density, delimiter=",")

    @jax.jit
    def predict(pedestrian_strength, pedestrian_shape, wall_strength, wall_shape):
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
        return vmap(transformations.grid_density, (None, None, 0))(
            kernel, gridpositions, sol.ys[:, :, 0:2]
        )

    @jax.jit
    def compute_loss(params, y):
        pedestrian_strength, pedestrian_shape, wall_strength, wall_shape = params
        y_pred = predict(
            pedestrian_strength, pedestrian_shape, wall_strength, wall_shape
        )
        loss = jnp.mean(optax.l2_loss(y_pred, y))
        return loss

    ### Point-estimate Parameter Calibration with ADAM
    learning_rate = 0.3
    optimizer = optax.adam(learning_rate)
    params = jnp.array(
        [1.0, 0.5, 0.5, 0.5]
    )  # Initialise parameters at arbitrary values
    opt_state = optimizer.init(params)

    # Train:
    for _ in range(1000):
        grads = jax.grad(compute_loss)(params, true_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Printing the loss is very inefficient, speed up the training by removing this line
        # print(f"Parameters: {params}, Loss: {compute_loss(params, true_data)}")
        print(f"Parameters: {params}, Grads: {grads}")

    # Final parameters:
    print(f"Final parameters, {params}")

    ### Bayesian Parameter Calibration with NUTS
    uFactor = 2.0  # Increase to make inference have a wider space to explore

    def fitSF(data):
        sigma_hyper = numpyro.sample("sigma_hyper", dist.InverseGamma(2, 3))
        # tao = numpyro.sample('tao', dist.Uniform(relaxation_time/uFactor, relaxation_time*uFactor))
        V0 = numpyro.sample(
            "V0",
            dist.Uniform(pedestrian_strength / uFactor, pedestrian_strength * uFactor),
        )
        sigma = numpyro.sample(
            "sigma",
            dist.Uniform(pedestrian_shape / uFactor, pedestrian_shape * uFactor),
        )
        U0 = numpyro.sample(
            "U0",
            dist.Uniform(wall_strength / uFactor, wall_strength * uFactor),
        )
        r = numpyro.sample(
            "r",
            dist.Uniform(wall_shape / uFactor, wall_shape * uFactor),
        )

        predicted = predict(V0, sigma, U0, r)

        numpyro.sample("obs", dist.Normal(predicted, sigma_hyper), obs=data)

    ### NUTS ###
    # Set random seed for reproducibility.
    nsamples = 200
    rng_key = random.PRNGKey(0)
    nuts = MCMC(
        NUTS(fitSF, target_accept_prob=0.65, max_tree_depth=10),
        num_samples=nsamples,
        num_warmup=50,
    )
    nuts.run(rng_key, true_data)  # run sampler.
    nuts_samples = nuts.get_samples()  ## collect samplers.

    def plot_chain(samples):
        fig, axs = plt.subplots(len(samples), 2)
        xs = jnp.arange(nsamples)
        for i, variable in enumerate(samples):
            row = axs[i]
            print(row)
            sns.lineplot(xs, samples[variable], ax=row[0])
            sns.kdeplot(samples[variable], ax=row[1])
            row[0].set_ylabel(variable)

        axs[0][0].set_title("Trace Plots")
        axs[0][1].set_title("Density Plots")
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.suptitle("Trace and Density Plots for NUTS")
        plt.savefig("plots/samples.png")

    plot_chain(nuts_samples)
