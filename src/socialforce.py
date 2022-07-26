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

config.update("jax_debug_nans", True)
from matplotlib.animation import FuncAnimation, PillowWriter

key = random.PRNGKey(42)


def V(distance, strength, shape):
    return strength * jnp.exp(-distance / shape)


def VPrime(distance, strength, shape):
    return -V(distance, strength, shape) / shape


def pedestrian_repulsion(xi, xj, strength, shape):
    Delta_x = xj - xi
    distance = jnp.linalg.norm(xj - xi)
    v = VPrime(distance, strength, shape) / distance
    return v * Delta_x


def two_groups_datum(key, N, D=5, V=2):
    A = N // 2
    B = N - A
    x1 = V * (2 * random.uniform(key, (A, 1)) - 1) - D
    x2 = V * (2 * random.uniform(key, (B, 1)) - 1) + D
    x = jnp.row_stack((x1, x2))
    y = V * (2 * random.uniform(key, (N, 1)) - 1)
    v1 = jnp.ones((A, 1))
    v2 = -jnp.ones((B, 1))
    v = jnp.row_stack((v1, v2))
    w = jnp.zeros((N, 1))
    xs = jnp.column_stack((x, y, v, w))
    return xs


def relax_to_desired(current_velocity, desired_velocity, relaxation_time):
    return (desired_velocity - current_velocity) / relaxation_time


vrelax_to_desired = vmap(relax_to_desired, (0, 0, None))


def total_pedestrian_repulsion(xi, xs, strength, shape):
    forces = vmap(
        pedestrian_repulsion,
        (None, 0, None, None),
    )(xi, xs, strength, shape)
    return jnp.sum(forces, 0, where=jnp.invert(jnp.isnan(forces)))


def pairwise_total_pedestrian_repulsion(xs, strength, shape):
    return vmap(total_pedestrian_repulsion, (0, None, None, None))(
        xs, xs, strength, shape
    )


def closest_point_to_segment(point, segment_start, segment_end):
    segment = segment_end - segment_start
    t0 = jnp.clip(
        (jnp.dot((point - segment_start), segment)) / jnp.dot(segment, segment), 0, 1
    )
    return segment_start + t0 * segment


def closest_point_to_vsegment(point, vsegment):
    return closest_point_to_segment(point, vsegment[0:2], vsegment[2:4])


def get_closest_wall(xi, walls):
    closest_points = vmap(closest_point_to_vsegment, (None, 0))(xi, walls)
    distances = vmap(jnp.linalg.norm)(closest_points)
    return closest_points[jnp.argmin(distances), :]


def total_wall_repulsion(xi, walls, strength, shape):
    xs = vmap(closest_point_to_vsegment, (None, 0))(xi, walls)
    forces = vmap(
        pedestrian_repulsion,
        (None, 0, None, None),
    )(xi, xs, strength, shape)
    return jnp.sum(forces, 0, where=jnp.invert(jnp.isnan(forces)))


def pairwise_total_wall_repulsion(xs, walls, strength, shape):
    return vmap(total_wall_repulsion, (0, None, None, None))(xs, walls, strength, shape)


def step_social_force(t, y, args):
    (
        pedestrian_strength,
        pedestrian_shape,
        wall_strength,
        wall_shape,
        relaxation_time,
        desired_velocities,
        walls,
    ) = args

    ### Relaxation to desired velocity
    current_velocities = y[:, 2:4]
    accelerations = vmap(relax_to_desired, (0, 0, None))(
        current_velocities, desired_velocities, relaxation_time
    )

    ### Agent repulsion
    pedestrian_repulsions = pairwise_total_pedestrian_repulsion(
        y[:, 0:2], pedestrian_strength, pedestrian_shape
    )

    ### Wall repulsion (pairwise meaning every pedestrian with every wall)
    wall_repulsions = pairwise_total_wall_repulsion(
        y[:, 0:2], walls, wall_strength, wall_shape
    )

    ### Derivative of position equals velocity
    dxdy = y[:, 2:4]
    dvdy = accelerations + pedestrian_repulsions + wall_repulsions
    return jnp.column_stack((dxdy, dvdy))


def total_density(gridposition, agentpositions, shape=jnp.diag(jnp.ones(2))):
    return jnp.sum(
        vmap(jax.scipy.stats.multivariate_normal.pdf, (None, 0, None))(
            gridposition, agentpositions, shape
        )
    )


def grid_density(gridpositions, xs):
    return vmap(vmap(total_density, (0, None)), (0, None))(gridpositions, xs)


if __name__ == "__main__":
    N = 40
    width = 5
    height = 3
    x0 = two_groups_datum(key, N, width, 0.66 * height)
    pedestrian_strength = 2.1
    pedestrian_shape = 1.0
    wall_strength = 10.0
    wall_shape = 2.5
    relaxation_time = 0.5
    desired_velocities = x0[:, 2:4]
    top_wall = jnp.array([-100, height, 100, height])
    bottom_wall = jnp.array([-100, -height, 100, -height])
    walls = jnp.row_stack((top_wall, bottom_wall))

    pairwise_total_wall_repulsion(x0[:, 0:2], walls, wall_strength, wall_shape)

    args = (
        pedestrian_strength,
        pedestrian_shape,
        wall_strength,
        wall_shape,
        relaxation_time,
        desired_velocities,
        walls,
    )

    step_social_force(0.0, x0, args)

    term = ODETerm(step_social_force)
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
    datat = sol.ys[1]

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

    # ani = FuncAnimation(fig, animate, frames=nsaves, interval=10, repeat=False)
    # ani.save("plots/animation.gif", dpi=300, writer=PillowWriter(fps=60))

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

    true_data = vmap(grid_density, (None, 0))(gridpositions, sol.ys[:, :, 0:2])

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
        return vmap(grid_density, (None, 0))(gridpositions, sol.ys[:, :, 0:2])

    def compute_loss(params, y):
        pedestrian_strength, pedestrian_shape = params
        y_pred = predict(pedestrian_strength, pedestrian_shape)
        loss = jnp.mean(optax.l2_loss(y_pred, y))
        return loss

    jax.grad(compute_loss)(params, true_data)

    for _ in range(1000):
        grads = jax.grad(compute_loss)(params, true_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
