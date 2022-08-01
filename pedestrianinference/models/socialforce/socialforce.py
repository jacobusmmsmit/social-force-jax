import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import lax
from jax import vmap, jit
from jax.config import config

# from utils import vectorized_cond


@jit
def relax_to_desired(current_velocity, desired_velocity, relaxation_time):
    return (desired_velocity - current_velocity) / relaxation_time


vrelax_to_desired = vmap(relax_to_desired, (0, 0, None))


@jit
def V(distance, strength, shape):
    return strength * jnp.exp(-distance / shape)


@jit
def VPrime(distance, strength, shape):
    return -V(distance, strength, shape) / shape


@jit
def pedestrian_repulsion(xi, xj, strength, shape):
    Delta_x = xj - xi
    # jnp.clip(d, 0.1, 10.0)
    distance = jnp.linalg.norm(Delta_x)
    # distance = jnp.power(jnp.power(Delta_x[0], 2) + jnp.power(Delta_x[1], 2), 1 / 2)
    # distance = 1.0
    v = VPrime(distance, strength, shape) / (distance + 0.1)
    # return lax.cond(
    #     jnp.alltrue(Delta_x == 0), lambda x: v * x, lambda x: jnp.zeros(2), Delta_x
    # )
    return Delta_x * v
    # return (distance, strength, shape)


# def total_pedestrian_repulsion(xi, xs, strength, shape):
#     forces = vmap(
#         pedestrian_repulsion,
#         (None, 0, None, None),
#     )(xi, xs, strength, shape)
#     return jnp.sum(forces, 0, where=jnp.invert(jnp.isnan(forces)))


@jit
def select_different(point, points):
    return points[
        jnp.where(
            ~vmap(jnp.array_equal, (0, None))(points, point),
            size=len(points) - 1,
            fill_value=0,
        )
    ]


@jit
def total_pedestrian_repulsion(xi, xs, strength, shape):
    xs_without_xi = select_different(xi, xs)
    forces = vmap(
        pedestrian_repulsion,
        (None, 0, None, None),
    )(xi, xs_without_xi, strength, shape)
    return jnp.sum(forces, 0)


@jit
def pairwise_total_pedestrian_repulsion(xs, strength, shape):
    return vmap(total_pedestrian_repulsion, (0, None, None, None))(
        xs, xs, strength, shape
    )


@jit
def closest_point_to_segment(point, segment_start, segment_end):
    segment = segment_end - segment_start
    t0 = jnp.clip(
        (jnp.dot((point - segment_start), segment)) / jnp.dot(segment, segment), 0, 1
    )
    return segment_start + t0 * segment


@jit
def closest_point_to_vsegment(point, vsegment):
    return closest_point_to_segment(point, vsegment[0:2], vsegment[2:4])


@jit
def get_closest_wall(xi, walls):
    closest_points = vmap(closest_point_to_vsegment, (None, 0))(xi, walls)
    distances = vmap(jnp.linalg.norm)(closest_points)
    return closest_points[jnp.argmin(distances), :]


def in_segment(pt, seg, epsilon=1e-6):
    a, b = pt - seg[0:2], pt - seg[2:4]
    cp = jnp.cross(a, b)
    dp = jnp.dot(a, b)
    return abs(cp) <= epsilon and dp > 0 and jnp.linalg.norm(a, b) < dp


@jit
def total_wall_repulsion(xi, walls, strength, shape):
    # doesn't work because dynamic sized walls
    # walls = walls[jnp.where(~vmap(in_segment, (None, 0))(xi, walls), size=len(walls))]
    xs = vmap(closest_point_to_vsegment, (None, 0))(xi, walls)
    forces = vmap(
        pedestrian_repulsion,
        (None, 0, None, None),
    )(xi, xs, strength, shape)
    return jnp.sum(forces, 0, where=jnp.invert(jnp.isnan(forces)))


@jit
def pairwise_total_wall_repulsion(xs, walls, strength, shape):
    return vmap(total_wall_repulsion, (0, None, None, None))(xs, walls, strength, shape)


@jit
def step(t, y, args):
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
