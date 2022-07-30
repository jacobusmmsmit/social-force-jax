# from jax.config import config

# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import pytest

from jax import random, lax
from jax import vmap
from jax.config import config

from diffrax import (
    diffeqsolve,
    Tsit5,
    Kvaerno3,
    ODETerm,
    SaveAt,
    PIDController,
)

import src.socialforce as sf
from src import transformations

config.update("jax_debug_nans", False)

key = random.PRNGKey(42)


def no_nan_gradients():
    N = 40
    width = 5.0
    height = 3.0
    x0 = sf.two_groups_datum(key, N, width, 0.66 * height)
    xs = x0[:, 0:2]
    pedestrian_strength = 2.1
    pedestrian_shape = 1.0
    wall_strength = 10.0
    wall_shape = 2.5
    relaxation_time = 0.5
    desired_velocities = x0[:, 2:4]
    top_wall = jnp.array([-100, height, 100, height])
    bottom_wall = jnp.array([-100, -height, 100, -height])
    walls = jnp.row_stack((top_wall, bottom_wall))

    # Acceleration gradients
    current_vel = x0[0, 2:4]
    good_vel = current_vel + jnp.array([1.0, 1.0])  # as in shouldn't cause NaN
    bad_vel = current_vel
    relax_good_current_grad, relax_good_target_grad, relax_good_time_grad = jax.jacrev(
        sf.relax_to_desired, (0, 1, 2)
    )(current_vel, good_vel, relaxation_time)
    assert ~jnp.all(jnp.isnan(relax_good_current_grad))
    assert ~jnp.all(jnp.isnan(relax_good_target_grad))
    assert ~jnp.all(jnp.isnan(relax_good_time_grad))

    relax_bad_current_grad, relax_bad_target_grad, relax_bad_time_grad = jax.jacrev(
        sf.relax_to_desired, (0, 1, 2)
    )(current_vel, bad_vel, relaxation_time)
    assert ~jnp.all(jnp.isnan(relax_bad_current_grad))
    assert ~jnp.all(jnp.isnan(relax_bad_target_grad))
    assert ~jnp.all(jnp.isnan(relax_bad_time_grad))

    # vrelax_to_desired is just a vmap of relax_to_desired so I skip

    # Pedestrian repulsion gradients
    ped_ped_dist = 0.0
    V_dist_grad, V_str_grad, V_shape_grad = jax.jacrev(sf.V, (0, 1, 2))(
        ped_ped_dist, pedestrian_strength, pedestrian_shape
    )
    assert ~jnp.isnan(V_dist_grad)
    assert ~jnp.isnan(V_str_grad)
    assert ~jnp.isnan(V_shape_grad)

    VPrime_dist_grad, VPrime_str_grad, VPrime_shape_grad = jax.jacrev(sf.V, (0, 1, 2))(
        0.1, pedestrian_strength, pedestrian_shape
    )
    assert ~jnp.isnan(VPrime_dist_grad)
    assert ~jnp.isnan(VPrime_str_grad)
    assert ~jnp.isnan(VPrime_shape_grad)

    # return jax.jacrev(sf.pedestrian_repulsion)(
    #     xs[0], xs[0], pedestrian_strength, pedestrian_shape
    # )

    # return jax.jacrev(sf.pedestrian_repulsion)(
    #     xs[0], xs[0], pedestrian_strength, pedestrian_shape
    # )
    # return sf.total_pedestrian_repulsion(
    #     xs[0], xs, pedestrian_strength, pedestrian_shape
    # )
    tpr_xi_grad, tpr_xs_grad, tpr_str_grad, tpr_shape_grad = jax.jacrev(
        sf.total_pedestrian_repulsion, (0, 1, 2, 3)
    )(xs[0], xs, pedestrian_strength, pedestrian_shape)

    assert ~jnp.all(jnp.isnan(tpr_xi_grad))
    assert ~jnp.all(jnp.isnan(tpr_xs_grad))
    assert ~jnp.all(jnp.isnan(tpr_str_grad))
    assert ~jnp.all(jnp.isnan(tpr_shape_grad))

    ptpr_xs_grad, ptpr_str_grad, ptpr_shape_grad = jax.jacrev(
        sf.pairwise_total_pedestrian_repulsion, (0, 1, 2)
    )(xs, pedestrian_strength, pedestrian_shape)

    assert ~jnp.all(jnp.isnan(ptpr_xs_grad))
    assert ~jnp.all(jnp.isnan(ptpr_str_grad))
    assert ~jnp.all(jnp.isnan(ptpr_shape_grad))

    # Wall force grads
    point = jnp.array([0.0, 2.9])  # if a point touches a segment, we get `nan`s
    points = jnp.array([[0.0, 2.9], [1.0, 0.9]])
    segment = jnp.array([-1.0, 1.0, 2.0, 1.0])
    vseg_point_grad, vseg_segment_grad = jax.jacrev(
        sf.closest_point_to_vsegment, (0, 1)
    )(point, segment)

    assert ~jnp.all(jnp.isnan(vseg_point_grad))
    assert ~jnp.all(jnp.isnan(vseg_segment_grad))
    # return (point, walls, wall_strength, wall_shape)
    # return jax.jacrev(sf.total_wall_repulsion)(point, walls, wall_strength, wall_shape)
    # return jax.jacrev(sf.pairwise_total_wall_repulsion)(
    #     points, walls, wall_strength, wall_shape
    # )


def physical_laws_obeyed():
    N = 40
    width = 5.0
    height = 3.0
    x0 = sf.two_groups_datum(key, N, width, 0.66 * height)
    xs = x0[:, 0:2]
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

    ### Set-up and solve model
    term = ODETerm(sf.step)
    solver = Kvaerno3()
    tspan = (0.0, 50.0)
    nsaves = 1200
    saveat = SaveAt(ts=jnp.linspace(tspan[0], tspan[1], nsaves))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

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

    assert jnp.all(sol.ys[:, :, 1] > -height)
    assert jnp.all(sol.ys[:, :, 1] < height)


no_nan_gradients()
physical_laws_obeyed()
