"""Implement a jacobian-tracking reflection/refraction integrator for INFORMAL HMC"""
from typing import Callable, NamedTuple

import blackjax.mcmc as mcmc
import jax
import jax.numpy as jnp
from blackjax.mcmc.metrics import EuclideanKineticEnergy
from jaxtyping import Array, Float


class JacobianTrackingIntegratorState(NamedTuple):
    """
    State of the trajectory integration.

    In addition to basic features, also keeps track of the determinant of the
    Jacobian of the overall integration (used to correct the acceptance probability).

    For simplicity, we currently only support positions (and momenta) that are one
    dimensional jax vectors
    """

    position: Float[Array, " n"]
    momentum: Float[Array, " n"]
    potential_energy: float
    potential_energy_grad: Float[Array, " n"]
    accumulated_jacobian_determinant: float = 1.0


JacobianTrackingEuclideanIntegrator = Callable[
    [JacobianTrackingIntegratorState, float], JacobianTrackingIntegratorState
]


def new_integrator_state(potential_fn, position, momentum):
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return JacobianTrackingIntegratorState(
        position, momentum, potential_energy, potential_energy_grad
    )


def informal_integrator(
    potential_fn: Callable,
    kinetic_energy_fn: EuclideanKineticEnergy,
    max_lipschitz: float,
) -> mcmc.integrators.EuclideanIntegrator:
    """
    Implement oracle-free Fixed-Orientation Momentum Adjusting Leapfrog integration.

    args:
        potential_fn: the potential function of the Hamiltonian dynamics
        kinetic_energy_fn: the kinetic energy function of the Hamiltonian dynamics
        max_lipschitz: an upper bound on the lipschitz constant inside continuous
            domains.
    """
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    # Define the one-step update function that also tracks the accumulated
    # determinant of the Jacobian
    def one_step(
        state: JacobianTrackingEuclideanIntegrator, step_size: float
    ) -> JacobianTrackingEuclideanIntegrator:
        position, momentum, potential_energy_prev, potential_energy_grad, det_J = state

        # First half-step for momentum
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        # Now we need to figure out if there's a discontinuity between the current
        # position and the next position. We detect this by comparing the change in
        # potential between the two positions with the given Lipshitz constant

        # Update the position to get the next position
        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        # Get the potential at this new position (also get the gradient to use later)
        potential_energy, potential_energy_grad = potential_and_grad_fn(position)

        # Compare the change in potential with the Lipschitz constant
        delta_U = potential_energy - potential_energy_prev
        delta_position_norm = jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda s, x: s + jnp.linalg.norm(x) ** 2,
                a2 * step_size * kinetic_grad,
                initializer=0.0,
            )
        )

        # Adjust the momentum if the potential change is too big
        reflect = lambda p, _: -p
        refract = (
            lambda p, delta_U: p
            * jnp.sqrt(1.0 - 2 * delta_U / (1e-6 + jnp.linalg.norm(p) ** 2))
        )
        adjust = lambda p, delta_U: jax.lax.cond(
            jnp.linalg.norm(p) ** 2 > 2 * delta_U,
            refract,
            reflect,
            p,
            delta_U,
        )
        do_nothing = lambda p, _: p
        observed_lipschitz = jnp.abs(delta_U) / delta_position_norm
        momentum = jax.lax.cond(
            observed_lipschitz > max_lipschitz, adjust, do_nothing, momentum, delta_U
        )

        # Refraction is the only part of this procedure that is not a shear
        # transformation or reflection (i.e. the only bit where det(J) != 1.0),
        # so we can manually accumulate that determinant into the running product
        def det_J_refraction(p, delta_U):
            p_norm_2 = jnp.linalg.norm(p) ** 2
            c = jnp.sqrt(1 - 2 * delta_U / (1e-6 + p_norm_2))  # refraction scale factor
            grad_c_wrt_p = (
                2 * delta_U * p / (p_norm_2**2 * jnp.sqrt(1 - delta_U / p_norm_2))
            ).reshape(-1, 1)
            d_refract_dp = c * jnp.eye(p.shape[0]) + grad_c_wrt_p @ p.reshape(1, -1)
            return jnp.abs(jnp.linalg.det(d_refract_dp))

        det_J_adjustment = lambda p, delta_U: jax.lax.cond(
            jnp.linalg.norm(p) ** 2 > 2 * delta_U,
            det_J_refraction,
            lambda *_: 1.0,  # reflection has absolute value of determinant = 1.0
            p,
            delta_U,
        )
        det_J = det_J * jax.lax.cond(
            observed_lipschitz > max_lipschitz,
            det_J_adjustment,
            lambda *_: 1.0,
            momentum,
            delta_U,
        )

        # Take the final step size in momentum (from the current potential)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return JacobianTrackingIntegratorState(
            position, momentum, potential_energy, potential_energy_grad, det_J
        )

    return one_step
