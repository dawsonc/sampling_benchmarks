"""Implement a Blackjax sampler implementing INFORMAL HMC"""
from typing import Callable, Tuple, Union, NamedTuple

import blackjax.mcmc as mcmc
import jax
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm  # TODO upgrade to MCMCSamplingAlgorithm
from blackjax.types import PRNGKey
from jaxtyping import Array, Float, PyTree

from sampling_benchmarks.samplers.informal.integrator import (
    informal_integrator,
    JacobianTrackingIntegratorState,
)


# We start by implementing the Blackjax interface, which wraps the init and kernel
# closures that we define below.
class informal_hmc:
    """
    Implements the Blackjax interface for the INFORMAL HMC kernel.

    Based on the HMC kernel implementation in the Blackjax library

    args:
    logprob_fn: The logprobability density function we wish to draw samples from. This
        is minus the potential function. Currnetly only supports logprobs with vector
        input (no general PyTree shenanigans).
    step_size: The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a
        value for the momentum and computing the kinetic energy.
    max_lipschitz: An upper bound on the Lipschitz constant for the potential
        function within each piecewise continuous region.
    num_integration_steps: The number of steps we take with the symplectic integrator
        at each sample step before returning a sample.
    divergence_threshold: The absolute value of the difference in energy between two
        states above which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    def __new__(  # type: ignore[misc]
        cls,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Float[Array, " n"],
        max_lipschitz: float,
        num_integration_steps: int,
        divergence_threshold: int = 1000,
    ) -> SamplingAlgorithm:
        integrator = informal_integrator
        step = informal_hmc_kernel(integrator, divergence_threshold, max_lipschitz)

        def init_fn(position: PyTree):
            # We can just re-use the standard HMC initialization function
            return mcmc.hmc.init(position, logprob_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(
                rng_key,
                state,
                logprob_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)


def informal_hmc_kernel(
    integrator: Callable,
    divergence_threshold: float,
    max_lipschitz: float,
):
    """
    Build a kernel for INFORMAL HMC.

    Developed from the vanilla HMC kernal in Blackjax.

    args:
        integrator: The symplectic integrator to use to integrate the Hamiltonian
            dynamics.
        divergence_threshold: Value of the difference in energy above which we consider
            that the transition is divergent.
        max_lipschitz: rate of change of the potential above which we assume that
            we have passed a discontinuity.

    returns:
        A kernel that takes a rng_key and a Pytree that contains the current state
        of the chain and that returns a new state of the chain along with
        information about the transition.
    """

    def one_step(
        rng_key: PRNGKey,
        state: mcmc.hmc.HMCState,
        logprob_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
    ) -> Tuple[mcmc.hmc.HMCState, mcmc.hmc.HMCInfo]:
        """
        Generate a new sample with the HMC kernel using the INFORMAL algorithm
        """

        def potential_fn(x):
            return -logprob_fn(x)

        momentum_generator, kinetic_energy_fn, _ = mcmc.metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
        nearly_symplectic_integrator = integrator(
            potential_fn, kinetic_energy_fn, max_lipschitz
        )
        proposal_generator = informal_hmc_proposal(
            nearly_symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, potential_energy, potential_energy_grad = state
        momentum = momentum_generator(key_momentum, position)

        integrator_state = JacobianTrackingIntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = mcmc.hmc.HMCState(
            proposal.position, proposal.potential_energy, proposal.potential_energy_grad
        )

        return proposal, info

    return one_step


def informal_hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, PyTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = mcmc.proposal.static_binomial_sampling,
) -> Callable:
    """INFORMAL HMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. The acceptance probability is modified to account for the jacobian
    of the integration (allowing a non-volume-preserving integration).

    Based on the code in Blackjax.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.
    Returns
    -------
    A kernel that generates a new chain state and information about the transition.
    """
    build_trajectory = mcmc.trajectory.static_integration(integrator)
    init_proposal, generate_proposal = informal_proposal_generator(
        kinetic_energy, divergence_threshold
    )

    def generate(
        rng_key, state: JacobianTrackingIntegratorState
    ) -> Tuple[JacobianTrackingIntegratorState, mcmc.hmc.HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        # We don't need to flip the momentum in practice
        # end_state = mcmc.hmc.flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = mcmc.hmc.HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def informal_proposal_generator(
    kinetic_energy: Callable, divergence_threshold: float
) -> Tuple[Callable, Callable]:
    def new(state: JacobianTrackingIntegratorState) -> mcmc.proposal.Proposal:
        energy = state.potential_energy + kinetic_energy(state.momentum)
        return mcmc.proposal.Proposal(state, energy, 0.0, -jnp.inf)

    def update(
        initial_energy: float, state: JacobianTrackingIntegratorState
    ) -> Tuple[mcmc.proposal.Proposal, bool]:
        """
        Generate a new proposal from a trajectory state.

        The trajectory state records information about the position in the state
        space and corresponding potential energy. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous states
        as well as the current state.
        Parameters
        ----------
        initial_energy:
            The initial energy.
        state:
            The new state.
        """
        new_energy = state.potential_energy + kinetic_energy(state.momentum)

        delta_energy = initial_energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        J = state.accumulated_jacobian_determinant

        # Probability of acceptance is J * exp(H0 - H(z_new))

        # Weight is log probability  TODO fixme
        weight = delta_energy + jnp.log(J)
        sum_log_p_accept = jnp.minimum(delta_energy + jnp.log(J), 0.0)

        return (
            mcmc.proposal.Proposal(
                state,
                new_energy,
                weight,
                sum_log_p_accept,
            ),
            is_transition_divergent,
        )

    return new, update
