from abc import ABC, abstractmethod, abstractproperty

import jax
import jax.numpy as jnp
from beartype import beartype
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Float, jaxtyped


class Benchmark(ABC):
    """
    Defines a generic benchmark for optimization-as-sampling problems.

    By "optimization-as-sampling problems", we mean problems where we reformulate an
    optimization

        min_x U(x)

    as a sampling problem, where we wish to sample from the probability distribution

        p(x) \propto e^{-U(x)}

    All benchmarks need to implement 3 methods:
        1. `u(x)`: compute the potential as a function of decision variables
        2. `prior_logprob(x)`: compute the prior log-probability of decision variables x
        3. `sample_prior()`: return a sample from the prior distribution over x

    They also need to implement one property:
        3. `global_minimum`: the ground-truth global optimum of min_x U(x)

    By default, the prior distribution is a Gaussian with diagonal covariance and mean
    specified as parameters at initialization (this means that `prior_logprob` and
    `sample_prior` are implemented already if you're OK with a Gaussian prior).

    args:
        dimension: problem dimension (use this many parallel examples)
        initial_guess: mean of prior distribution
        initial_guess_std: standard deviation of prior distribution
    """

    name: str

    @jaxtyped
    @beartype
    def __init__(
        self,
        dimension: int,
        initial_guess: Float[Array, " dim"],
        initial_guess_std: Float[Array, ""],
    ) -> None:
        self.dim = dimension
        self.name = "Generic Benchmark"
        self.initial_guess = initial_guess
        self.initial_cov = initial_guess_std**2 * jnp.eye(self.dim)

    @abstractmethod
    def u(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute the potential given the decision variables x."""
        pass

    @jaxtyped
    @beartype
    def prior_logprob(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute the prior log probability of decision variables x."""
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, self.initial_guess, self.initial_cov
        )

    @jaxtyped
    @beartype
    def sample_prior(self, key: PRNGKeyArray) -> Float[Array, " dim"]:
        """Sample a value of x from the prior distribution."""
        return jax.random.multivariate_normal(key, self.initial_guess, self.initial_cov)

    @abstractproperty
    def global_minimum(self) -> Float[Array, ""]:
        """Return the global minimum of the potential"""
        pass
