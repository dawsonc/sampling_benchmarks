import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from sampling_benchmarks.benchmarks import Benchmark


class Quadratic(Benchmark):
    """
    Defines a quadratic potential for testing.

    U(x) = x.T x / dim(x)

    Uses a Gaussian prior for x.

    args:
        dimension: problem dimension
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
        super().__init__(dimension, initial_guess, initial_guess_std)
        self.name = f"Quadratic (n={self.dim})"

    @jaxtyped
    @beartype
    def u(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute the potential given the decision variables x."""
        return jnp.dot(x, x) / self.dim  # normalize by dimension

    @property
    @jaxtyped
    @beartype
    def global_minimum(self) -> Float[Array, ""]:
        """Return the global minimum of the potential"""
        return jnp.array(0.0)
