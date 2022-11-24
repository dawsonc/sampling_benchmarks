import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from sampling_benchmarks.benchmarks import Benchmark


class Heaviside(Benchmark):
    """
    Defines a Heaviside potential.

    U(x) = mean(H(x))

    where H(x) = 0 if x < 0, 0.5 if x == 0, and 1 if x > 0 is the Heaviside step
    function.

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
        self.name = f"Heaviside (n={self.dim})"

    @jaxtyped
    @beartype
    def u(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute the potential given the decision variables x."""
        return jnp.heaviside(x, 0.5).mean()

    @property
    @jaxtyped
    @beartype
    def global_minimum(self) -> Float[Array, ""]:
        """Return the global minimum of the potential"""
        return jnp.array(0.0)
