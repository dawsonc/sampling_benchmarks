import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from sampling_benchmarks.benchmarks import Benchmark


class Ballistic(Benchmark):
    """
    Defines the potential for the ballistic trajectory optimization from Suh 2022.

    U(x) = -distance travelled by thrown ball, with an inelastic obstacle in the way

    Uses a Gaussian prior for x.

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
        super().__init__(dimension, initial_guess, initial_guess_std)
        self.name = f"Ballistic (n={self.dim})"
        self.v0 = 10.0
        self.dt = 0.01
        self.g = 9.81
        max_t_flight = self.v0 * 2 / self.g
        self.T = int(max_t_flight / self.dt + 1)

    @jaxtyped
    @beartype
    def u(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute the potential given the decision variables x."""
        # The decision variables are the launch angles of the balls.
        # Convert that into initial velocity
        px0 = jnp.zeros_like(x)
        py0 = jnp.zeros_like(x)
        vx0 = self.v0 * jnp.cos(x)
        vy0 = self.v0 * jnp.sin(x)

        # Simulate the trajectory of all of the balls in free flight,
        # but if any of them hit the wall, their x velocity is set to zero.
        # If any hit the ground, their x and y velocities are set to zero
        wall_x = 7.0
        wall_height = 2.5
        wall_width = 0.1

        def step_fn(carry, dummy_input):
            # Extract states
            x, y, vx, vy = carry

            # Update states
            x = x + self.dt * vx
            y = y + self.dt * vy
            vy = vy - self.dt * self.g

            # Check for collisions
            wall_collision = jnp.logical_and(
                jnp.abs(x - wall_x) <= wall_width / 2.0, y <= wall_height
            )
            ground_collision = y <= 0.0
            vx = jnp.where(wall_collision, jnp.zeros_like(vx), vx)
            vx = jnp.where(ground_collision, jnp.zeros_like(vx), vx)
            vy = jnp.where(ground_collision, jnp.zeros_like(vy), vy)

            # Update the carry
            carry = (x, y, vx, vy)
            return carry, carry

        _, (px, py, _, _) = jax.lax.scan(
            step_fn, (px0, py0, vx0, vy0), None, length=self.T
        )

        # The potential here is the negative mean distance travelled by the balls
        return -px[-1, :].mean()

    @property
    @jaxtyped
    @beartype
    def global_minimum(self) -> Float[Array, ""]:
        """Return the global minimum of the potential"""
        # This value is determined empirically. There is a closed-form expression
        # for how far something will fly in free space (at 45 degrees), but the wall
        # hecks that up. Instead, I just ran a sweep with 10k angles and found the best
        # output
        return jnp.array(-10.236)
