import jax
import jax.numpy as jnp

from sampling_benchmarks.benchmarks import Ballistic


def test_ballistic():
    dim = 10
    problem = Ballistic(dim, jnp.ones((dim,)), jnp.array(1.0))

    # Test that potential runs
    assert problem.u(jnp.linspace(0, jnp.pi / 2, dim))

    # Test global minimum (this value is empirically determined so we don't actually
    # care what it is)
    assert problem.global_minimum


if __name__ == "__main__":
    # Plot the cost landscape
    import matplotlib.pyplot as plt

    dim = 1
    problem = Ballistic(dim, jnp.ones((dim,)), jnp.array(1.0))

    N = 10000
    x = jnp.linspace(0, jnp.pi/2, N).reshape(-1, 1)
    u = jax.vmap(problem.u)(x)

    plt.plot(x, u)
    plt.plot(x, x * 0.0 + problem.global_minimum, "k--", label="Global Minimum")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()

    plt.show()
