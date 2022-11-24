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
