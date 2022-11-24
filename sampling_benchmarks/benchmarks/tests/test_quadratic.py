import jax.numpy as jnp

from sampling_benchmarks.benchmarks import Quadratic

def test_heaviside():
    dim = 3
    problem = Quadratic(dim, jnp.ones((dim,)), jnp.array(1.0))
    
    # Test potential
    assert jnp.isclose(problem.u(jnp.ones((dim,))), 1.0)
    assert jnp.isclose(problem.u(-jnp.ones((dim,))), 1.0)

    # Test global minimum
    assert problem.global_minimum == 0.0
