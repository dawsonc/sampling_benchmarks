from functools import partial
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List
from blackjax.base import MCMCSamplingAlgorithm
from jaxtyping import Array, Float, Integer, jaxtyped, PyTree

from sampling_benchmarks.benchmarks import Benchmark


@dataclass
class TestCase:
    """
    Defines a single test case.

    args:
        name: the name of this test case
        sampler: the MCMCSamplingAlgorithm to use for this test
        benchmark: the benchmark associated with this test. Should match the log
            probability function supplied to sampler.
        num_samples: how many steps to sample in this chain
    """

    name: str
    sampler: MCMCSamplingAlgorithm
    benchmark: Benchmark
    num_samples: int


class BenchmarkRunner:
    """
    Runs a set of benchmarks with a set of sampling algorithms and hyperparameters.

    args:
        cases: a list of tuples of names, MCMCSamplingAlgorithms, and Benchmarks. Each
            sampler should already include a reference to the log probability function
            it uses, but the benchmark is included to get objective values at each step.
        num_trials_per_case: how many trials to run for each case
        compute_potentials: If True, return the potential of each sample
    """

    test_cases: List[TestCase]
    num_trials_per_case: int
    compute_potentials: bool

    def __init__(
        self,
        test_cases: List[TestCase],
        num_trials_per_case: int,
        compute_potentials: bool = True,
    ) -> None:
        self.test_cases = test_cases
        self.num_trials_per_case = num_trials_per_case
        self.compute_potentials = compute_potentials

    @jaxtyped
    @beartype
    def inference_loop(
        self,
        rng_key: Integer[Array, "..."],
        kernel,
        initial_state,
        num_samples: int,
    ) -> Float[Array, "num_samples dim"]:
        """
        Run the given kernel for the specified number of steps, returning the samples.

        args:
            rng_key: JAX rng key
            kernel: the MCMC kernel to run
            initial_state: the starting state for the sampler
            num_samples: how many samples to take
        """

        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states.position

    @jaxtyped
    @beartype
    def run(self, initial_guess: Float[Array, " dim"]) -> PyTree:
        """
        Run the benchmark suite (with as much JAX optimization as possible).

        args:
            initial_guess: the starting point for the sampling process
        """
        results = {}

        for tc in self.test_cases:
            print(f"Running {tc.name}...")

            # Setup the sampler
            initial_state = tc.sampler.init(initial_guess)
            kernel = jax.jit(tc.sampler.step)

            # Run inference once to JIT, then re-run to get runtimes
            single_chain = lambda key: self.inference_loop(
                key, kernel, initial_state, tc.num_samples
            )
            single_chain = jax.jit(single_chain)
            base_key = jax.random.PRNGKey(0)
            single_chain(base_key)

            keys = jax.random.split(base_key, self.num_trials_per_case)
            start = time.perf_counter()
            # samples = multiple_chains(keys)
            samples = [single_chain(key) for key in keys]
            end = time.perf_counter()
            samples = jnp.stack(samples)

            # Get the potential values for each sample
            if self.compute_potentials:
                u = jax.vmap(jax.vmap(tc.benchmark.u))(samples)

            # Save results
            results[tc.name] = {
                "samples": samples,
                "potential": u if self.compute_potentials else None,
                "time": end - start,
            }

            print(f"Done with {tc.name}")

        return results
