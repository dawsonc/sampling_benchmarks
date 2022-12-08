"""Sanity check by sampling from a quadratic potential"""
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

# isort: split

import blackjax

from sampling_benchmarks.samplers.informal import informal_hmc

from sampling_benchmarks import BenchmarkRunner, TestCase
from sampling_benchmarks.benchmarks import Quadratic


def main(dimension: int):
    # Define the log probability for the benchmark problem
    benchmark = Quadratic(dimension, jnp.ones((dimension,)), jnp.array(1.0))
    scale = 1.0
    logprob = lambda x: -scale * benchmark.u(x)
    logprob = jax.jit(logprob)

    # Make INFORMAL HMC sampler
    cases = []

    # Make sure both take approximately similar steps and use the same overall number
    # of function evaluations
    learning_rates = [0.1]
    num_function_evaluations = 100

    # INFORMAL HMC
    for lr in learning_rates:
        inv_mass_matrix = jnp.ones((dimension,))
        num_integration_steps = 1  # 50
        max_lipschitz = 10.0
        step_size = lr / num_integration_steps
        hmc_num_samples = int(num_function_evaluations / num_integration_steps)
        hmc = informal_hmc(
            logprob, step_size, inv_mass_matrix, max_lipschitz, num_integration_steps
        )
        cases.append(
            TestCase("iHMC (lr {:.2e})".format(lr), hmc, benchmark, hmc_num_samples)
        )

    # Make the benchmark runner
    num_trials_per_sampler = 1
    runner = BenchmarkRunner(cases, num_trials_per_sampler)

    # Run!
    initial_guess = 10 * jnp.ones((dimension,))
    result = runner.run(initial_guess)

    # Plot results and print times
    for case in result.keys():
        # Report runtime
        case_time = result[case]["time"]

        # Report acceptance rate
        steps = jnp.linalg.norm(jnp.diff(result[case]["samples"], axis=1), axis=-1)
        acceptance_rate = (steps > 0).mean()

        print(
            "{}: {:.4f} s overall, {:.4f} s per chain, {:.2f}% acceptance rate".format(
                case,
                case_time,
                case_time / num_trials_per_sampler,
                acceptance_rate * 100,
            )
        )

        case_mean = result[case]["potential"].mean(axis=0)
        case_max = result[case]["potential"].max(axis=0)
        case_min = result[case]["potential"].min(axis=0)

        x_case = jnp.linspace(0, num_function_evaluations, case_mean.size)
        h = plt.plot(x_case, case_mean, linestyle="-", label=case)
        plt.plot(
            x_case,
            result[case]["potential"].T,
            linestyle="-",
            alpha=0.4,
            color=h[0].get_c(),
        )
        plt.fill_between(x_case, case_min, case_max, alpha=0.2, color=h[0].get_c())

    plt.plot(x_case, 0 * x_case + benchmark.global_minimum, "k--", label="Optimum")

    plt.xlabel("Potential function evaluations")
    plt.ylabel("Potential")
    plt.title(
        "Quadratic (n={}, scale={}); {} trials".format(
            dimension,
            scale,
            num_trials_per_sampler,
        )
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    dimension = 1
    main(dimension)
