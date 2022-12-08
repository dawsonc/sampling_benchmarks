"""Sampling from the potential for the ballistic benchmark"""
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# isort: split

import blackjax

from sampling_benchmarks import BenchmarkRunner, TestCase
from sampling_benchmarks.benchmarks import Ballistic


def main(dimension: int):
    # Define the log probability for the benchmark problem
    benchmark = Ballistic(dimension, jnp.zeros((dimension,)), jnp.array(1.0))
    scale = 1.0 * dimension
    logprob = lambda x: -scale * benchmark.u(x)

    # Make a bunch of samplers to test
    cases = []

    # Make sure all take approximately similar steps and use the same overall number
    # of function evaluations
    hmc_learning_rates = [1e-2, 0.25]
    learning_rates = [1e-4, 1e-3, 1e-2][:1]
    num_function_evaluations = 100_000

    # MCMC
    for lr in learning_rates:
        lr = lr ** 0.5
        sigma = lr * jnp.ones((dimension,))
        rmh = blackjax.rmh(logprob, sigma)
        rmh_num_samples = num_function_evaluations
        cases.append(
            TestCase("RMH (lr {:.1e})".format(lr), rmh, benchmark, rmh_num_samples)
        )

    # MALA
    for lr in learning_rates:
        mala = blackjax.mala(logprob, lr)
        mala_num_samples = num_function_evaluations
        cases.append(
            TestCase("MALA (lr {:.1e})".format(lr), mala, benchmark, mala_num_samples)
        )

    # Make the benchmark runner
    num_trials_per_sampler = 1
    runner = BenchmarkRunner(cases, num_trials_per_sampler, compute_potentials=False)

    # Run!
    initial_guess = jnp.zeros((dimension,)) + 0.1
    result = runner.run(initial_guess)

    # Plot results and print times
    _, axs = plt.subplots(1, 1)
    axs = [axs]
    means = []
    for case in result.keys():
        # Report runtime
        case_time = result[case]["time"]

        # Report acceptance rate
        steps = jnp.linalg.norm(jnp.diff(result[case]["samples"], axis=1), axis=-1)
        acceptance_rate = (steps > 0).mean()

        print(
            "{}: {:.4f} s overall, {:.4f} s per chain, {:.2f}%% acceptance rate".format(
                case,
                case_time,
                case_time / num_trials_per_sampler,
                acceptance_rate * 100,
            )
        )

        potential = jax.vmap(jax.vmap(benchmark.u))(result[case]["samples"][:, ::100, :])
        means.append(potential.mean())

        case_mean = potential.mean(axis=0)
        # case_max = potential.max(axis=0)
        # case_min = potential.min(axis=0)

        x_case = jnp.linspace(0, num_function_evaluations, case_mean.size)
        ls = "-" if "MALA" in case else "--"
        h = axs[0].plot(x_case, case_mean, linestyle=ls, label=case)
        # axs[0].plot(x_case, case_min, linestyle="--", color=h[0].get_c())
        axs[0].plot(
            x_case,
            potential.T,
            linestyle="-",
            alpha=0.2,
            color=h[0].get_c(),
        )
        # axs[0].fill_between(x_case, case_min, case_max, alpha=0.1, color=h[0].get_c())

    # ind = jnp.arange(len(result.keys()))
    # plt.bar(ind, means)
    # plt.xticks(ind, labels=result.keys())
    xmin, xmax = axs[0].get_xlim()
    axs[0].plot(
        [xmin, xmax],
        [benchmark.global_minimum, benchmark.global_minimum],
        "k--",
        label="Optimum",
    )
    axs[0].plot(
        [xmin, xmax],
        [-5.5, -5.5],
        "k:",
        label="Flat region",
    )

    axs[0].set_xlabel("Potential function evaluations")
    axs[0].set_ylabel("Potential (avg. over last 100 samples)")
    plt.title(
        "Ballistic (n={}, scale={}); {} trials".format(
            dimension,
            scale,
            num_trials_per_sampler,
        )
    )
    axs[0].legend()

    plt.show()

    # # Animate the progress
    # fig, _ = plt.subplots()
    # plotting_problem = Ballistic(1, jnp.ones((1,)), jnp.array(1.0))
    # x = jnp.linspace(0, jnp.pi / 2, 1000).reshape(-1, 1)
    # u = jax.vmap(plotting_problem.u)(x)
    # plt.plot(x, u)

    # x_sol = result[list(result.keys())[0]]["samples"][:, 0, :].T
    # u_sol = jax.vmap(plotting_problem.u)(x_sol)
    # (sol_plt,) = plt.plot(x_sol, u_sol, "o")

    # def animate(i):
    #     x_sol = result[list(result.keys())[0]]["samples"][:, i, :].T
    #     u_sol = jax.vmap(plotting_problem.u)(x_sol)
    #     sol_plt.set_data(x_sol, u_sol)
    #     return (sol_plt,)

    # anim = animation.FuncAnimation(
    #     fig,
    #     animate,
    #     frames=mala_num_samples,
    #     interval=10,
    #     blit=True,
    #     repeat=True,
    # )

    # anim.save(f"ballistic_mala_{mala_learning_rates[0]}.mp4", writer="ffmpeg", fps=30)


if __name__ == "__main__":
    dimension = 500
    main(dimension)
