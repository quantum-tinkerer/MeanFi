import statistics as stats
import time
import tracemalloc

import numpy as np
from scipy.optimize._nonlin import NoConvergence

from meanfi import Model, guess_tb, solver


def hubbard_chain_model(U=2.0):
    hopping = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopping + hopping.T.conj(), (1,): hopping, (-1,): hopping.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=0.1,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=1e-5,
    )
    guess = guess_tb(frozenset(h_int), len(next(iter(h_0.values()))))
    return model, guess


def benchmark(fn, repeat=5, warmup=1):
    for _ in range(warmup):
        fn()
    timings = []
    peaks = []
    last = None
    for _ in range(repeat):
        tracemalloc.start()
        start = time.perf_counter()
        last = fn()
        timings.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak / 1024**2)
        tracemalloc.stop()
    return {
        "median_ms": 1e3 * stats.median(timings),
        "mean_ms": 1e3 * stats.mean(timings),
        "stdev_ms": 1e3 * stats.pstdev(timings),
        "peak_mb": max(peaks),
        "last": last,
    }


def print_case(name, result):
    _, info = result["last"]
    print(name)
    print(
        f"  median {result['median_ms']:.3f} ms | mean {result['mean_ms']:.3f} ms | "
        f"stdev {result['stdev_ms']:.3f} ms | peak {result['peak_mb']:.3f} MiB"
    )
    print(
        "  solver stats: "
        f"{info.iterations} SCF iterations, "
        f"residual {info.residual_norm:.3e}, "
        f"mu {info.mu:.6f}, "
        f"{info.total_charge_integration_calls} charge integrations, "
        f"{info.total_density_integration_calls} density integrations, "
        f"{info.total_kernel_evals} kernel evals, "
        f"{info.total_evaluator_evals} evaluator evals"
    )
    print(
        "  last density stats: "
        f"{info.last_density_info.n_leaves} leaves, "
        f"{info.last_density_info.n_leaf_nodes} leaf nodes, "
        f"{info.last_density_info.root_iterations} root iterations"
    )


if __name__ == "__main__":
    np.random.seed(0)
    model, guess = hubbard_chain_model()

    anderson = benchmark(
        lambda: solver(model, guess, return_info=True, mixing="anderson", max_scf_steps=40)
    )

    print_case("1D Hubbard SCF benchmark (Anderson)", anderson)

    try:
        linear = benchmark(
            lambda: solver(
                model,
                guess,
                return_info=True,
                mixing="linear",
                mixing_kwargs={"alpha": 0.2},
                max_scf_steps=120,
            )
        )
    except NoConvergence:
        print("1D Hubbard SCF benchmark (Linear)")
        print("  linear mixing did not converge within 120 iterations at alpha=0.2")
    else:
        print_case("1D Hubbard SCF benchmark (Linear)", linear)
        print(f"  Anderson speedup over linear: {linear['median_ms'] / anderson['median_ms']:.2f}x")
