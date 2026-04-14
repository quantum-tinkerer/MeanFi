import statistics as stats
import time
import tracemalloc

import numpy as np
from scipy.integrate import cubature
from scipy.optimize import root_scalar

from meanfi import density_matrix, fermi_dirac
from meanfi.tb.transforms import tb_to_kfunc


def qi_wu_zhang(m=0.5):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return {
        (0, 0): m * sz,
        (1, 0): 0.5 * sz - 0.5j * sx,
        (-1, 0): 0.5 * sz + 0.5j * sx,
        (0, 1): 0.5 * sz - 0.5j * sy,
        (0, -1): 0.5 * sz + 0.5j * sy,
    }


def complex_cubature(integrand, a, b, *, atol, rtol):
    def wrapped(k):
        value = integrand(k)
        return np.stack([value.real, value.imag], axis=-1)

    result = cubature(wrapped, a, b, atol=atol, rtol=rtol)
    estimate = result.estimate[..., 0] + 1j * result.estimate[..., 1]
    error = result.error[..., 0] + 1j * result.error[..., 1]
    return estimate, error


def split_bundle_result(estimate, ndof, keys):
    charge = float(np.real(estimate[0]))
    rho_flat = estimate[2:].reshape(ndof, ndof, len(keys))
    rho = {key: rho_flat[..., idx] for idx, key in enumerate(keys)}
    return charge, rho


def bundle_from_cubature(h, mu, kT, keys, atol=1e-7, rtol=1e-7):
    hkfunc = tb_to_kfunc(h)
    keys_arr = np.array(keys, dtype=float)
    ndim = len(next(iter(h)))
    prefactor = 1 / (2 * np.pi) ** ndim

    def integrand(k):
        h_k = hkfunc(k)
        eigenvalues, eigenvectors = np.linalg.eigh(h_k)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        doccupation = occupation * (1 - occupation) / kT
        density_matrix_k = (
            eigenvectors
            * occupation[:, np.newaxis, :]
            @ eigenvectors.conj().transpose(0, 2, 1)
        )
        phase = np.exp(1j * np.dot(k, keys_arr.T))
        density_terms = (
            density_matrix_k[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, :]
        ).reshape(k.shape[0], -1)
        charge = np.sum(occupation, axis=-1, keepdims=True)
        dcharge = np.sum(doccupation, axis=-1, keepdims=True)
        return prefactor * np.concatenate([charge, dcharge, density_terms], axis=-1)

    ndim = len(next(iter(h)))
    return complex_cubature(
        integrand,
        np.array([-np.pi] * ndim),
        np.array([np.pi] * ndim),
        atol=atol,
        rtol=rtol,
    )


def density_matrix_cubature(h, filling, kT, keys, atol=1e-7, rtol=1e-7):
    bound = sum(np.linalg.norm(matrix, ord=2) for matrix in h.values()) + max(1.0, 10 * kT)
    lower = -float(bound)
    upper = float(bound)

    def residual(mu):
        estimate, error = bundle_from_cubature(h, mu, kT, keys, atol=atol, rtol=rtol)
        charge = float(np.real(estimate[0])) - filling
        charge_error = float(np.abs(error[0]))
        return charge, charge_error

    while True:
        lower_charge, _ = residual(lower)
        upper_charge, _ = residual(upper)
        if lower_charge <= 0 <= upper_charge:
            break
        lower *= 2
        upper *= 2

    def root_fn(mu):
        charge, _ = residual(mu)
        return charge

    mu = float(root_scalar(root_fn, bracket=(lower, upper), method="brentq").root)
    estimate, _ = bundle_from_cubature(h, mu, kT, keys, atol=atol, rtol=rtol)
    charge, rho = split_bundle_result(estimate, next(iter(h.values())).shape[0], keys)
    return rho, mu, charge


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


def max_rho_diff(lhs, rhs):
    return max(np.max(np.abs(lhs[key] - rhs[key])) for key in lhs)


def run_case(label, *, filling):
    h = qi_wu_zhang()
    kT = 0.1
    keys = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    stateful = benchmark(
        lambda: density_matrix(
            h, filling=filling, kT=kT, keys=keys, charge_tol=1e-7, density_atol=1e-7
        )
    )
    cubature_ref = benchmark(
        lambda: density_matrix_cubature(
            h, filling=filling, kT=kT, keys=keys, atol=1e-7, rtol=1e-7
        ),
        repeat=3,
        warmup=1,
    )

    rho_stateful, _, mu_stateful, info = stateful["last"]
    rho_cubature, mu_cubature, charge_cubature = cubature_ref["last"]

    print(label)
    print(
        f"  stateful: median {stateful['median_ms']:.3f} ms | mean {stateful['mean_ms']:.3f} ms | "
        f"stdev {stateful['stdev_ms']:.3f} ms | peak {stateful['peak_mb']:.3f} MiB"
    )
    print(
        f"  cubature: median {cubature_ref['median_ms']:.3f} ms | mean {cubature_ref['mean_ms']:.3f} ms | "
        f"stdev {cubature_ref['stdev_ms']:.3f} ms | peak {cubature_ref['peak_mb']:.3f} MiB"
    )
    print(f"  speedup: {cubature_ref['median_ms'] / stateful['median_ms']:.2f}x")
    print(f"  mu difference: {abs(mu_stateful - mu_cubature):.3e}")
    print(f"  charge difference: {abs(info.charge - charge_cubature):.3e}")
    print(f"  max rho difference: {max_rho_diff(rho_stateful, rho_cubature):.3e}")
    print(
        "  stateful stats: "
        f"{info.root_iterations} root iterations, "
        f"{info.charge_integration_calls} charge integrations, "
        f"{info.density_integration_calls} density integrations, "
        f"{info.n_kernel_evals} kernel evals, "
        f"{info.n_evaluator_evals} evaluator evals, "
        f"{info.n_leaves} leaves, "
        f"{info.n_leaf_nodes} leaf nodes"
    )


if __name__ == "__main__":
    run_case("2D QWZ density benchmark (half-filled symmetric)", filling=1.0)
    run_case("2D QWZ density benchmark (doped)", filling=0.8)
