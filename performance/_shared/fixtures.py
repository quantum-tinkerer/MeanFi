from __future__ import annotations

import json
import statistics
import subprocess
import sys
import textwrap
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from meanfi import fermi_dirac, tb_to_kfunc
from meanfi.density.kpoint.zero_dim import zero_dim_zero_temp_mu


@dataclass(frozen=True)
class DenseReference:
    mu: float
    charge: float
    rho: dict[tuple[int, ...], np.ndarray]
    nk: int


@dataclass(frozen=True)
class BenchmarkResult:
    median_s: float
    mean_s: float
    stdev_s: float
    last_result: object
    peak_traced_bytes: int | None = None


def spinful_chain(t: float = 1.0):
    hopping = -t * np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def shifted_spinful_chain(phi: float = np.pi / 3.0):
    hopping = -np.exp(1j * phi) * np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def dimerized_chain(delta: float = 0.2):
    onsite = np.array([[delta, 0.0], [0.0, -delta]], dtype=complex)
    hopping = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    return {
        (0,): onsite + hopping + hopping.T.conj(),
        (1,): hopping,
        (-1,): hopping.T.conj(),
    }


def qiwuzhang(m: float = 1.3):
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


def local_two_band_2d(energy: float = 1.0):
    return {(0, 0): np.diag([-energy, energy])}


def local_dense_model(
    ndof: int,
    *,
    energy_window: float = 1.0,
    coupling_scale: float = 0.2,
):
    grid = np.arange(ndof, dtype=float)
    distance = np.abs(grid[:, None] - grid[None, :])
    phase = np.cos(0.17 * (grid[:, None] + 1.0) * (grid[None, :] + 1.0) / ndof)
    matrix = coupling_scale * phase / (1.0 + distance)
    matrix += np.diag(np.linspace(-energy_window, energy_window, ndof))
    return {(0, 0): matrix.astype(complex)}


def duplicated_local_two_band_1d(energy: float = 1.0):
    primitive = {(0,): np.diag([-energy, energy])}
    doubled = {(0,): np.diag([-energy, energy, -energy, energy])}
    return primitive, doubled


def bipartite_hubbard_1d(U: float):
    hopping = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hopping + hopping.T.conj(), (1,): hopping, (-1,): hopping.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def bipartite_hubbard_2d(U: float):
    hopping = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {
        (0, 0): hopping + hopping.T.conj(),
        (1, 0): hopping,
        (-1, 0): hopping.T.conj(),
        (0, 1): hopping,
        (0, -1): hopping.T.conj(),
        (1, 1): hopping,
        (-1, -1): hopping.T.conj(),
    }
    h_int = {(0, 0): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def antiferromagnetic_guess(delta: float, ndim: int):
    return {(0,) * ndim: np.diag([-delta, delta, delta, -delta]).astype(complex)}


def staggered_magnetization(local_density: np.ndarray) -> float:
    occupations = np.real(np.diag(local_density))
    magnetization = 0.5 * (
        (occupations[0] - occupations[1]) + (occupations[3] - occupations[2])
    )
    return float(abs(magnetization))


def exact_spinful_chain_mu(filling: float) -> float:
    return float(-2.0 * np.cos(0.5 * np.pi * filling))


def exact_spinful_chain_charge(mu: float) -> float:
    if mu <= -2.0:
        return 0.0
    if mu >= 2.0:
        return 2.0
    return float(2.0 * np.arccos(-0.5 * mu) / np.pi)


def gamma_abs_samples(h_0, ndim: int, nk: int) -> np.ndarray:
    hkfunc = tb_to_kfunc(h_0)
    if ndim == 1:
        points = np.linspace(-np.pi, np.pi, nk, endpoint=False)[:, None]
    elif ndim == 2:
        axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        kx, ky = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack([kx.ravel(), ky.ravel()], axis=-1)
    else:
        raise ValueError("Only 1D and 2D Hubbard references are supported")

    h_k = hkfunc(points)
    spin_up_block = h_k[:, [0, 2]][:, :, [0, 2]]
    return np.abs(spin_up_block[:, 0, 1])


def solve_antiferromagnetic_gap(h_0, *, U: float, kT: float, ndim: int, nk: int):
    gamma_abs = gamma_abs_samples(h_0, ndim=ndim, nk=nk)

    def residual(delta: float) -> float:
        energy = np.sqrt(gamma_abs**2 + delta**2)
        return 1.0 - 0.5 * U * np.mean(np.tanh(energy / (2.0 * kT)) / energy)

    if residual(1e-12) >= 0.0:
        return 0.0

    upper = max(10.0, 2.0 * U)
    return float(brentq(residual, 1e-12, upper))


def max_density_error(
    lhs: dict[tuple[int, ...], np.ndarray],
    rhs: dict[tuple[int, ...], np.ndarray],
) -> float:
    return max(float(np.max(np.abs(lhs[key] - rhs[key]))) for key in rhs)


def max_density_estimate(error: dict[tuple[int, ...], np.ndarray] | None) -> float:
    if error is None:
        return float("nan")
    return max(float(np.max(np.abs(matrix))) for matrix in error.values())


def assert_estimator_covers_actual(
    actual_error: float,
    reported_error: float,
    *,
    factor: float = 2.0,
    slack: float = 1e-12,
) -> None:
    assert actual_error <= factor * reported_error + slack


def dense_grid_data(tb, nk: int):
    ndim = len(next(iter(tb)))
    hkfunc = tb_to_kfunc(tb)
    axes = [np.linspace(-np.pi, np.pi, nk, endpoint=False) for _ in range(ndim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=-1)
    h_k = hkfunc(points)
    if h_k.ndim == 2:
        h_k = h_k[np.newaxis, ...]
    eigenvalues, eigenvectors = np.linalg.eigh(h_k)
    return points, eigenvalues, eigenvectors


def dense_charge_from_data(data, mu: float, kT: float) -> float:
    _points, eigenvalues, _eigenvectors = data
    occupation = fermi_dirac(eigenvalues, kT, mu)
    return float(np.mean(np.sum(occupation, axis=-1)))


def dense_mu_from_data(
    data,
    *,
    filling: float,
    kT: float,
    mu_guess: float = 0.0,
) -> tuple[float, float]:
    _points, eigenvalues, _eigenvectors = data
    if kT == 0:
        total_filling = float(filling) * eigenvalues.shape[0]
        mu, charge_total = zero_dim_zero_temp_mu(
            eigenvalues.reshape(-1),
            filling=total_filling,
            mu_guess=mu_guess,
        )
        return float(mu), float(charge_total / eigenvalues.shape[0])

    lower = float(np.min(eigenvalues)) - max(1.0, 10.0 * kT)
    upper = float(np.max(eigenvalues)) + max(1.0, 10.0 * kT)
    mu = float(
        brentq(
            lambda candidate: dense_charge_from_data(data, candidate, kT) - filling,
            lower,
            upper,
        )
    )
    return mu, dense_charge_from_data(data, mu, kT)


def dense_density_from_data(data, mu: float, kT: float, keys: list[tuple[int, ...]]):
    points, eigenvalues, eigenvectors = data
    occupation = fermi_dirac(eigenvalues, kT, mu)
    density_matrix_k = (
        eigenvectors
        * occupation[:, np.newaxis, :]
        @ eigenvectors.conj().transpose(0, 2, 1)
    )

    rho = {}
    for key in keys:
        phase = np.exp(1j * np.dot(points, np.asarray(key, dtype=float)))
        rho[key] = np.einsum("k,kab->ab", phase / points.shape[0], density_matrix_k)
    return rho


def converged_dense_reference(
    tb,
    *,
    keys: list[tuple[int, ...]],
    kT: float,
    target_tol: float,
    nk_start: int,
    nk_max: int,
    mu: float | None = None,
    filling: float | None = None,
) -> DenseReference:
    if (mu is None) == (filling is None):
        raise ValueError("Exactly one of mu or filling must be provided")

    previous = None
    nk = nk_start
    while True:
        data = dense_grid_data(tb, nk)
        if mu is None:
            resolved_mu, charge = dense_mu_from_data(data, filling=filling, kT=kT)
        else:
            resolved_mu = float(mu)
            charge = dense_charge_from_data(data, resolved_mu, kT)

        reference = DenseReference(
            mu=resolved_mu,
            charge=charge,
            rho=dense_density_from_data(data, resolved_mu, kT, keys),
            nk=nk,
        )
        if previous is not None:
            if (
                abs(reference.mu - previous.mu) <= target_tol
                and abs(reference.charge - previous.charge) <= target_tol
                and max_density_error(reference.rho, previous.rho) <= target_tol
            ):
                return reference

        if nk >= nk_max:
            raise AssertionError(
                f"Dense reference did not self-converge by nk={nk_max}"
            )
        previous = reference
        nk = 2 * nk - 1


def benchmark(
    fn,
    *,
    repeat: int,
    warmup: int,
    track_tracemalloc: bool = False,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()

    timings = []
    peaks = []
    last_result = None
    for _ in range(repeat):
        if track_tracemalloc:
            tracemalloc.start()
        start = time.perf_counter()
        last_result = fn()
        timings.append(time.perf_counter() - start)
        if track_tracemalloc:
            _current, peak = tracemalloc.get_traced_memory()
            peaks.append(peak)
            tracemalloc.stop()

    peak_traced_bytes = max(peaks) if peaks else None
    return BenchmarkResult(
        median_s=statistics.median(timings),
        mean_s=statistics.mean(timings),
        stdev_s=statistics.pstdev(timings),
        last_result=last_result,
        peak_traced_bytes=peak_traced_bytes,
    )


def peak_rss_bytes(python_body: str) -> int | None:
    body = textwrap.indent(textwrap.dedent(python_body).strip(), "    ")
    if not body:
        body = "    pass"

    runner = "\n".join(
        [
            "import json",
            "import sys",
            "",
            "try:",
            "    import resource",
            "except ImportError:",
            '    print(json.dumps({"rss_bytes": None}))',
            "else:",
            body,
            "    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            '    if sys.platform.startswith("darwin"):',
            "        rss_bytes = int(rss)",
            "    else:",
            "        rss_bytes = int(rss * 1024)",
            '    print(json.dumps({"rss_bytes": rss_bytes}))',
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip().splitlines()
    if not output:
        return None
    return json.loads(output[-1])["rss_bytes"]
