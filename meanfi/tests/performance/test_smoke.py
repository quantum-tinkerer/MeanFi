import gc
import tracemalloc

import pytest

from meanfi import density_matrix
from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE
from meanfi.tests.helpers import (
    benchmark,
    dimerized_chain,
    local_dense_model,
    qiwuzhang,
    spinful_chain,
)


pytestmark = pytest.mark.performance
requires_native = pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)


def test_operation_counts_1d_smoke():
    _rho, _error, _mu, info = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-4,
        density_atol=1e-3,
    )

    assert info.root_iterations <= 4
    assert info.n_kernel_evals <= 500
    assert info.n_evaluator_evals <= 1500


def test_operation_counts_2d_smoke():
    _rho, _error, _mu, info = density_matrix(
        qiwuzhang(m=0.5),
        filling=1.0,
        kT=0.1,
        keys=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
        charge_tol=1e-4,
        density_atol=1e-3,
    )

    assert info.root_iterations <= 2
    assert info.n_kernel_evals <= 2000
    assert info.n_evaluator_evals <= 4000


@requires_native
def test_zero_temperature_operation_counts_smoke():
    _rho, _error, _mu, info = density_matrix(
        dimerized_chain(),
        filling=1.0,
        kT=0.0,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-4,
        density_atol=1e-3,
        max_subdivisions=None,
    )

    assert info.root_iterations <= 2
    assert info.n_kernel_evals <= 64
    assert info.n_evaluator_evals <= 400
    assert info.n_leaves <= 32


def test_repeated_density_solve_memory_growth_smoke():
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]

    tracemalloc.start()
    for _ in range(3):
        density_matrix(
            tb,
            filling=0.7,
            kT=0.15,
            keys=keys,
            charge_tol=1e-4,
            density_atol=1e-3,
        )
    gc.collect()
    baseline = tracemalloc.take_snapshot()

    for _ in range(5):
        density_matrix(
            tb,
            filling=0.7,
            kT=0.15,
            keys=keys,
            charge_tol=1e-4,
            density_atol=1e-3,
        )
    gc.collect()
    current = tracemalloc.take_snapshot()
    tracemalloc.stop()

    growth = sum(
        stat.size_diff
        for stat in current.compare_to(baseline, "filename")
        if stat.size_diff > 0
    )
    assert growth < 1_000_000


def test_local_density_walltime_ratio_smoke(perf_smoke_benchmark_config):
    small_tb = local_dense_model(32)
    large_tb = local_dense_model(64)

    small = benchmark(
        lambda: density_matrix(
            small_tb,
            filling=16.0,
            kT=0.1,
            keys=[(0, 0)],
            charge_tol=1e-6,
            density_atol=1e-6,
        ),
        **perf_smoke_benchmark_config,
    )
    large = benchmark(
        lambda: density_matrix(
            large_tb,
            filling=32.0,
            kT=0.1,
            keys=[(0, 0)],
            charge_tol=1e-6,
            density_atol=1e-6,
        ),
        **perf_smoke_benchmark_config,
    )

    ratio = large.median_s / small.median_s
    assert ratio <= 20.0
