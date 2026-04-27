import textwrap

import numpy as np
import pytest

from meanfi import AdaptiveQuadrature, AdaptiveSimplex, UniformGrid, density_matrix, density_matrix_at_mu
from meanfi.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.helpers import (
    benchmark,
    converged_dense_reference,
    dimerized_chain,
    local_dense_model,
    max_density_error,
    peak_rss_bytes,
)


pytestmark = [pytest.mark.performance, pytest.mark.perf_slow]
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def _local_density_call(tb, *, filling: float):
    return density_matrix(
        tb,
        filling=filling,
        kT=0.1,
        keys=[(0, 0)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
        filling_tol=1e-6,
    )


@requires_ext
def test_zero_temperature_density_convergence_is_second_order():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.0,
        keys=keys,
        target_tol=1e-6,
        nk_start=251,
        nk_max=4001,
    )

    hs = []
    errors = []
    for density_atol in (1e-2, 3e-4, 1e-5):
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.0,
            keys=keys,
            integration=AdaptiveSimplex(
                density_matrix_tol=density_atol,
                max_refinements=None,
            ),
        )
        hs.append(result.info.n_leaves ** -1.0)
        errors.append(max_density_error(result.density_matrix, reference.rho))

    assert hs[0] > hs[1] > hs[2]
    assert errors[0] > errors[1] > errors[2]

    slope = float(np.polyfit(np.log(hs), np.log(errors), 1)[0])
    assert slope >= 1.7


@requires_ext
def test_uniform_grid_density_convergence_improves_with_more_kpoints():
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.0,
        keys=keys,
        target_tol=1e-6,
        nk_start=251,
        nk_max=4001,
    )

    hs = []
    errors = []
    for nk in (17, 33, 65):
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.0,
            keys=keys,
            integration=UniformGrid(nk=nk),
        )
        hs.append(result.info.unique_evals ** -1.0)
        errors.append(max_density_error(result.density_matrix, reference.rho))

    assert hs[0] > hs[1] > hs[2]
    assert errors[0] > errors[1] > errors[2]


def test_local_density_runtime_scales_consistent_with_cubic_dense_diagonalization(
    perf_slow_benchmark_config,
):
    times = []
    kernel_evals = []
    sizes = (48, 96, 192)
    models = {ndof: local_dense_model(ndof) for ndof in sizes}

    for ndof in sizes:
        tb = models[ndof]
        result = benchmark(
            lambda tb=tb, filling=ndof / 2: _local_density_call(tb, filling=filling),
            **perf_slow_benchmark_config,
        )
        info = result.last_result.info
        times.append(result.median_s)
        kernel_evals.append(info.unique_evals)

    assert kernel_evals[0] == kernel_evals[1] == kernel_evals[2]

    for (smaller, larger), (t_small, t_large) in zip(
        zip(sizes[:-1], sizes[1:], strict=True),
        zip(times[:-1], times[1:], strict=True),
        strict=True,
    ):
        expected_ratio = (larger / smaller) ** 3
        actual_ratio = t_large / t_small
        assert actual_ratio <= 2.0 * expected_ratio


def test_local_density_memory_scales_with_dense_workspace(
    perf_slow_benchmark_config,
):
    sizes = (32, 96, 192)
    baseline_rss = peak_rss_bytes("import meanfi")
    if baseline_rss is None:
        traced_peaks = []
        models = {ndof: local_dense_model(ndof) for ndof in sizes}
        for ndof in sizes:
            tb = models[ndof]
            result = benchmark(
                lambda tb=tb, filling=ndof / 2: _local_density_call(tb, filling=filling),
                track_tracemalloc=True,
                **perf_slow_benchmark_config,
            )
            assert result.peak_traced_bytes is not None
            traced_peaks.append(max(result.peak_traced_bytes, 1))

        for (smaller, larger), (m_small, m_large) in zip(
            zip(sizes[:-1], sizes[1:], strict=True),
            zip(traced_peaks[:-1], traced_peaks[1:], strict=True),
            strict=True,
        ):
            expected_ratio = (larger / smaller) ** 2
            actual_ratio = m_large / m_small
            assert actual_ratio <= 2.5 * expected_ratio
        return

    rss_samples = []
    for ndof in sizes:
        body = textwrap.dedent(
            f"""
            from meanfi import AdaptiveQuadrature, density_matrix
            from meanfi.tests.helpers import local_dense_model

            tb = local_dense_model({ndof})
            density_matrix(
                tb,
                filling={ndof} / 2,
                kT=0.1,
                keys=[(0, 0)],
                integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
                filling_tol=1e-6,
            )
            """
        )
        rss = peak_rss_bytes(body)
        assert rss is not None
        assert rss >= baseline_rss
        rss_samples.append(max(rss, 1))

    for (smaller, larger), (m_small, m_large) in zip(
        zip(sizes[:-1], sizes[1:], strict=True),
        zip(rss_samples[:-1], rss_samples[1:], strict=True),
        strict=True,
    ):
        expected_ratio = (larger / smaller) ** 2
        actual_ratio = m_large / m_small
        assert actual_ratio <= 2.5 * expected_ratio
