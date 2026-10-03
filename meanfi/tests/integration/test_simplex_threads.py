"""Thread limits reuse discovery and restore the caller's runtime settings."""

import numpy as np
import pytest
import threadpoolctl

from meanfi import FermiSimplex, density_matrix_at_mu
from meanfi.density.integrate.simplex.mesh import _native_thread_context

pytestmark = pytest.mark.integration


def test_simplex_reuses_discovery_for_repeated_density_calls(monkeypatch):
    def rediscovery(*args, **kwargs):
        pytest.fail("Integration rescanned loaded thread runtimes")

    monkeypatch.setattr(threadpoolctl, "ThreadpoolController", rediscovery)
    # The two bands stay on opposite sides of mu for every k. Their projectors
    # are constant, so the exact density is diag(1, 0).
    hopping = -0.2 * np.eye(2)
    h = {(0,): np.diag([-1.0, 1.0]), (1,): hopping, (-1,): hopping.T}
    for threads in (1, 2, None):
        result = density_matrix_at_mu(
            h, mu=0.0, keys=[(0,)], integration=FermiSimplex(num_threads=threads)
        )
        np.testing.assert_allclose(result.to_tb()[(0,)], np.diag([1, 0]), atol=1e-13)


def test_simplex_restores_thread_limits_after_nested_calls_and_errors():
    controller = threadpoolctl.ThreadpoolController()
    openmp = controller.select(user_api="openmp")
    if not openmp.info():
        pytest.skip("Native extension has no OpenMP runtime")
    blas_before = controller.select(user_api="blas").info()

    with openmp.limit(limits=4):
        with _native_thread_context(2):
            assert all(info["num_threads"] == 2 for info in openmp.info())
            with pytest.raises(RuntimeError, match="integration failed"):
                with _native_thread_context(1):
                    assert all(info["num_threads"] == 1 for info in openmp.info())
                    raise RuntimeError("integration failed")
            assert all(info["num_threads"] == 2 for info in openmp.info())
        with _native_thread_context(None):
            assert all(info["num_threads"] == 4 for info in openmp.info())
    assert controller.select(user_api="blas").info() == blas_before
