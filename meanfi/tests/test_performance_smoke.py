import gc
import tracemalloc

import numpy as np

from meanfi import density_matrix


def _spinful_chain():
    hopping = -np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def _qiwuzhang(m=0.5):
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


def test_operation_counts_1d_smoke():
    _, _, _, info = density_matrix(
        _spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-8,
        density_atol=1e-8,
    )

    assert info.root_iterations <= 8
    assert info.n_kernel_evals <= 1500
    assert info.n_evaluator_evals <= 2500


def test_operation_counts_2d_smoke():
    _, _, _, info = density_matrix(
        _qiwuzhang(),
        filling=1.0,
        kT=0.1,
        keys=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
        charge_tol=1e-7,
        density_atol=1e-7,
    )

    assert info.root_iterations <= 4
    assert info.n_kernel_evals <= 25000
    assert info.n_evaluator_evals <= 60000


def test_repeated_density_solve_memory_growth_smoke():
    tb = _spinful_chain()
    keys = [(0,), (1,), (-1,)]

    tracemalloc.start()
    for _ in range(3):
        density_matrix(tb, filling=0.7, kT=0.15, keys=keys, charge_tol=1e-8, density_atol=1e-8)
    gc.collect()
    baseline = tracemalloc.take_snapshot()

    for _ in range(5):
        density_matrix(tb, filling=0.7, kT=0.15, keys=keys, charge_tol=1e-8, density_atol=1e-8)
    gc.collect()
    current = tracemalloc.take_snapshot()
    tracemalloc.stop()

    growth = sum(
        stat.size_diff
        for stat in current.compare_to(baseline, "filename")
        if stat.size_diff > 0
    )
    assert growth < 1_000_000
