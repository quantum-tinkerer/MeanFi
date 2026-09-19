"""Callable composition preserves exact sums and checks matrix shapes."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from meanfi.hamiltonian import BlochHamiltonian, add_correction


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("finite_range", [False, True])
def test_corrected_callable_matches_exact_sum_and_owns_correction(sparse, finite_range):
    def bare(kx):
        return np.diag([np.cos(kx), -np.cos(kx)]).astype(complex)

    onsite = np.array([[0.3, 0.2j], [-0.2j, -0.1]])
    hopping = np.array([[0.1, 0.2j], [0.0, -0.2]])
    correction = {(0,): onsite.copy()}
    if finite_range:
        correction.update({(1,): hopping.copy(), (-1,): hopping.conj().T.copy()})
    if sparse:
        correction = {key: csr_matrix(value) for key, value in correction.items()}
    h = add_correction(BlochHamiltonian(bare), correction)
    if sparse:
        correction[(0,)].data[:] = 7
    else:
        correction[(0,)][:] = 7
    nested = add_correction(h, {(0,): np.eye(2) * 0.4})
    for kx in (0.0, 0.23, 1.9):
        expected = bare(kx) + onsite
        if finite_range:
            expected += np.exp(-1j * kx) * hopping + np.exp(1j * kx) * hopping.conj().T
        np.testing.assert_allclose(h(kx), expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            nested(kx), expected + 0.4 * np.eye(2), rtol=0, atol=1e-14
        )
    output = h(0.0)
    output[:] = 99
    assert h(0.0)[0, 0] != 99


@pytest.mark.parametrize(
    "bad,message",
    [
        (1.0, "square"),
        (np.ones(2), "square"),
        (np.ones((1, 1)), "shape must remain constant"),
    ],
)
def test_composition_rejects_invalid_callback_away_from_origin(bad, message):
    bare = BlochHamiltonian(lambda kx: np.eye(2) if kx == 0 else bad)
    corrected = add_correction(bare, {(0,): np.eye(2)})
    nested = add_correction(corrected, {(0,): np.eye(2)})
    for h in (corrected, nested):
        with pytest.raises(ValueError, match=message):
            h(0.2)
        with pytest.raises(ValueError, match="coordinates"):
            h(0.2, 0.3)


def test_callable_evaluations_do_not_scan_matrix_entries(monkeypatch):
    matrix = np.array([[0.1, 0.2j], [-0.2j, -0.3]])
    bare = BlochHamiltonian(lambda kx: matrix)
    corrected = add_correction(bare, {(0,): np.eye(2)})
    nested = add_correction(corrected, {(0,): np.eye(2)})

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Callable evaluation must not scan matrix entries")

    with monkeypatch.context() as patch:
        patch.setattr(np, "isfinite", unexpected_scan)
        patch.setattr(np, "abs", unexpected_scan)
        assert bare(0.2) is matrix
        actual = corrected(0.2)
        nested_actual = nested(0.2)
    np.testing.assert_allclose(actual, matrix + np.eye(2), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        nested_actual, matrix + 2 * np.eye(2), rtol=0, atol=1e-14
    )
