# %%
import numpy as np
import pytest

from pymf.tests.test_graphene import compute_gap
from pymf import (
    Model,
    solver,
    generate_guess,
    add_tb,
)

repeat_number = 3


# %%
def gap_relation_hubbard(Us, nk, nk_dense, tol=1e-3):
    """Gap relation for the Hubbard model.

    Parameters
    ----------
    Us : np.array
        The Hubbard U parameter.
    nk : int
        The number of k-points to use in the grid.
    nk_dense : int
        The number of k-points to use in the dense grid for calculating the gap.
    tol : float
        The tolerance for the fitting of the gap.
    """
    hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
    gaps = []
    for U in Us:
        h_int = {
            (0,): U * np.kron(np.eye(2), np.ones((2, 2))),
        }
        guess = generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
        full_model = Model(h_0, h_int, filling=2)
        mf_sol = solver(full_model, guess, nk=nk)
        _gap = compute_gap(add_tb(h_0, mf_sol), fermi_energy=0, nk=nk_dense)
        gaps.append(_gap)

    fit_gap = np.polyfit(Us, np.array(gaps), 1)[0]
    assert np.abs(fit_gap - 1) < tol


@pytest.mark.parametrize("seed", range(repeat_number))
def test_gap_hubbard(seed):
    """Test the gap prediction for the Hubbard model."""
    np.random.seed(seed)
    Us = np.linspace(8, 10, 15, endpoint=True)
    gap_relation_hubbard(Us, nk=20, nk_dense=100, tol=1e-1)
