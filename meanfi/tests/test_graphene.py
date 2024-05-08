# %%
import numpy as np
import pytest

from meanfi.kwant_helper import kwant_examples, utils
from meanfi import (
    Model,
    solver,
    tb_to_kgrid,
    guess_tb,
    add_tb,
)


def compute_gap(tb, fermi_energy=0, nk=100):
    """Compute gap.

    Parameters
    ----------
    tb : dict
        Tight-binding model for which to compute the gap.
    fermi_energy : float
     Fermi energy.
    nk : int
     Number of k-points to sample along each dimension.

    Returns
    -------
     gap : float
     Indirect gap.
    """
    kham = tb_to_kgrid(tb, nk)
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)


repeat_number = 5
# %%
graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
h_0 = utils.builder_to_tb(graphene_builder)


# %%
def gap_prediction(U, V):
    """Test if the mean-field theory predicts the gap correctly for a given U and V.

    Parameters
    ----------
    U : float
        The Hubbard U parameter. Rounded to one decimal.
    V : float
        The nearest-neighbor interaction parameter. Rounded to one decimal.
    """
    U = np.round(U, 1)
    V = np.round(V, 1)
    params = {"U": U, "V": V}

    # Compare to phase diagram in https://arxiv.org/pdf/1204.4531.pdf
    upper_phase_line = 0.181 * U + 0.416
    lower_phase_line = 1.707 * U - 3.823
    triple_point = (2.78, 0.92)

    gapped = False
    if triple_point < (U, V):
        gapped = True
    elif (upper_phase_line < V) | (lower_phase_line > V):
        gapped = True

    # the mean-field calculation
    filling = 2
    nk = 40

    h_int = utils.builder_to_tb(int_builder, params)
    guess = guess_tb(frozenset(h_int), len(list(h_0.values())[0]))
    model = Model(h_0, h_int, filling)

    mf_sol = solver(model, guess, nk=nk, optimizer_kwargs={"M": 0, "f_tol": 1e-8})
    gap = compute_gap(add_tb(h_0, mf_sol), nk=200)

    # Check if the gap is predicted correctly
    if gap > 0.1:
        gapped_predicted = True
    else:
        gapped_predicted = False
    assert (
        gapped == gapped_predicted
    ), f"Mean-field theory failed to predict the gap for U = {U}, V = {V}"


# %%
@pytest.mark.parametrize("seed", range(repeat_number))
def test_gap(seed):
    """Test if the mean-field theory predicts the gap correctly for a random U and V."""
    np.random.seed(seed)
    U = np.random.uniform(0, 4)
    V = np.random.uniform(0, 1)
    gap_prediction(U, V)
