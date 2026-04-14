import numpy as np

from meanfi import Model, add_tb, density_matrix, guess_tb, solver
from meanfi.kwant_helper import kwant_examples, utils


def test_graphene_kwant_end_to_end_regression():
    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, {"U": 1.0, "V": 0.0})
    guess = guess_tb(frozenset(h_int), len(next(iter(h_0.values()))))

    model = Model(
        h_0,
        h_int,
        filling=2,
        kT=0.05,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=5e-4,
    )
    mf_sol, solver_info = solver(model, guess, max_scf_steps=20, return_info=True)
    h_mf = add_tb(h_0, mf_sol)
    rho, _, mu, density_info = density_matrix(
        h_mf,
        filling=2,
        kT=model.kT,
        keys=list(h_int),
        charge_tol=model.charge_tol,
        density_atol=model.density_atol,
        density_rtol=model.density_rtol,
    )

    assert solver_info.residual_norm <= model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    for key, matrix in mf_sol.items():
        opposite = tuple(-np.array(key))
        assert np.allclose(matrix, mf_sol[opposite].conj().T)
        assert np.all(np.isfinite(rho[key]))
