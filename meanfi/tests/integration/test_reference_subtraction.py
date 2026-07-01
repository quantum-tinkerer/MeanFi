import numpy as np
import pytest

from meanfi import AdaptiveQuadrature, LinearMixing, Model, density_matrix, solver
from meanfi.meanfield import meanfield


pytestmark = pytest.mark.integration


def test_reference_density_subtracts_full_mean_field_correction():
    h_0 = {(): np.array([[0.3, 0.7], [0.7, -0.2]], dtype=complex)}
    h_int = {(): np.array([[0.5, 1.2], [1.2, 0.8]], dtype=complex)}
    rho_ref = {
        (): np.array(
            [[0.6, 0.15 + 0.08j], [0.15 - 0.08j, 0.4]],
            dtype=complex,
        )
    }
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        reference_density_matrix=rho_ref,
    )

    unsubtracted_correction = meanfield(rho_ref, h_int)[()]
    hamiltonian = model.hamiltonian_from_rho(rho_ref)

    assert abs(unsubtracted_correction[0, 1]) > 1e-12
    np.testing.assert_allclose(hamiltonian[()], h_0[()], atol=1e-12)


def test_solver_reference_density_fixed_point_has_zero_interaction_correction():
    h_0 = {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    h_int = {(): np.array([[0.4, 1.1], [1.1, 0.6]], dtype=complex)}
    integration = AdaptiveQuadrature(density_matrix_tol=1e-10)
    rho_ref = density_matrix(
        h_0,
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=integration,
        filling_tol=1e-10,
    ).density_matrix
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        reference_density_matrix=rho_ref,
    )

    result = solver(
        model,
        {(): np.zeros((2, 2), dtype=complex)},
        integration=integration,
        scf=LinearMixing(max_iterations=3, alpha=1.0),
        scf_tol=1e-8,
        filling_tol=1e-10,
    )
    interaction_correction = result.mf[()] + result.density_matrix_result.mu * np.eye(2)

    np.testing.assert_allclose(
        result.density_matrix_result.density_matrix[()],
        rho_ref[()],
        atol=1e-8,
    )
    np.testing.assert_allclose(interaction_correction, np.zeros((2, 2)), atol=1e-8)
