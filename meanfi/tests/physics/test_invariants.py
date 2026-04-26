import numpy as np
import pytest

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    UniformGrid,
    add_tb,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
)
from meanfi.zero_temp import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.helpers import (
    antiferromagnetic_guess,
    bipartite_hubbard_1d,
    duplicated_local_two_band_1d,
    exact_spinful_chain_charge,
    exact_spinful_chain_mu,
    spinful_chain,
)


pytestmark = pytest.mark.physics
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_zero_dimensional_density_matrix_at_mu_matches_exact_occupation():
    h = {(): np.diag([-1.0, 2.0])}
    mu = 0.3
    kT = 0.4

    result = density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=[()],
        integration=AdaptiveQuadrature(),
    )
    expected = np.diag(fermi_dirac(np.array([-1.0, 2.0]), kT, mu))

    assert np.allclose(result.density_matrix[()], expected, atol=1e-12)
    assert np.allclose(result.density_matrix_error[()], np.zeros((2, 2)), atol=0.0)
    assert result.info.n_kernel_evals == 1
    assert result.info.unique_evals == 1
    assert result.info.n_evaluator_evals == 1


def test_density_matrix_respects_hermiticity_and_charge_sum_rule():
    h_0, _h_int = bipartite_hubbard_1d(U=4.0)
    h_trial = add_tb(h_0, antiferromagnetic_guess(0.7, 1))

    result = density_matrix(
        h_trial,
        filling=2.0,
        kT=0.1,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-8),
        filling_tol=1e-8,
    )

    assert np.allclose(
        result.density_matrix[(0,)],
        result.density_matrix[(0,)].conj().T,
        atol=1e-8,
    )
    assert np.allclose(
        result.density_matrix[(-1,)],
        result.density_matrix[(1,)].conj().T,
        atol=1e-8,
    )
    assert abs(np.trace(result.density_matrix[(0,)]).real - 2.0) <= 1e-8
    assert abs(result.filling - 2.0) <= 1e-8


def test_half_filling_keeps_particle_hole_symmetry():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.2,
        keys=[(0,)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-8),
        filling_tol=1e-9,
    )

    assert abs(result.mu) < 5e-7
    assert abs(result.filling - 1.0) < 1e-9


@requires_ext
def test_zero_temperature_fixed_filling_tracks_exact_mu_on_analytic_chain():
    tb = spinful_chain()
    for filling in (0.1, 0.25, 0.5, 0.9, 1.1, 1.5, 1.75, 1.9):
        result = density_matrix(
            tb,
            filling=filling,
            kT=0.0,
            keys=[(0,), (1,)],
            integration=AdaptiveSimplex(
                density_matrix_tol=1e-2,
                max_refinements=600,
            ),
            filling_tol=5e-5,
        )

        assert abs(result.mu - exact_spinful_chain_mu(filling)) <= 1e-4
        assert abs(exact_spinful_chain_charge(result.mu) - filling) <= 3e-5
        assert result.info.refinements > 0


@requires_ext
def test_zero_temperature_fixed_filling_default_mu_iteration_limit_matches_explicit_limit():
    tb = spinful_chain()

    default_result = density_matrix(
        tb,
        filling=0.1,
        kT=0.0,
        keys=[(0,), (1,)],
        integration=AdaptiveSimplex(
            density_matrix_tol=1e-2,
            max_refinements=600,
        ),
        filling_tol=5e-5,
    )
    explicit_result = density_matrix(
        tb,
        filling=0.1,
        kT=0.0,
        keys=[(0,), (1,)],
        integration=AdaptiveSimplex(
            density_matrix_tol=1e-2,
            max_refinements=600,
        ),
        filling_tol=5e-5,
        max_mu_iterations=128,
    )

    assert abs(default_result.mu - explicit_result.mu) <= 1e-12
    assert abs(default_result.filling - explicit_result.filling) <= 1e-12


@requires_ext
def test_zero_temperature_density_is_invariant_under_equivalent_local_supercell():
    primitive, doubled = duplicated_local_two_band_1d()

    primitive_result = density_matrix(
        primitive,
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=AdaptiveSimplex(
            density_matrix_tol=1e-12,
            max_refinements=4,
        ),
        filling_tol=1e-12,
    )
    doubled_result = density_matrix(
        doubled,
        filling=2.0,
        kT=0.0,
        keys=[(0,)],
        integration=AdaptiveSimplex(
            density_matrix_tol=1e-12,
            max_refinements=4,
        ),
        filling_tol=1e-12,
    )

    assert primitive_result.info.error_estimate_available is True
    assert doubled_result.info.error_estimate_available is True
    assert primitive_result.info.refinements == doubled_result.info.refinements == 0
    assert abs(primitive_result.mu - doubled_result.mu) < 1e-12
    assert np.allclose(
        primitive_result.density_matrix[(0,)],
        doubled_result.density_matrix[(0,)][:2, :2],
        atol=1e-12,
    )
    assert np.allclose(
        primitive_result.density_matrix[(0,)],
        doubled_result.density_matrix[(0,)][2:, 2:],
        atol=1e-12,
    )
    assert np.allclose(
        doubled_result.density_matrix[(0,)][:2, 2:],
        np.zeros((2, 2)),
        atol=1e-12,
    )


def test_uniform_grid_reports_unique_eval_count():
    result = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.0,
        keys=[(0,)],
        integration=UniformGrid(nk=7),
    )

    assert result.info.n_kpoints == 7
    assert result.info.unique_evals == 7
