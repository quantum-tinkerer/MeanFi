from meanfi.density.kpoint.matrix_functions.rational.common import SparseRationalLayout
import numpy as np
import pytest
import scipy.sparse as sp

from meanfi import (
    DirectDiagonalization,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
)
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


@pytest.mark.perf_slow
@pytest.mark.usefixtures("require_mumps")
def test_sparse_rational_fixed_filling_matches_dense_reference():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    keys = [(0,), (1,), (-1,)]
    dense_result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=DirectDiagonalization(),
        ),
        tol=1e-9,
        filling_tol=1e-9,
        mu_tol=1e-8,
    )
    sparse_result = density_matrix(
        sparse_tb,
        filling=1.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=RationalFOE(initial_poles=4, max_poles=64),
        ),
        tol=1e-9,
        filling_tol=1e-9,
        mu_tol=1e-8,
    )

    assert abs(sparse_result.mu - dense_result.mu) <= 1e-8
    assert abs(sparse_result.filling - dense_result.filling) <= 1e-8
    for key in keys:
        assert (
            np.max(np.abs(sparse_result.to_tb()[key] - dense_result.to_tb()[key]))
            <= 5e-4
        )


@pytest.mark.perf_slow
@pytest.mark.usefixtures("require_mumps")
def test_sparse_rational_fixed_mu_matches_dense_reference():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    keys = [(0,), (1,), (-1,)]
    dense_result = density_matrix_at_mu(
        spinful_chain(),
        tol=1e-9,
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=DirectDiagonalization(),
        ),
    )
    sparse_result = density_matrix_at_mu(
        sparse_tb,
        tol=1e-9,
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=RationalFOE(initial_poles=4, max_poles=64),
        ),
    )
    for key in keys:
        assert (
            np.max(np.abs(sparse_result.to_tb()[key] - dense_result.to_tb()[key]))
            <= 1e-8
        )


@pytest.mark.perf_slow
@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_rational_mumps_prepared_node_matches_solve_backend():
    from meanfi.space.coordinates import full_density_coordinates
    from meanfi.density.kpoint.matrix_functions.common import shift_by_mu
    from scipy.special import expit
    from meanfi.density.kpoint.matrix_functions.rational import (
        PreparedMumpsRationalNode,
    )

    matrix = sp.csr_matrix(
        np.array(
            [[0.2, 0.15 + 0.05j], [0.15 - 0.05j, -0.2]],
            dtype=complex,
        )
    )
    options = RationalFOE(initial_poles=4, max_poles=64)
    q_diag = np.array([1.0, -1.0], dtype=float)
    trace_weights = np.array([1.0, 0.0], dtype=float)
    coords = full_density_coordinates([tuple()], size=2)

    mumps_node = PreparedMumpsRationalNode(
        matrix,
        kT=0.2,
        q_diag=q_diag,
        options=options,
        charge_tolerance=1e-9,
        layout=SparseRationalLayout.build(
            density_coordinates=coords,
            trace_weights_diag=trace_weights,
            include_all_diagonal=False,
        ),
        matrix_function_tol=1e-9,
    )

    for mu in (0.05, -0.3):
        eigenvalues, eigenvectors = np.linalg.eigh(
            shift_by_mu(matrix, mu, q_diag).toarray()
        )
        direct_density = (
            eigenvectors * expit(-eigenvalues / 0.2)
        ) @ eigenvectors.conj().T
        reference_charge = float(
            np.real(np.sum(trace_weights * np.diag(direct_density)))
        )
        reference_values = coords.values_from_assembled_matrix(direct_density)
        mumps_charge = mumps_node.charge(mu)
        mumps_density = mumps_node.density_values_from_charge_order(mu)
        assert abs(mumps_charge - reference_charge) <= 1e-8
        assert np.max(np.abs(mumps_density - reference_values)) <= 1e-8
    with pytest.raises(ValueError, match="Evaluate charge at the requested mu"):
        mumps_node.density_values_from_charge_order(0.05)
