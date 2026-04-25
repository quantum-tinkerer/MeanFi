import numpy as np
import pytest

from meanfi import AdaptiveQuadrature, LinearMixing, Model, solver


pytestmark = pytest.mark.physics


def _pairing(delta: complex):
    return {(0, 0): np.array([[0.0, delta], [np.conj(delta), 0.0]], dtype=complex)}


def _local_gap_reference(*, coupling: float, kT: float) -> float:
    lower = 1e-12
    upper = coupling
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        residual = midpoint - 0.5 * coupling * np.tanh(midpoint / (2.0 * kT))
        if residual > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return float(0.5 * (lower + upper))


def test_bdg_solver_matches_2d_local_gap_equation():
    coupling = 1.0
    kT = 0.2
    local = (0, 0)

    def bdg_meanfield(density):
        return _pairing(-coupling * density[local][0, 1])

    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=kT,
        superconducting=True,
        bdg_meanfield=bdg_meanfield,
    )

    result = solver(
        model,
        _pairing(0.3),
        integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
        scf=LinearMixing(max_iterations=80, alpha=0.8),
        scf_tol=1e-6,
        filling_tol=1e-6,
    )

    assert result.info.residual_norm <= 1e-6
    assert abs(result.density_matrix_result.filling - model.filling) <= 1e-6
    assert abs(result.mf[local][0, 1].real - _local_gap_reference(coupling=coupling, kT=kT)) <= 5e-5
