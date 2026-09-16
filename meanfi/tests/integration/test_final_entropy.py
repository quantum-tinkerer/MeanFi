"""Entropy is optional output, never part of the SCF map or its stopping rule."""

from dataclasses import asdict

import numpy as np
import pytest
from scipy import sparse
from scipy.special import entr, expit

import meanfi as mf

pytestmark = pytest.mark.integration


def _model(use_sparse=False):
    h = np.array([[-0.3, 0.07j], [-0.07j, 0.7]])
    interaction = np.array([[0.0, 0.7], [0.7, 0.0]])
    if use_sparse:
        h, interaction = sparse.csr_matrix(h), sparse.csr_matrix(interaction)
    return mf.Model({(): h}, {(): interaction}, filling=0.8, kT=0.2)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("converges", [False, True])
def test_scf_computes_entropy_only_once_after_termination(
    monkeypatch, request, use_sparse, converges
):
    from meanfi.scf.problem import SCFProblem
    import meanfi.density.integrate.periodic_grid as periodic
    import meanfi.density.kpoint.matrix_functions.rational.prepared_sparse as rational

    if use_sparse:
        request.getfixturevalue("require_mumps")
    model = _model(use_sparse)
    records, entropy_calls = [], []
    original_evaluate = SCFProblem.evaluate_mean_field
    original_entropy = (
        rational.fit_entropy if use_sparse else periodic.occupation_entropy
    )
    enabled = False

    def entropy(*args, **kwargs):
        assert enabled, "Entropy work leaked into an SCF iteration"
        entropy_calls.append(True)
        return original_entropy(*args, **kwargs)

    def evaluate(self, mean_field, *args, **kwargs):
        nonlocal enabled
        enabled = kwargs.get("compute_entropy", False)
        records.append((mean_field, kwargs.copy()))
        return original_evaluate(self, mean_field, *args, **kwargs)

    monkeypatch.setattr(SCFProblem, "evaluate_mean_field", evaluate)
    monkeypatch.setattr(
        rational if use_sparse else periodic,
        "fit_entropy" if use_sparse else "occupation_entropy",
        entropy,
    )

    def solve(compute):
        try:
            result = mf.solver(
                model,
                {(): np.zeros((2, 2))},
                integration=mf.UniformGrid(),
                scf=mf.LinearMixing(alpha=0.5, max_iterations=100 if converges else 1),
                tol=1e-7,
                compute_free_energy=compute,
            )
            assert converges
        except mf.NoConvergence as exc:
            assert not converges
            result = exc.result
        return result

    result = solve(True)
    assert len(entropy_calls) == 1
    assert all(not kwargs.get("compute_entropy", False) for _, kwargs in records[:-1])
    assert records[-1][1]["compute_entropy"] is True
    assert records[-1][1]["mu"] == result.mu
    # Even on failure, evaluate the input Hamiltonian of the returned density.
    assert records[-1][0] is records[-2][0]
    matrix = model.hamiltonian_from_meanfield(records[-1][0])[()]
    matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    occupations = expit((result.mu - eigenvalues) / model.kT)
    exact_density = (eigenvectors * occupations) @ eigenvectors.conj().T
    exact_entropy = np.mean(entr(occupations) + entr(1 - occupations))
    np.testing.assert_allclose(
        result.density.values,
        result.density.coordinates.values_from_assembled_matrix(exact_density),
        atol=1e-8,
        rtol=0,
    )
    assert abs(result.entropy - exact_entropy) <= (result.errors.entropy or 0) + 1e-12
    assert all(step.errors.entropy is None for step in result.history)
    assert asdict(result)["density"]["entropy"] == result.entropy

    records.clear()
    entropy_calls.clear()
    lean = solve(False)
    assert not entropy_calls
    assert all(not kwargs.get("compute_entropy", False) for _, kwargs in records)
    assert lean.entropy is lean.free_energy is lean.errors.entropy is None
    assert lean.history == result.history
    assert lean.internal_energy == result.internal_energy
    np.testing.assert_array_equal(lean.density.values, result.density.values)


@pytest.mark.parametrize("method,kT", [(mf.FermiSimplex(), 0), (mf.UniformGrid(), 0.2)])
def test_standalone_density_entropy_flag_and_common_errors(method, kT):
    model = mf.Model(
        {(): np.diag([-0.3, 0.3])}, {(): np.zeros((2, 2))}, filling=1, kT=kT
    )
    for evaluate in (
        lambda **kw: mf.density_matrix(model, **kw),
        lambda **kw: mf.density_matrix_at_mu(model, 0, **kw),
    ):
        full = evaluate(integration=method)
        lean = evaluate(integration=method, compute_free_energy=False)
        assert full.entropy is not None
        assert full.errors.entropy == 0
        assert lean.entropy is lean.free_energy is lean.errors.entropy is None
        assert full.internal_energy == lean.internal_energy
        np.testing.assert_array_equal(full.values, lean.values)
        assert asdict(full)["entropy"] == full.entropy
        with pytest.raises(ValueError, match="entropy"):
            mf.free_energy(model, lean)


def test_final_entropy_failure_preserves_valid_result(monkeypatch):
    import meanfi.scf.problem as problem

    original = problem.evaluate_density

    def evaluate(*args, **kwargs):
        if kwargs.get("compute_entropy"):
            raise mf.ConvergenceError("synthetic entropy failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(problem, "evaluate_density", evaluate)
    with pytest.raises(mf.SolverFailure, match="Final entropy") as caught:
        mf.solver(_model(), {(): np.zeros((2, 2))})
    assert caught.value.result.converged
    assert caught.value.result.internal_energy is not None
    assert caught.value.result.entropy is None
    assert isinstance(caught.value.__cause__, mf.ConvergenceError)


@pytest.mark.parametrize(
    "settings", [mf.UniformGrid(nk=4), mf.UniformGrid(initial_nk=4)]
)
def test_finite_zero_temperature_bdg_ignores_mesh_settings(settings):
    model = mf.Model(
        {(): np.diag([-1.0, 1.0])},
        {(): np.zeros((2, 2))},
        filling=1,
        superconducting=True,
    )
    reference = mf.density_matrix(model)
    with pytest.warns(UserWarning, match="Finite systems do not use nk"):
        actual = mf.density_matrix(model, integration=settings)
    np.testing.assert_array_equal(actual.values, reference.values)
    assert actual.errors == reference.errors
    assert actual.statistics.requested_nk is None
    assert actual.statistics.n_kpoints == 1
