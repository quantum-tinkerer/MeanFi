"""EDIIS defers costly energy work while returning physical final observables."""

import numpy as np
import pytest

import meanfi as mf
import meanfi.density.integrate.simplex as simplex
from meanfi.scf.problem import SCFProblem


@pytest.mark.parametrize("limit", [1, 30])
def test_simplex_energy_is_evaluated_only_on_final_mesh(monkeypatch, limit):
    model = mf.Model(
        {
            (0,): np.diag([-0.6, 0.6]),
            (1,): np.diag([0.05, -0.05]),
            (-1,): np.diag([0.05, -0.05]),
        },
        mf.BilinearInteraction([mf.BilinearTerm(0.4, np.eye(2), np.eye(2))]),
        filling=1,
    )
    calls, samples = [], []
    integrate = simplex.integrate_energies
    evaluate = SCFProblem.evaluate_mean_field

    def energy(*args, **kwargs):
        calls.append(1)
        return integrate(*args, **kwargs)

    def density(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        assert result.band_energy is None
        assert result.statistics.n_energy_evaluations == 0
        samples.append((result.values.copy(), result.mu))
        return result

    monkeypatch.setattr(simplex, "integrate_energies", energy)
    monkeypatch.setattr(SCFProblem, "evaluate_mean_field", density)
    try:
        result = mf.solver(
            model,
            model.random_meanfield(rng=12, scale=0.1),
            scf=mf.EnergyDIIS(max_iterations=limit),
            tol=1e-5,
        )
        assert limit == 30
    except mf.NoConvergence as error:
        assert limit == 1
        result = error.result
    assert len(calls) == 1
    assert np.isfinite(result.internal_energy)
    assert result.density._energy_evaluation is None
    np.testing.assert_array_equal(result.density.values, samples[-1][0])
    assert result.mu == samples[-1][1]
    assert all(item.internal_energy is None for item in result.history[:-1])
    assert result.history[-1].internal_energy == result.internal_energy


def test_final_energy_failure_retains_valid_density(monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("energy quadrature failed")

    monkeypatch.setattr(simplex, "integrate_energies", failure)
    model = mf.Model(
        {(0,): np.diag([-1.0, 1.0])},
        {(0,): np.array([[0.0, 0.2], [0.2, 0.0]])},
        filling=1,
    )
    with pytest.raises(
        mf.SolverFailure, match="Final energy calculation failed"
    ) as error:
        mf.solver(model, {(0,): np.zeros((2, 2))})
    result = error.value.result
    assert result.density._energy_evaluation is None
    assert result.internal_energy is None
    np.testing.assert_allclose(
        result.density.values,
        [np.diag([1, 0])[i, j] for _, i, j in result.density.coordinates.entries],
        atol=1e-12,
    )
