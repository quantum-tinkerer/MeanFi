"""Physical momentum weights agree with independent normalized formulations."""

import numpy as np
import pytest

import meanfi as mf
from meanfi.hamiltonian import add_correction, electron_to_bdg

pytestmark = pytest.mark.integration


def chain():
    return mf.BlochHamiltonian(
        lambda k: np.diag([-2 * np.cos(k) - 0.1, -2 * np.cos(k) + 0.1])
    )


@pytest.mark.parametrize("measure", [0.0017, 2.5])
@pytest.mark.parametrize("temperature", [0.0, 0.2])
def test_transformed_integrals_scale_without_changing_spectral_units(
    measure, temperature
):
    bare = chain()
    transformed = bare.transform(lambda k: (k,), measure=measure)
    kwargs = dict(mu=0.2, kT=temperature, keys=[(0,)], compute_free_energy=True)
    unit = mf.density_matrix_at_mu(bare, **kwargs)
    physical = mf.density_matrix_at_mu(transformed, **kwargs)
    np.testing.assert_allclose(physical.values, measure * unit.values, atol=1e-13)
    assert physical.filling == pytest.approx(measure * unit.filling, abs=1e-13)
    assert physical.entropy == pytest.approx(measure * unit.entropy, abs=1e-13)
    assert physical.mu == unit.mu
    for name in (
        "density_matrix_integration",
        "density_cut_estimate",
        "charge_integration",
        "entropy",
    ):
        left, right = getattr(unit.errors, name), getattr(physical.errors, name)
        if left is not None:
            assert right == pytest.approx(measure * left, abs=1e-14)


@pytest.mark.parametrize("temperature", [0.0, 0.2])
@pytest.mark.parametrize("measure", [0.0017, 2.5])
def test_weighted_scf_matches_rescaled_interaction_reference(measure, temperature):
    bare = chain()
    interaction = {(0,): np.array([[0.0, 0.4], [0.4, 0.0]])}
    reference = {(0,): np.eye(2) * 0.1}
    unit = mf.Model(bare, interaction, filling=0.8, kT=temperature, reference=reference)
    physical = mf.Model(
        bare.transform(lambda k: (k,), measure=measure),
        {key: value / measure for key, value in interaction.items()},
        filling=0.8 * measure,
        kT=temperature,
        reference={key: value * measure for key, value in reference.items()},
    )
    guess = unit.random_meanfield(rng=2, scale=0.05)
    np.testing.assert_allclose(
        physical.random_meanfield(rng=2, scale=0.05)[(0,)], guess[(0,)]
    )
    a = mf.solver(unit, guess, compute_free_energy=True)
    b = mf.solver(physical, guess, compute_free_energy=True)
    assert a.converged and b.converged
    np.testing.assert_allclose(b.density.values, measure * a.density.values, atol=2e-12)
    np.testing.assert_allclose(b.mean_field[(0,)], a.mean_field[(0,)], atol=2e-11)
    assert b.mu == pytest.approx(a.mu, abs=2e-11)
    assert b.internal_energy == pytest.approx(measure * a.internal_energy, abs=2e-12)
    assert b.free_energy == pytest.approx(measure * a.free_energy, abs=2e-12)
    assert b.errors.scf_residual <= measure * 1e-3


def test_explicit_absolute_targets_are_not_rescaled_by_public_policy():
    measure = 0.02
    h = mf.BlochHamiltonian(lambda k: np.array([[-2 * np.cos(k)]]))
    h = h.transform(lambda k: (k,), measure=measure)
    targets = mf.default_solver_tolerances(1e-5)
    result = mf.density_matrix(h, filling=0.013, keys=[(0,)], tol=targets)
    exact = measure * np.arccos(-result.mu / 2) / np.pi
    assert abs(result.filling - exact) <= result.errors.charge_integration + 1e-12
    assert result.errors.charge_integration <= targets.charge_integration
    assert result.errors.filling_residual <= targets.filling_residual
    assert (
        result.errors.density_matrix_integration <= targets.density_matrix_integration
    )
    with pytest.raises(ValueError, match="filling"):
        mf.Model(h, {(0,): np.zeros((1, 1))}, filling=0.03)


def test_transform_composition_and_hamiltonian_wrappers_preserve_measure():
    h = mf.BlochHamiltonian(lambda q: np.array([[q, 0], [0, -q]]))
    mapped = h.transform(lambda k: (2 * k,), measure=0.2).transform(
        lambda k: (k + 1,), measure=3.0
    )
    np.testing.assert_allclose(mapped(0.4), h(2 * 1.4))
    assert mapped.measure == pytest.approx(0.6)
    corrected = add_correction(
        mapped, {(0,): np.eye(2), (1,): np.eye(2), (-1,): np.eye(2)}
    )
    np.testing.assert_allclose(
        corrected(0.4), mapped(0.4) + (1 + 2 * np.cos(0.4)) * np.eye(2)
    )
    assert corrected.measure == mapped.measure
    assert electron_to_bdg(mapped).measure == mapped.measure


@pytest.mark.parametrize("measure", [0, -1, np.nan, np.inf])
def test_invalid_transform_measures_are_rejected(measure):
    with pytest.raises(ValueError, match="measure"):
        chain().transform(lambda k: (k,), measure=measure)


def test_weighted_bdg_density_energy_and_entropy_use_the_same_measure():
    measure = 0.04
    bare = chain()
    interaction = {(0,): np.array([[0.0, -1.0], [-1.0, 0.0]])}
    a = mf.Model(bare, interaction, filling=1, kT=0.2, superconducting=True)
    b = mf.Model(
        bare.transform(lambda k: (k,), measure=measure),
        {key: value / measure for key, value in interaction.items()},
        filling=measure,
        kT=0.2,
        superconducting=True,
    )
    delta = np.array([[0.0, 0.2], [-0.2, 0.0]])
    correction = {
        (0,): np.block([[np.zeros((2, 2)), delta], [-delta, np.zeros((2, 2))]])
    }
    kwargs = dict(mu=0.15, mean_field=correction, compute_free_energy=True)
    unit = mf.density_matrix_at_mu(a, **kwargs)
    physical = mf.density_matrix_at_mu(b, **kwargs)
    np.testing.assert_allclose(physical.values, measure * unit.values, atol=1e-13)
    assert physical.filling == pytest.approx(measure * unit.filling, abs=1e-13)
    assert physical.band_energy == pytest.approx(measure * unit.band_energy, abs=1e-13)
    assert physical.free_energy == pytest.approx(measure * unit.free_energy, abs=1e-13)
    assert physical.entropy == pytest.approx(measure * unit.entropy, abs=1e-13)
