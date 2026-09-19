"""Density-response comparisons against converged integrals and exact limits."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import meanfi as mf
from meanfi.scf.energy import (
    EnergySample,
    comparison_points,
    energy_sample,
    relative_energy,
)
from meanfi.tests.integration.test_intercell_bdg import _chain_model


def sample(model, field, density):
    evaluation = SimpleNamespace(
        density=density,
        mean_field=field,
        output_state=model._density_state(density),
        internal_energy=density.internal_energy,
    )
    return energy_sample(model, evaluation)


@pytest.mark.parametrize("finite_range", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_response_converges_to_dense_internal_energy_difference(
    finite_range, reference
):
    onsite = np.diag([-0.7, 0.7])
    hop = np.array([[0.03, 0.10], [0.04, -0.03]])
    a = np.array([[1, 0.2j], [-0.2j, 0.3]])
    b = np.array([[0.1, 0.3], [0.3, 0.8]])
    terms = [mf.BilinearTerm(0.2, a, b)]
    if finite_range:
        terms.append(mf.BilinearTerm(0.1, a, b, displacement=(1,)))
    model = mf.Model(
        {(0,): onsite, (1,): hop, (-1,): hop.T},
        mf.BilinearInteraction(terms),
        filling=1,
    )
    if reference:
        ref = mf.density_matrix_at_mu(model, 0, integration=mf.UniformGrid(nk=512))
        model = replace(model, reference=ref)
    direction = model.random_meanfield(rng=4, scale=1)
    left_field = {k: 0.03 * v for k, v in direction.items()}

    def evaluate(field, nk=512):
        return mf.density_matrix_at_mu(
            model, 0, mean_field=field, integration=mf.UniformGrid(nk=nk), tol=1e-12
        )

    left_density = evaluate(left_field)
    left = replace(sample(model, left_field, left_density), internal_energy=None)
    errors = []
    for step in [0.04, 0.02]:
        field = {k: v + step * direction[k] for k, v in left_field.items()}
        density = evaluate(field)
        fine = evaluate(field, 1024)
        assert abs(fine.internal_energy - density.internal_energy) < 2e-13
        exact = density.internal_energy - left_density.internal_energy
        right = replace(sample(model, field, density), internal_energy=None)
        errors.append(abs(relative_energy(model, left, right) - exact))
    # Endpoint quadrature has cubic local error on this smooth gapped path.
    assert errors[1] < errors[0] / 5, errors
    assert errors[1] < 1e-5, errors
    print(
        f"finite_range={finite_range}, reference={reference}: response errors={errors}"
    )


def test_signed_filling_term_and_expiring_anchor():
    model = mf.Model({(): np.diag([-1.0, 1.0])}, {(): np.zeros((2, 2))}, filling=1)
    history = [
        EnergySample(
            np.zeros(model._space.num_params), {(): np.zeros((2, 2))}, 0, n, mu, None
        )
        for n, mu in [(0, -2), (1, 0), (2, 2)]
    ]
    np.testing.assert_allclose(
        [p.energy for p in comparison_points(model, history)], [0, 0.5, 0], atol=1e-14
    )
    np.testing.assert_allclose(
        [p.energy for p in comparison_points(model, history[1:])], [0, -0.5], atol=1e-14
    )
    shifted = [replace(p, mu=p.mu + 7) for p in history]
    np.testing.assert_array_equal(
        [p.energy for p in comparison_points(model, shifted)],
        [p.energy for p in comparison_points(model, history)],
    )


@pytest.mark.parametrize("paired", [False, True])
def test_thermal_free_energy_comparisons_match_dense_reference(paired):
    model = _chain_model(paired, 0.2)
    integration = mf.UniformGrid(nk=128)
    records = []
    exact = []
    for seed in [4, 5, 6]:
        field = model.random_meanfield(rng=seed, scale=0.1)
        density = mf.density_matrix(
            model,
            mean_field=field,
            keys=[(0,), (1,), (-1,)],
            integration=integration,
            tol=1e-10,
            compute_free_energy=True,
        )
        fine = mf.density_matrix(
            model,
            mean_field=field,
            keys=[(0,), (1,), (-1,)],
            integration=mf.UniformGrid(nk=256),
            tol=1e-10,
            compute_free_energy=True,
        )
        assert abs(fine.internal_energy - density.internal_energy) < 2e-12
        assert abs(fine.free_energy - density.free_energy) < 2e-12
        exact.append(mf.evaluate_free_energy(model, density))
        records.append(sample(model, field, density))
    actual = [p.energy for p in comparison_points(model, records)]
    np.testing.assert_allclose(actual, np.asarray(exact) - exact[0], atol=3e-13, rtol=0)


def test_thermal_response_needs_entropy_term_to_recover_internal_energy():
    model = mf.Model(
        {(): np.diag([-0.3, 0.3])},
        {(): np.array([[0.0, 0.2], [0.2, 0.0]])},
        filling=1,
        kT=0.4,
    )
    records, densities = [], []
    for mass in [0.01, 0.011]:
        field = {(): np.diag([-mass, mass])}
        density = mf.density_matrix(
            model,
            mean_field=field,
            integration=mf.UniformGrid(),
            tol=1e-12,
            compute_free_energy=True,
        )
        densities.append(density)
        records.append(sample(model, field, density))
    exact = densities[1].internal_energy - densities[0].internal_energy
    # Deliberately apply the zero-T identity to thermal states to quantify the
    # missing term, using the same physical energy functional.
    naive = relative_energy(
        replace(model, kT=0),
        replace(records[0], internal_energy=None),
        replace(records[1], internal_energy=None),
    )
    thermal_term = model.kT * (densities[1].entropy - densities[0].entropy)
    assert abs(naive + thermal_term - exact) < 1e-8
    assert abs(naive - exact) > 1e-5
    assert relative_energy(model, *records) == pytest.approx(
        exact - thermal_term, abs=1e-14
    )
    with pytest.raises(ValueError, match="requires entropy"):
        relative_energy(model, records[0], replace(records[1], entropy=None))


def test_thermal_free_energy_uses_signed_filling_correction_once():
    model = mf.Model(
        {(): np.diag([-1.0, 1.0])}, {(): np.zeros((2, 2))}, filling=1, kT=0.2
    )
    left = EnergySample(np.zeros(model._space.num_params), {}, 0, 0.9, -0.3, 0.7, 0.4)
    right = replace(left, filling=1.2, mu=0.5, internal_energy=0.9, entropy=0.6)
    expected = (0.9 - 0.7) - 0.2 * (0.6 - 0.4) - (0.5 * 0.2 - (-0.3) * (-0.1)) / 2
    assert relative_energy(model, left, right) == pytest.approx(expected, abs=1e-14)
