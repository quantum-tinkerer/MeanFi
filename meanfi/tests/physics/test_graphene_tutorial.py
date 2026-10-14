"""Independent exact limits for the executable continuum paper tutorial."""

from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.linalg import expm

import meanfi as mf
from docs.source.tutorial.scripts.rhombohedral_graphene import (
    Graphene,
    IDENTITY,
    LOCAL,
    PAULI,
    disk_coordinates,
    graphene_interaction,
)
from docs.source.tutorial.scripts.graphene_observables import band_edges, mass_rule
from meanfi.tests.fixtures.graphene_reference import insulating_reference

pytestmark = pytest.mark.physics


@pytest.fixture(scope="module")
def tutorial_cells():
    source = (
        Path(__file__).resolve().parents[3]
        / "docs/source/tutorial/rhombohedral_graphene.md"
    )
    return re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", source.read_text(), re.S
    )


@pytest.fixture(scope="module")
def tutorial_definitions(tutorial_cells):
    namespace = dict(np=np, mf=mf)
    for name in ("disk_coordinates", "graphene_interaction"):
        cell = next(cell for cell in tutorial_cells if f"def {name}(" in cell)
        exec(compile(cell, "graphene tutorial", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def tutorial_callback(tutorial_cells, tutorial_definitions):
    """Test the actual compiled callback shown in the notebook, without SCF."""
    numba = pytest.importorskip("numba")
    namespace = dict(tutorial_definitions, njit=numba.njit, Graphene=Graphene)
    cell = next(cell for cell in tutorial_cells if "def disk_hamiltonian(" in cell)
    exec(compile(cell, "graphene tutorial", "exec"), namespace)
    return namespace["disk_hamiltonian"]


def test_inline_interaction_matches_exact_two_particle_energies(tutorial_definitions):
    interaction = tutorial_definitions["graphene_interaction"](
        sites=1,
        u=40,
        v=-8,
        hund=5,
        measure=1,
    )
    model = mf.Model({LOCAL: np.zeros((4, 4))}, interaction, filling=2)
    for occupations, expected in [([1, 0, 1, 0], 27), ([1, 0, 0, 1], 37)]:
        energy = 4 * mf.evaluate_internal_energy(model, {LOCAL: np.diag(occupations)})
        assert abs(energy - expected) < 1e-13


def test_isolated_dimers_and_public_disk_integration():
    p = Graphene(gamma0=0, gamma2=0, gamma3=0, gamma4=0, offset=0, delta=0.02, soc=0)
    h = p.hamiltonian(0.027, -0.053)
    energies, vectors = np.linalg.eigh(h)
    occupied = vectors[:, :20]
    exact = occupied @ occupied.conj().T
    result = mf.density_matrix(
        p.bloch_hamiltonian(),
        filling=20,
        keys=[LOCAL],
        integration=mf.FermiSimplex(nk=25),
        tol=1e-9,
    )
    error = np.max(np.abs(result.to_tb()[LOCAL] - exact))
    assert error < 2e-12, f"Constant-spectrum disk density error: {error}"
    assert result.band_energy == pytest.approx(energies[:20].sum() / 40, abs=2e-14)
    dimers = replace(p, delta=0)
    expected = [-0.38] * 16 + [0.0] * 8 + [0.38] * 16
    np.testing.assert_allclose(
        np.linalg.eigvalsh(dimers.hamiltonian(0.02, -0.03)), expected, atol=1e-14
    )


def test_time_reversal_chiral_limit_and_offset_convention():
    p = Graphene()
    reverse = np.kron(np.eye(10), np.kron(PAULI[0], 1j * PAULI[1]))
    h = p.hamiltonian(0.032, -0.071)
    np.testing.assert_allclose(
        reverse @ h.conj() @ reverse.conj().T, p.hamiltonian(-0.032, 0.071), atol=1e-14
    )
    chiral = replace(p, gamma4=0, offset=0, delta=0, soc=0)
    h = chiral.hamiltonian(0.034, 0.02)
    symmetry = np.kron(np.diag([1.0, -1.0] * 5), np.eye(4))
    np.testing.assert_allclose(symmetry @ h @ symmetry, -h, atol=1e-14)
    dimer = replace(p, delta=0, soc=0).hamiltonian(0, 0).diagonal().real.reshape(10, 4)
    outer = (
        replace(p, delta=0, soc=0, offset_sites="outer")
        .hamiltonian(0, 0)
        .diagonal()
        .real.reshape(10, 4)
    )
    np.testing.assert_allclose(dimer[[0, -1]], 0)
    np.testing.assert_allclose(outer[[0, -1]], p.offset)
    np.testing.assert_allclose(dimer + outer, p.offset)


def test_paper_hund_normalization_and_spin_rotation():
    interaction = graphene_interaction(sites=1, u=40, v=-8, hund=5, measure=1)
    model = mf.Model({LOCAL: np.zeros((4, 4))}, interaction, filling=2)
    parallel = {LOCAL: np.diag([1.0, 0.0, 1.0, 0.0])}
    antiparallel = {LOCAL: np.diag([1.0, 0.0, 0.0, 1.0])}
    # Exact two-particle energies: U + V - J and U + V + J.
    assert 4 * mf.evaluate_internal_energy(model, parallel) == pytest.approx(
        27.0, abs=1e-13
    )
    assert 4 * mf.evaluate_internal_energy(model, antiparallel) == pytest.approx(
        37.0, abs=1e-13
    )
    rotation = np.kron(IDENTITY, expm(1j * (0.3 * PAULI[0] + 0.2 * PAULI[1])))
    rotated = {LOCAL: rotation @ parallel[LOCAL] @ rotation.conj().T}
    assert mf.evaluate_internal_energy(model, rotated) == pytest.approx(
        mf.evaluate_internal_energy(model, parallel), abs=1e-13
    )
    np.testing.assert_allclose(
        model.mean_field(rotated)[LOCAL],
        rotation @ model.mean_field(parallel)[LOCAL] @ rotation.conj().T,
        atol=1e-13,
    )


def test_disk_map_measure_and_affine_hamiltonian():
    p = Graphene()
    assert p.measure == pytest.approx(0.0017642524653497347)
    assert p.model().filling == 20
    np.testing.assert_allclose(
        disk_coordinates(np.pi, np.pi, p.cutoff), [0, 0], atol=1e-14
    )
    np.testing.assert_allclose(
        disk_coordinates(np.pi, 2 * np.pi, p.cutoff), [0, p.cutoff], atol=1e-14
    )
    for k1, k2 in [(0, 0), (1.2, 0.7), (2 * np.pi, 2 * np.pi)]:
        q = disk_coordinates(k1, k2, p.cutoff)
        np.testing.assert_allclose(
            p.bloch_hamiltonian()(k1, k2), p.hamiltonian(*q), atol=1e-14
        )


def test_independent_reference_and_band_edges_constant_spectrum():
    p = Graphene(gamma0=0, gamma2=0, gamma3=0, gamma4=0, offset=0, soc=0, delta=0.02)
    h = p.hamiltonian(0, 0)
    energy, vectors = np.linalg.eigh(h)
    result = insulating_reference(
        p, np.zeros_like(h), 0.0, radial_order=5, angular_order=12
    )
    exact = vectors[:, :20] @ vectors[:, :20].conj().T
    np.testing.assert_allclose(result["density"], exact, atol=2e-14)
    assert result["band_energy_average"] == pytest.approx(sum(energy[:20]), abs=2e-14)
    edges = band_edges(p, np.zeros_like(h), radial_points=3, angular_points=6)
    assert edges["indirect_gap_meV"] == pytest.approx(
        1000 * (energy[20] - energy[19]), abs=1e-10
    )
    with pytest.raises(ValueError, match="global gap"):
        insulating_reference(p, np.zeros_like(h), 1.0, radial_order=5, angular_order=12)


def test_mass_rule_refuses_metal_and_spin_coherence():
    p = Graphene(delta=0, soc=0, gamma2=0, gamma3=0, gamma4=0, offset=0)
    correction = p.seed("qah0")[LOCAL]
    assert mass_rule(p, correction, mu=0, indirect_gap_meV=1)["conditional_chern"] == 5
    assert (
        mass_rule(p, correction, mu=0, indirect_gap_meV=-1)["conditional_chern"] is None
    )
    assert (
        mass_rule(p, p.seed("laf_x")[LOCAL], mu=0, indirect_gap_meV=1)[
            "conditional_chern"
        ]
        is None
    )


def test_selected_local_observables_match_full_density():
    from docs.source.tutorial.scripts.graphene_observables import local_blocks

    p = Graphene(layers=2)
    model = p.model()
    kwargs = dict(
        mu=0.1,
        mean_field=model.random_meanfield(rng=5, scale=0.03),
        integration=mf.FermiSimplex(nk=25),
    )
    selected = mf.density_matrix_at_mu(model, **kwargs)
    full = mf.density_matrix_at_mu(model, keys=[LOCAL], **kwargs).to_tb()[LOCAL]
    expected = np.array([full[4 * i : 4 * i + 4, 4 * i : 4 * i + 4] for i in range(4)])
    np.testing.assert_allclose(local_blocks(p, selected), expected, atol=2e-14)


@pytest.mark.perf_slow
def test_tutorial_parameter_points_converge_from_common_random_guess(tutorial_callback):
    tol = replace(
        mf.default_solver_tolerances(5e-3),
        charge_integration=2e-2,
        filling_residual=5e-3,
    )
    for delta, hund in [(-0.012, 10), (-0.015, 10), (-0.015, 5), (-0.019, 5)]:
        p = Graphene(delta=delta, hund=hund)
        model = mf.Model(
            mf.BlochHamiltonian(tutorial_callback(p)),
            graphene_interaction(sites=10, u=40, v=-8, hund=hund, measure=p.measure),
            filling=20,
        )
        result = mf.solver(
            model,
            model.random_meanfield(rng=81, scale=0.025),
            integration=mf.FermiSimplex(),
            scf=mf.EnergyDIIS(),
            tol=tol,
        )
        assert result.converged
        assert result.density.statistics.requested_nk is None
        assert result.density.statistics.error_estimate_available
        assert (
            result.errors.density_matrix_integration <= tol.density_matrix_integration
        )
        assert result.errors.scf_residual <= tol.scf_residual
        assert np.isfinite(result.internal_energy)


def test_concentric_map_has_constant_jacobian_and_inversion_symmetry():
    cutoff = 0.16
    step = 1e-5
    for point in np.array([[0.4, 2.3], [2.3, 0.4], [4.0, 5.2], [5.2, 4.0]]):
        q = np.asarray(disk_coordinates(*point, cutoff))
        np.testing.assert_allclose(
            disk_coordinates(*(2 * np.pi - point), cutoff), -q, atol=1e-14
        )
        columns = []
        for direction in step * np.eye(2):
            plus = np.asarray(disk_coordinates(*(point + direction), cutoff))
            minus = np.asarray(disk_coordinates(*(point - direction), cutoff))
            columns.append((plus - minus) / (2 * step))
        determinant = np.linalg.det(np.column_stack(columns))
        assert determinant == pytest.approx(cutoff**2 / (4 * np.pi), rel=1e-8)


def test_tutorial_numba_callback_matches_original_model(tutorial_callback):
    points = np.vstack(
        [
            np.random.default_rng(18).uniform(0, 2 * np.pi, (30, 2)),
            [
                [0, 0],
                [0, 2 * np.pi],
                [2 * np.pi, 0],
                [2 * np.pi, 2 * np.pi],
                [np.pi, np.pi],
            ],
        ]
    )
    maximum = 0.0
    for p in [Graphene(), Graphene(layers=3, cutoff=0.2, delta=0.03)]:
        reference = p.bloch_hamiltonian()
        actual = mf.BlochHamiltonian(tutorial_callback(p))
        for point in points:
            maximum = max(
                maximum, float(np.max(abs(actual(*point) - reference(*point))))
            )
    assert maximum < 1e-14, maximum


def test_band_plot_retains_intervalley_mixing():
    plt = pytest.importorskip("matplotlib.pyplot")
    from types import SimpleNamespace
    from docs.source.tutorial.scripts.graphene_figure import plot_bands

    p = Graphene(
        gamma0=0, gamma1=0, gamma2=0, gamma3=0, gamma4=0, offset=0, delta=0, soc=0
    )
    field = 0.01 * np.kron(np.eye(10), np.kron(PAULI[0], np.eye(2)))
    result = SimpleNamespace(mean_field={LOCAL: field}, mu=0.002)
    fig = plot_bands([(p, result)])
    try:
        points = np.asarray(fig.axes[0].collections[0].get_offsets()).reshape(-1, 8, 2)
        # The exact valley-mixed spectrum is ±10 meV, shifted by mu=2 meV.
        expected = np.array([-12.0] * 4 + [8.0] * 4)
        assert np.max(abs(points[:, :, 1] - expected)) < 1e-12
    finally:
        plt.close(fig)
