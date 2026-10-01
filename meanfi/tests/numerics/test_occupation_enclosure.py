"""Exercise the experimental charge stage through the full filling/density API."""

import numpy as np

import meanfi as mf


def test_fixed_filling_cosine_density_and_fourier_moment():
    filling = 0.37
    hopping = {(1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    result = mf.density_matrix(
        hopping,
        filling=filling,
        keys=[(0,), (1,)],
        integration=mf.FermiSimplex(max_refinements=500),
        tol=mf.ErrorTolerances(
            scf_residual=1e-4,
            density_matrix_integration=1e-5,
            charge_integration=1e-5,
            filling_residual=1e-7,
            matrix_function_tol=1e-8,
        ),
    )
    density = result.to_tb()
    assert abs(result.filling - filling) < 1e-7
    assert abs(result.mu + np.cos(np.pi * filling)) < 5e-5
    assert abs(density[(0,)][0, 0] - filling) < 1e-5
    # Exact integral of exp(ik) on [acos(mu), 2*pi-acos(mu)].
    assert abs(density[(1,)][0, 0] + np.sin(np.pi * filling) / np.pi) < 2e-5
    assert result.errors.charge_integration <= 1e-5


def test_gapped_projector_density_against_converged_periodic_reference():
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    x = 0.5 * sz - 0.5j * sx
    y = 0.15 * sx + 0.125 * sz - 0.25j * sy
    model = {
        (0, 0): 0.7 * sx + 1.5 * sz,
        (1, 0): x,
        (-1, 0): x.conj().T,
        (0, 1): y,
        (0, -1): y.conj().T,
    }

    def reference(side):
        angles = 2 * np.pi * (np.arange(side) + 0.5) / side
        kx, ky = np.meshgrid(angles, angles, indexing="ij")
        dx = 0.7 - np.sin(kx) + 0.3 * np.cos(ky)
        dy = -0.5 * np.sin(ky)
        dz = 1.5 + np.cos(kx) + 0.25 * np.cos(ky)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        # The occupied projector is (I - d.sigma / |d|)/2; dz >= .25
        # gives a uniform gap, so the smooth periodic reference converges fast.
        return (
            np.eye(2)
            - sum(
                np.mean(component / norm) * matrix
                for component, matrix in zip((dx, dy, dz), (sx, sy, sz))
            )
        ) / 2

    expected = reference(256)
    assert np.linalg.norm(expected - reference(128), 2) < 1e-12
    result = mf.density_matrix_at_mu(model, mu=0, keys=[(0, 0)], tol=1e-4)
    density = result.to_tb()[(0, 0)]
    assert np.linalg.norm(density - expected, 2) < 2e-5
    assert result.errors.density_matrix_integration <= 2e-5
    assert result.errors.density_cut_estimate == 0
    assert result.statistics.p_refinements > 0
