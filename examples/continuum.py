"""A local Dirac interaction using a user-defined map to a momentum disk.

Integrals are normalized disk averages. Couplings below are defined in that
normalization; physical momentum-space measures must be converted explicitly.
"""

import numpy as np

import meanfi as mf


MASS, CUTOFF, FILLING = 0.2, 0.16, 1.15


def hamiltonian(kx, ky):
    """Equal-measure map from BZ coordinates to a disk of radius CUTOFF."""
    radius = CUTOFF * np.sqrt(kx / (2 * np.pi))
    angle = ky
    z = radius * np.exp(1j * angle) / CUTOFF
    return np.array([[MASS, z.conjugate()], [z, -MASS]])


def main():
    h0 = mf.BlochHamiltonian(hamiltonian)
    interaction = mf.BilinearInteraction(
        [mf.BilinearTerm(0.1, np.diag([1, 0]), np.diag([0, 1]))]
    )
    integration = mf.FermiSimplex(initial_nk=81, max_points=100_000)
    bare = mf.density_matrix(
        h0, filling=FILLING, keys=[(0, 0)], integration=integration, tol=5e-4
    )
    mu_exact = np.sqrt(MASS**2 + FILLING - 1)
    polarization = 2 * MASS * (np.sqrt(MASS**2 + 1) - mu_exact)
    exact = np.diag([(FILLING - polarization) / 2, (FILLING + polarization) / 2])
    density_error = np.max(np.abs(bare.to_tb()[(0, 0)] - exact))
    print(
        f"Noninteracting density error against analytic disk average: {density_error:.3e}"
    )
    assert density_error < 2e-4

    model = mf.Model(h0, interaction, filling=FILLING)
    solution = mf.solver(
        model, model.mean_field(bare), integration=integration, tol=5e-4
    )
    print(f"SCF converged: {solution.converged}; mu: {solution.mu:.6f}")
    print(f"Internal energy per orbital: {solution.internal_energy:.8f}")
    print("Energy accuracy requires a separate convergence check.")


if __name__ == "__main__":
    main()
