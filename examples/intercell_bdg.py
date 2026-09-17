"""Spinless intercell pairing with a callable Hamiltonian and UniformGrid."""

from dataclasses import replace

import numpy as np

import meanfi as mf


def main():
    def h(kx):
        return np.array([[-2 * np.cos(kx)]])

    interaction = mf.BilinearInteraction(
        [
            mf.BilinearTerm(-1.5, np.eye(1), np.eye(1), displacement=(1,)),
        ]
    )
    model = mf.Model(
        mf.BlochHamiltonian(h), interaction, filling=0.5, kT=0.08, superconducting=True
    )
    # Electron-first Nambu blocks, with Delta(-R) = -Delta(R).T.
    guess = {
        (1,): np.array([[0, 0.2], [-0.2, 0]]),
        (-1,): np.array([[0, -0.2], [0.2, 0]]),
    }
    tolerance = replace(
        mf.default_solver_tolerances(1e-7), filling_residual=1e-12, mu_tol=1e-12
    )
    result = mf.solver(
        model,
        guess,
        integration=mf.UniformGrid(nk=128),
        tol=tolerance,
        compute_free_energy=True,
    )
    finer = mf.density_matrix(
        model,
        mean_field=result.mean_field,
        integration=mf.UniformGrid(nk=256),
        tol=tolerance,
    )
    energy_change = abs(result.internal_energy - finer.internal_energy)
    print(f"SCF converged: {result.converged}")
    print(
        f"Nearest-neighbor pairing amplitude: {abs(result.mean_field[(1,)][0, 1]):.8f}"
    )
    print(f"Internal energy per orbital: {result.internal_energy:.10f}")
    print(f"Energy change on doubling the final grid: {energy_change:.3e}")
    assert abs(result.mean_field[(1,)][0, 1]) > 0.05
    assert energy_change < 1e-8


if __name__ == "__main__":
    main()
