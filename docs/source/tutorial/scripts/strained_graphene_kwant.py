"""Strained honeycomb supercell and a Gamma-K-M-K'-Gamma band plot."""

import kwant
import numpy as np


def create_system(n=16, nk=15):
    """Build the modulation of Manesco and Lado, 2D Materials 8, 035057 (2021)."""
    lattice = kwant.lattice.honeycomb(a=1, norbs=2)
    symmetry = kwant.TranslationalSymmetry(lattice.vec((n, 0)), lattice.vec((0, n)))
    bulk = kwant.Builder(symmetry)
    periods = np.asarray(symmetry.periods)
    reciprocal = np.linalg.inv(periods).T
    wavevectors = (
        2 * np.pi * np.array([reciprocal[0], reciprocal[1], -reciprocal.sum(axis=0)])
    )
    period = np.linalg.norm(periods[0])

    def hopping(site1, site2, xi):
        midpoint = (site1.pos + site2.pos) / 2
        displacement = site1.pos - site2.pos
        cross = (
            displacement[0] * wavevectors[:, 1] - displacement[1] * wavevectors[:, 0]
        )
        wavevector = wavevectors[np.argmin(abs(cross))]
        modulation = np.sin(wavevector @ midpoint) / np.linalg.norm(wavevectors[0])
        strength = (3 / 2) * xi**2 / (period**2 * np.sqrt(3))
        return (1 + strength * modulation) * np.eye(2)

    bulk[lattice.shape(lambda pos: True, (0, 0))] = np.zeros((2, 2))
    bulk[lattice.neighbors()] = hopping

    # Momentum coordinates are phases along the two supercell translations.
    gamma = np.zeros(2)
    corner = 2 * np.pi * np.array([2 / 3, 1 / 3])
    next_corner = corner[::-1]
    path = np.concatenate(
        [
            np.linspace(gamma, corner, nk, endpoint=False),
            np.linspace(corner, next_corner, nk, endpoint=False),
            np.linspace(next_corner, gamma, nk + 1),
        ]
    )
    return bulk, lattice, path


def plot_bands(h_of_k, k_path, *, mu=0.0):
    """Plot dense band eigenvalues relative to the specified chemical potential."""
    import matplotlib.pyplot as plt

    energies = np.array([np.linalg.eigvalsh(h_of_k(k)) for k in k_path]) - mu
    intervals = len(k_path) - 1
    plt.figure(figsize=(6, 4))
    plt.plot(energies, color="black", linewidth=1)
    ticks = np.array([0, 1 / 3, 1 / 2, 2 / 3, 1]) * intervals
    plt.xticks(ticks, [r"$\Gamma$", "$K$", "$M$", "$K'$", r"$\Gamma$"])
    for tick in ticks[1:-1]:
        plt.axvline(tick, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlim(0, intervals)
    plt.ylim(-0.1, 0.1)
    plt.ylabel(r"$(E-\mu)/t$")
    plt.tight_layout()
    plt.show()
