"""Plot full band spectra without assuming a spin or valley ordering pattern."""

import numpy as np

from .rhombohedral_graphene import LOCAL


def plot_bands(solutions):
    """Color the bands nearest half filling by outer-surface polarization."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(solutions),
        figsize=(3.2 * len(solutions), 3.8),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    qx = np.linspace(-0.12, 0.12, 301)
    for ax, (parameters, result) in zip(axes[0], solutions, strict=True):
        correction = result.mean_field[LOCAL]
        h0 = parameters.affine_hamiltonian()
        sites = 2 * parameters.layers
        surface = np.diag([1.0] + [0.0] * (sites - 2) + [-1.0])
        observable = np.kron(surface, np.eye(4))
        occupied = 4 * parameters.layers
        bands = slice(occupied - 4, occupied + 4)
        energies, colors = [], []
        for x in qx:
            e, v = np.linalg.eigh(h0(x, 0.0) + correction)
            energies.append((e[bands] - result.mu) * 1000)
            colors.append(
                np.einsum(
                    "ib,ij,jb->b", v[:, bands].conj(), observable, v[:, bands]
                ).real
            )
        artist = ax.scatter(
            np.repeat(qx, 8),
            np.array(energies).ravel(),
            c=np.array(colors).ravel(),
            s=2,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            rasterized=True,
        )
        ax.axhline(0, color=".7", lw=0.6)
        ax.set(
            xlim=(-0.12, 0.12),
            ylim=(-40, 40),
            xticks=[-0.1, 0, 0.1],
            xlabel=r"$a_0 k_x$",
            title=rf"$\Delta={parameters.delta * 1000:g}$ meV, $J_H={parameters.hund:g}$",
        )
    axes[0, 0].set_ylabel("Energy relative to μ (meV)")
    fig.colorbar(
        artist, ax=axes.ravel().tolist(), shrink=0.8, label="Outer-surface polarization"
    )
    return fig
