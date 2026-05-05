# %%
from __future__ import annotations


import numpy as np

# %%
from matplotlib import pyplot as plt

from meanfi import (
    AdaptiveQuadrature,
    LinearMixing,
    Model,
    solver,
    guess_tb,
)

# %%


def chiral_square_problem():
    hopping = 0.5
    coupling = 1
    filling = 0.5
    kT = 0.16
    guess_scale = 0.15

    h_0 = {
        (0, 0): np.zeros((1, 1), dtype=complex),
        (1, 0): -hopping * np.eye(1, dtype=complex),
        (-1, 0): -hopping * np.eye(1, dtype=complex),
        (0, 1): -hopping * np.eye(1, dtype=complex),
        (0, -1): -hopping * np.eye(1, dtype=complex),
    }
    h_int = {
        (1, 0): coupling * np.ones((1, 1), dtype=complex),
        (-1, 0): coupling * np.ones((1, 1), dtype=complex),
        (0, 1): coupling * np.ones((1, 1), dtype=complex),
        (0, -1): coupling * np.ones((1, 1), dtype=complex),
    }
    guess = {
        (1, 0): np.array([[0.0, guess_scale], [-guess_scale, 0.0]], dtype=complex),
        (-1, 0): np.array([[0.0, -guess_scale], [guess_scale, 0.0]], dtype=complex),
        (0, 1): np.array(
            [[0.0, 1j * guess_scale], [1j * guess_scale, 0.0]], dtype=complex
        ),
        (0, -1): np.array(
            [[0.0, -1j * guess_scale], [-1j * guess_scale, 0.0]], dtype=complex
        ),
    }
    model = Model(
        h_0,
        h_int,
        filling=filling,
        kT=kT,
        superconducting=True,
    )
    integration = AdaptiveQuadrature()
    scf = LinearMixing(max_iterations=220, alpha=0.2)
    return model, guess, integration, scf, h_int


def solve_chiral_square_state():
    model, guess, integration, scf, h_int = chiral_square_problem()
    guess_random = guess_tb(frozenset(h_int), ndof=1, scale=0.5, superconducting=True)
    np.random.seed(11)
    guess_random = guess_tb(
        frozenset(h_int),
        ndof=1,
        scale=0.5,
        superconducting=True,
    )

    print(guess_random)
    result = solver(
        model,
        guess_random,
        # integration=integration,
        # scf=scf,
        # scf_tol=1e-3,
        # filling_tol=1e-3,
    )
    return model, result


# %%
model, result = solve_chiral_square_state()
# %%


def _electron_and_pairing_symbols(model: Model, result, *, nk: int):
    axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    phase_grid = np.stack([kx, ky], axis=-1)

    electron_tb = {
        key: np.array(value, dtype=complex) for key, value in model.h_0.items()
    }
    pairing_tb = {}
    for key, matrix in result.mf.items():
        array = np.asarray(matrix, dtype=complex)
        electron_tb[key] = (
            electron_tb.get(key, np.zeros((1, 1), dtype=complex)) + array[:1, :1]
        )
        pairing_tb[key] = array[:1, 1:]

    dispersion = np.zeros_like(kx, dtype=complex)
    gap = np.zeros_like(kx, dtype=complex)
    for key, matrix in electron_tb.items():
        phase = np.exp(
            -1j
            * np.tensordot(phase_grid, np.asarray(key, dtype=float), axes=([-1], [0]))
        )
        dispersion += matrix[0, 0] * phase
    for key, matrix in pairing_tb.items():
        phase = np.exp(
            -1j
            * np.tensordot(phase_grid, np.asarray(key, dtype=float), axes=([-1], [0]))
        )
        gap += matrix[0, 0] * phase

    xi = dispersion.real - result.density_matrix_result.mu
    gap_abs = np.abs(gap)
    quasiparticle = np.sqrt(xi**2 + gap_abs**2)
    return axis, xi, gap, gap_abs, quasiparticle


def _phase_winding(
    gap_function: np.ndarray, radius: float = 0.45, n_points: int = 361
) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    kx = radius * np.cos(theta)
    ky = radius * np.sin(theta)
    axis = np.linspace(-np.pi, np.pi, gap_function.shape[0], endpoint=False)

    sample = np.empty(n_points, dtype=complex)
    for index, (x_val, y_val) in enumerate(zip(kx, ky, strict=False)):
        ix = int(np.argmin(np.abs(axis - x_val)))
        iy = int(np.argmin(np.abs(axis - y_val)))
        sample[index] = gap_function[ix, iy]

    phase = np.unwrap(np.angle(sample))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


nk = 31
# model, result = solve_chiral_square_state()
axis, xi, gap, gap_abs, quasiparticle = _electron_and_pairing_symbols(
    model,
    result,
    nk=nk,
)
winding = _phase_winding(gap)

delta_x = result.mf[(1, 0)][0, 1]
delta_y = result.mf[(0, 1)][0, 1]

with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    extent = (-np.pi, np.pi, -np.pi, np.pi)

    disp = axes[0, 0].imshow(
        xi.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        interpolation="bicubic",
    )
    axes[0, 0].contour(axis, axis, xi.T, levels=[0.0], colors="white", linewidths=1.2)
    axes[0, 0].set_title("Normal Dispersion $\\xi(\\mathbf{k})$")
    axes[0, 0].set_xlabel("$k_x$")
    axes[0, 0].set_ylabel("$k_y$")
    fig.colorbar(disp, ax=axes[0, 0], fraction=0.046, pad=0.04)

    qp = axes[0, 1].imshow(
        quasiparticle.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="bicubic",
        vmin=0,
    )

    axes[0, 1].contour(
        axis, axis, xi.T, levels=[0.0], colors="cyan", linewidths=1.0, alpha=0.9
    )
    axes[0, 1].set_title("BdG Quasiparticle Energy $E(\\mathbf{k})$")
    axes[0, 1].set_xlabel("$k_x$")
    axes[0, 1].set_ylabel("$k_y$")
    fig.colorbar(qp, ax=axes[0, 1], fraction=0.046, pad=0.04)

    mag = axes[1, 0].imshow(
        gap_abs.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="bicubic",
        vmin=0,
    )
    axes[1, 0].contour(
        axis, axis, xi.T, levels=[0.0], colors="white", linewidths=1.0, alpha=0.9
    )
    axes[1, 0].set_title("Gap Magnitude $|\\Delta(\\mathbf{k})|$")
    axes[1, 0].set_xlabel("$k_x$")
    axes[1, 0].set_ylabel("$k_y$")
    fig.colorbar(mag, ax=axes[1, 0], fraction=0.046, pad=0.04)

    phase = np.angle(gap)
    phase_image = axes[1, 1].imshow(
        phase.T,
        origin="lower",
        extent=extent,
        cmap="twilight_shifted",
        interpolation="bicubic",
        vmin=-np.pi,
        vmax=np.pi,
        alpha=np.clip(gap_abs.T / np.max(gap_abs), 0.15, 1.0),
    )
    axes[1, 1].contour(
        axis, axis, xi.T, levels=[0.0], colors="white", linewidths=1.0, alpha=0.9
    )
    axes[1, 1].set_title("Phase Texture $\\arg\\,\\Delta(\\mathbf{k})$")
    axes[1, 1].set_xlabel("$k_x$")
    axes[1, 1].set_ylabel("$k_y$")
    fig.colorbar(
        phase_image,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
        ticks=[-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
    )

    fig.suptitle(
        "Spinless Square-Lattice Chiral $p_x + i p_y$ Mean-Field State",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        (
            f"$\\mu={result.density_matrix_result.mu:.3f}$, "
            f"$\\Delta_x={delta_x.real:.4f}$, "
            f"$\\Delta_y={delta_y.imag:.4f}i$, "
            f"winding $\\approx {winding:+.2f}$, "
            f"SCF residual $={result.info.residual_norm:.2e}$"
        ),
        ha="center",
        fontsize=11,
    )
plt.show()
# %%
