---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Chiral $p$-wave superconductivity

In the previous tutorials, we focused on normal-state mean-field problems.
Here we turn on the superconducting mode and solve for a chiral $p_x + i p_y$ state on a square lattice.
The top-level workflow is the same as before: define a tight-binding dictionary, define an interaction dictionary, choose a guess, and call {autolink}`~meanfi.solver`.
The main difference is that we now ask MeanFi to search for anomalous pairing terms as well as the normal mean field.

## Model definition

We consider a spinless square lattice with nearest-neighbor hopping and nearest-neighbor attraction.
The key API change is `superconducting=True` in {py:class}`meanfi.model.Model`.
That tells MeanFi to lift the problem to electron-first BdG space and solve for anomalous pairing together with the normal correction.
Even though the normal-state Hamiltonian has only one orbital per unit cell, every mean-field correction matrix is now `2 x 2`.
This is also what makes the calculation heavier than the normal-state tutorials.
In the BdG problem, the charge operator no longer commutes with the Hamiltonian, so the fixed-filling solve has to repeatedly rediagonalize while adjusting the chemical potential.

We also keep a small but nonzero temperature `kT`.
This smooths the occupation function, regularizes the fixed-filling BdG solve, and makes the calculation more reliable than the strictly zero-temperature problem.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import meanfi
import numpy as np

hopping = 0.5
coupling = 1.0
filling = 0.5
kT = 0.16

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

model = meanfi.Model(
    h_0,
    h_int,
    filling=filling,
    kT=kT,
    superconducting=True,
)
```

## A symmetry-informed guess

To target a chiral $p_x + i p_y$ state, we can assemble the guess directly.
The $x$ bonds get a real odd-parity pairing amplitude and the $y$ bonds get an imaginary odd-parity pairing amplitude.

```{code-cell} ipython3
guess = {
    (1, 0): np.array([[0.0, 0.15], [-0.15, 0.0]], dtype=complex),
    (-1, 0): np.array([[0.0, -0.15], [0.15, 0.0]], dtype=complex),
    (0, 1): np.array([[0.0, 1j * 0.15], [1j * 0.15, 0.0]], dtype=complex),
    (0, -1): np.array([[0.0, -1j * 0.15], [-1j * 0.15, 0.0]], dtype=complex),
}

result = meanfi.solver(
    model,
    guess,
)

delta_x = result.mf[(1, 0)][0, 1]
delta_y = result.mf[(0, 1)][0, 1]

print(f"residual norm: {result.info.residual_norm:.3e}")
print(f"chemical potential: {result.density_matrix_result.mu:.6f}")
print(f"Delta_x = {delta_x:.6f}")
print(f"Delta_y = {delta_y:.6f}")
```

The converged solution has a real $\Delta_x$ and an imaginary $\Delta_y$, which is the structure we wanted.

```{code-cell} ipython3
:tags: [hide-input]

def gap_texture(result, *, nk=81):
    axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    phase_grid = np.stack([kx, ky], axis=-1)

    electron_tb = {key: np.array(value, dtype=complex) for key, value in h_0.items()}
    gap = np.zeros_like(kx, dtype=complex)
    for key, matrix in result.mf.items():
        array = np.asarray(matrix, dtype=complex)
        electron_tb[key] = electron_tb.get(key, np.zeros((1, 1), dtype=complex)) + array[:1, :1]
        phase = np.exp(
            -1j
            * np.tensordot(phase_grid, np.asarray(key, dtype=float), axes=([-1], [0]))
        )
        gap += array[0, 1] * phase

    dispersion = np.zeros_like(kx, dtype=complex)
    for key, matrix in electron_tb.items():
        phase = np.exp(
            -1j
            * np.tensordot(phase_grid, np.asarray(key, dtype=float), axes=([-1], [0]))
        )
        dispersion += matrix[0, 0] * phase

    xi = dispersion.real - result.density_matrix_result.mu
    return axis, xi, gap


def phase_winding(result):
    axis, xi, gap = gap_texture(result, nk=301)
    figure, axes = plt.subplots()
    contour = axes.contour(axis, axis, xi.T, levels=[0.0])
    plt.close(figure)

    windings = []
    for path in contour.get_paths():
        vertices = path.vertices
        if len(vertices) < 20:
            continue
        sample = np.empty(len(vertices), dtype=complex)
        for index, (kx, ky) in enumerate(vertices):
            ix = int(np.argmin(np.abs(axis - kx)))
            iy = int(np.argmin(np.abs(axis - ky)))
            sample[index] = gap[ix, iy]
        phase = np.unwrap(np.angle(sample))
        windings.append(float((phase[-1] - phase[0]) / (2.0 * np.pi)))

    if not windings:
        raise ValueError("Could not resolve a Fermi-surface contour for the winding calculation")
    return float(max(windings, key=abs))


axis, xi, gap = gap_texture(result)
gap_abs = np.abs(gap)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
extent = (-np.pi, np.pi, -np.pi, np.pi)

mag = axes[0].imshow(
    gap_abs.T,
    origin="lower",
    extent=extent,
    cmap="viridis",
    interpolation="bicubic",
)
axes[0].contour(axis, axis, xi.T, levels=[0.0], colors="white", linewidths=1.0)
axes[0].set_title(r"$|\Delta(\mathbf{k})|$")
axes[0].set_xlabel(r"$k_x$")
axes[0].set_ylabel(r"$k_y$")
fig.colorbar(mag, ax=axes[0], shrink=0.8)

phase = axes[1].imshow(
    np.angle(gap).T,
    origin="lower",
    extent=extent,
    cmap="twilight_shifted",
    interpolation="bicubic",
    vmin=-np.pi,
    vmax=np.pi,
    alpha=np.clip(gap_abs.T / np.max(gap_abs), 0.15, 1.0),
)
axes[1].contour(axis, axis, xi.T, levels=[0.0], colors="white", linewidths=1.0)
axes[1].set_title(r"$\arg\,\Delta(\mathbf{k})$")
axes[1].set_xlabel(r"$k_x$")
axes[1].set_ylabel(r"$k_y$")
fig.colorbar(phase, ax=axes[1], shrink=0.8)
plt.show()

print(f"phase winding: {phase_winding(result):+.1f}")
```

The phase winds once around the Fermi surface, which is the characteristic chiral $p_x + i p_y$ texture.

## Why guesses matter

In superconducting calculations, the initial guess matters even more than in the normal-state examples because several self-consistent pairing patterns can coexist.
If we instead start from a random BdG guess, the solver can converge to a different superconducting state.

```{code-cell} ipython3
np.random.seed(11)
random_guess = meanfi.guess_tb(
    frozenset(h_int),
    ndof=1,
    scale=0.5,
    superconducting=True,
)

random_result = meanfi.solver(
    model,
    random_guess,
)

random_delta_x = random_result.mf[(1, 0)][0, 1]
random_delta_y = random_result.mf[(0, 1)][0, 1]

print(f"Delta_x (random) = {random_delta_x:.6f}")
print(f"Delta_y (random) = {random_delta_y:.6f}")
print(f"random residual norm: {random_result.info.residual_norm:.3e}")
```

With this seed, the random initial condition converges to a more nematic superconducting solution.
It still develops anomalous order, but it does not retain the clean $C_4$-symmetric chiral structure obtained from the symmetry-informed guess.

```{code-cell} ipython3
:tags: [hide-input]

random_axis, random_xi, random_gap = gap_texture(random_result)
random_gap_abs = np.abs(random_gap)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

mag = axes[0].imshow(
    random_gap_abs.T,
    origin="lower",
    extent=extent,
    cmap="viridis",
    interpolation="bicubic",
)
axes[0].contour(
    random_axis,
    random_axis,
    random_xi.T,
    levels=[0.0],
    colors="white",
    linewidths=1.0,
)
axes[0].set_title(r"$|\Delta(\mathbf{k})|$ from random guess")
axes[0].set_xlabel(r"$k_x$")
axes[0].set_ylabel(r"$k_y$")
fig.colorbar(mag, ax=axes[0], shrink=0.8)

phase = axes[1].imshow(
    np.angle(random_gap).T,
    origin="lower",
    extent=extent,
    cmap="twilight_shifted",
    interpolation="bicubic",
    vmin=-np.pi,
    vmax=np.pi,
    alpha=np.clip(random_gap_abs.T / np.max(random_gap_abs), 0.15, 1.0),
)
axes[1].contour(
    random_axis,
    random_axis,
    random_xi.T,
    levels=[0.0],
    colors="white",
    linewidths=1.0,
)
axes[1].set_title(r"$\arg\,\Delta(\mathbf{k})$ from random guess")
axes[1].set_xlabel(r"$k_x$")
axes[1].set_ylabel(r"$k_y$")
fig.colorbar(phase, ax=axes[1], shrink=0.8)
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

def mean_field_energy_density(result):
    return meanfi.total_energy(model, result.density_matrix_result.density_matrix)


print(f"chiral energy density: {mean_field_energy_density(result):.6f}")
print(f"random energy density: {mean_field_energy_density(random_result):.6f}")
print(f"chiral winding: {phase_winding(result):+.1f}")
print(f"random winding: {phase_winding(random_result):+.1f}")
```

For this parameter set, the symmetry-informed $p_x + i p_y$ solution has the lower mean-field energy density.
This is why guesses are not just a convenience in superconducting calculations: they can decide which self-consistent state you reach.
When several pairing patterns are possible, it is worth comparing the energies of competing converged solutions rather than trusting the first one that appears.
