---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Chiral $p$-wave superconductivity

Set `superconducting=True` to solve for pairing alongside the normal mean field.
This spinless square-lattice example uses attractive nearest-neighbor interactions
(`h_int < 0`) and starts from a chiral $p_x+i p_y$ guess.

## Define the model

```{code-cell} ipython3
import matplotlib.pyplot as plt
import meanfi
import numpy as np

bonds = [(1, 0), (-1, 0), (0, 1), (0, -1)]
h_0 = {(0, 0): np.zeros((1, 1)), **{key: -0.5 * np.eye(1) for key in bonds}}
h_int = {key: -np.ones((1, 1)) for key in bonds}
model = meanfi.Model(h_0, h_int, filling=0.5, kT=0.04, superconducting=True)
```

The physical model has one orbital. Its mean-field matrices have two Nambu
components, ordered electron then hole. Filling still counts physical electrons,
not the doubled matrix dimension. A nonzero temperature smooths the occupations.

## Choose a pairing guess and solve

The $x$-bond pairing is real and the $y$-bond pairing imaginary, with odd parity
under bond reversal. The normal correction is initially zero.

```{code-cell} ipython3
guess = {
    (1, 0): np.array([[0, 0.15], [-0.15, 0]]),
    (-1, 0): np.array([[0, -0.15], [0.15, 0]]),
    (0, 1): np.array([[0, 0.15j], [0.15j, 0]]),
    (0, -1): np.array([[0, -0.15j], [-0.15j, 0]]),
}
result = meanfi.solver(model, guess)
print(f"SCF residual: {result.errors.scf_residual:.2e}")
print(f"Delta_x: {result.mean_field[(1, 0)][0, 1]:.6f}")
print(f"Delta_y: {result.mean_field[(0, 1)][0, 1]:.6f}")
```

The solver uses the same default EDIIS and tolerances as the normal-state
examples. Different initial guesses can select different pairing states;
`model.random_meanfield(rng=11)` provides a general BdG guess.

## Inspect the gap texture

```{code-cell} ipython3
axis = np.linspace(-np.pi, np.pi, 101)
kx, ky = np.meshgrid(axis, axis)
h = model.hamiltonian_from_meanfield(result.mean_field)
h_k = meanfi.tb_to_kfunc(h)(np.stack([kx, ky], axis=-1))
normal_energy = h_k[..., 0, 0].real - result.mu
gap = h_k[..., 0, 1]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for ax, values, title, cmap in zip(
    axes, [abs(gap), np.angle(gap)], [r"$|\Delta(k)|$", r"$\arg\Delta(k)$"],
    ["viridis", "twilight_shifted"]
):
    image = ax.imshow(values, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi), cmap=cmap)
    ax.contour(axis, axis, normal_energy, levels=[0], colors="white", linewidths=1)
    ax.set(title=title, xlabel=r"$k_x$", ylabel=r"$k_y$")
    fig.colorbar(image, ax=ax)
plt.show()
```

The white contour marks the normal-state Fermi surface. The relative phase of
the two bond amplitudes produces the chiral gap texture.

EDIIS uses internal energy. To compare the free energies of competing states at
this nonzero temperature, request `compute_free_energy=True` when solving.
Then `result.free_energy = result.internal_energy - model.kT * result.entropy`;
all three quantities are per cell per physical orbital.
