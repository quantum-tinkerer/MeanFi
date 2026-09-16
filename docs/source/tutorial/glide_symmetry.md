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

# Symmetry-constrained mean fields

A symmetry constraint reduces the density entries used by SCF. Here we compare
unconstrained and glide-constrained solutions of the same two-orbital model.

## Define the glide and model

The glide reflects $y$ and exchanges the orbitals:
$g|R,A\rangle=|M_yR,B\rangle$ and
$g|R,B\rangle=|M_yR+\hat x,A\rangle$. Applying it twice translates by one cell.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from scipy.special import expit

U0 = np.array([[0, 0], [1, 0]], dtype=complex)
U1 = U0.T

glide = meanfi.SpatialSymmetry(
    lattice_matrix=np.diag([1, -1]),
    unitaries_by_shift={(0, 0): U0, (1, 0): U1},
)

identity = np.eye(2)
sigma_x = U0 + U1
sigma_z = np.diag([1, -1])
h_0 = {
    (0, 0): sigma_x,
    (1, 0): U1,
    (-1, 0): U0,
    (0, 1): -0.25 * identity + 0.45j * sigma_z,
    (0, -1): -0.25 * identity - 0.45j * sigma_z,
}

V = 3.0
h_int = {(0, 0): V * sigma_x, (1, 0): V * U1, (-1, 0): V * U0}
model_free = meanfi.Model(h_0, h_int, filling=1, kT=0.3)
model_glide = meanfi.Model(
    h_0, h_int, filling=1, kT=0.3, spatial_symmetries=(glide,)
)
```

The glide exchanges the two interacting bonds, so their strengths are equal.
Both the bare Hamiltonian and the interaction respect the symmetry.

## Solve with and without the constraint

Both calculations use the default EDIIS method and tolerances. At this coupling,
the unconstrained solution spontaneously breaks the glide.

```{code-cell} ipython3
free_result = meanfi.solver(model_free, model_free.random_meanfield(rng=10, scale=0.05))
glide_result = meanfi.solver(model_glide, model_glide.random_meanfield(rng=10, scale=0.05))

for label, model, result in [
    ("unconstrained", model_free, free_result),
    ("glide-constrained", model_glide, glide_result),
]:
    print(f"{label}: {model.required_coordinates.value_count} density entries, "
          f"residual {result.errors.scf_residual:.2e}, energy {result.internal_energy:.6f}")
```

## See the symmetry in the occupations

Let $p(k)=n_A(k)-n_B(k)$ be the sublattice occupation difference. A glide-symmetric
state satisfies $p(k_x,k_y)=-p(k_x,-k_y)$. Its even component

$$p_{\rm even}(k)=\tfrac12[p(k_x,k_y)+p(k_x,-k_y)]$$

therefore vanishes. We plot this quantity to locate the symmetry breaking in
momentum space. Unlike a Hamiltonian norm, it resolves how an onsite imbalance
affects the occupations of dispersing bands.

```{code-cell} ipython3
axis = np.linspace(-np.pi, np.pi, 81)
kx, ky = np.meshgrid(axis, axis)
points = np.stack([kx, ky], axis=-1)


def even_polarization(model, result):
    h = model.hamiltonian_from_meanfield(result.mean_field)
    energies, states = np.linalg.eigh(meanfi.tb_to_kfunc(h)(points))
    occupations = expit((result.mu - energies) / model.kT)
    weights = abs(states[..., 0, :])**2 - abs(states[..., 1, :])**2
    polarization = np.sum(weights * occupations, axis=-1)
    return (polarization + polarization[::-1]) / 2


free_even = even_polarization(model_free, free_result)
glide_even = even_polarization(model_glide, glide_result)
print(f"maximum constrained even component: {abs(glide_even).max():.2e}")
assert abs(glide_even).max() < 1e-12
assert abs(free_even).max() > 1e-2

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
limit = abs(free_even).max()
for ax, values, title in zip(
    axes, [free_even, glide_even], ["Unconstrained", "Glide-constrained"]
):
    image = ax.imshow(values, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                      cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(title=title, xlabel=r"$k_x$", ylabel=r"$k_y$")
fig.colorbar(image, ax=axes, label=r"$p_{\rm even}(k)$")
plt.show()
```

The constraint removes the symmetry-breaking density variables from SCF.
At weaker coupling, both calculations can reach the same symmetric state.
