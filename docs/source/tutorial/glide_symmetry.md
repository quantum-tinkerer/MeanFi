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

# Symmetry-reduced mean-field variables

`meanfi` can reduce the active SCF variables using any linear symmetry that can be written as a real-space action on the tight-binding basis: point symmetries, spin/orbital symmetries, and shifted spatial symmetries.
This tutorial uses a glide because it is the smallest example where the shifted representation is essential.

The SCF space is built in three steps:

1. `h_int` selects the density entries that can affect the mean-field Hamiltonian.
2. Hermiticity and user symmetries impose linear constraints on those entries.
3. `ActiveSCFSpace` chooses only the real-space density entries needed to recover the reduced SCF parameters.

There is no BdG doubling or anomalous density in this example.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import warnings

import meanfi
from meanfi.space.reducers import LinearConstraintReducer, OrbitReducer
from meanfi.space.support import normal_active_support
from meanfi.space.symmetry import HermiticityConstraint
```

```{code-cell} ipython3
:tags: [hide-input]

warnings.filterwarnings("ignore", message="Normal SCF guess contains values")
```

## A shifted glide symmetry

We use two orbitals on a 2D lattice and the glide

:::{math}
g = \{M_y \mid \hat x/2\},
\qquad
g^2 = T_{\hat x}.
:::

In the tight-binding basis this is encoded as

:::{math}
g|R,a\rangle
=
\sum_{s,b}U_s[b,a]|AR+s,b\rangle,
\qquad
A =
\begin{pmatrix}
1 & 0\\
0 & -1
\end{pmatrix}.
:::

The two shifted matrices below mean
`g |R,A> = |AR,B>` and `g |R,B> = |AR + x,A>`.

```{code-cell} ipython3
U0 = np.array([[0, 0], [1, 0]], dtype=complex)
U1 = np.array([[0, 1], [0, 0]], dtype=complex)

glide = meanfi.SpatialSymmetry(
    lattice_matrix=np.array([[1, 0], [0, -1]]),
    unitaries_by_shift={(0, 0): U0, (1, 0): U1},
)
```

## Model and interaction

The interaction is sparse on purpose.
Its nonzero structure defines the active density entries used by the SCF loop.

```{code-cell} ipython3
eye = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.diag([1, -1]).astype(complex)

h_0 = {
    (0, 0): sigma_x,
    (1, 0): np.array([[0, 1], [0, 0]], dtype=complex),
    (-1, 0): np.array([[0, 0], [1, 0]], dtype=complex),
    (0, 1): -0.25 * eye + 0.45j * sigma_z,
    (0, -1): -0.25 * eye - 0.45j * sigma_z,
}


def interaction_block(entries, strength):
    block = np.zeros((2, 2), dtype=complex)
    for row, col in entries:
        block[row, col] = strength
    return block


h_int = {
    (0, 0): interaction_block([(0, 0), (1, 1)], 1.8),
    (1, 0): interaction_block([(0, 1)], 1.0),
    (-1, 0): interaction_block([(1, 0)], 1.0),
    (2, 0): interaction_block([(1, 0)], 0.6),
    (-2, 0): interaction_block([(0, 1)], 0.6),
    (0, 1): interaction_block([(0, 1)], 0.7),
    (0, -1): interaction_block([(1, 0)], 0.7),
    (1, -1): interaction_block([(1, 0)], 0.5),
    (-1, 1): interaction_block([(0, 1)], 0.5),
}

model_free = meanfi.Model(h_0, h_int, filling=1.0, kT=0.3)
model_glide = meanfi.Model(
    h_0,
    h_int,
    filling=1.0,
    kT=0.3,
    spatial_symmetries=(glide,),
)
```

## What the symmetry removes

The raw active entries come from `h_int`.
Hermiticity removes conjugate redundancy; the glide then removes additional SCF variables.

```{code-cell} ipython3
support = normal_active_support(model_glide)
entries = support.coordinates.entries

hermitian_basis = OrbitReducer(entries).basis((HermiticityConstraint(),))
glide_basis = LinearConstraintReducer(
    entries,
    ndof=model_glide._ndof,
    family="normal",
).basis(hermitian_basis, (glide,))

print(f"raw active real variables:      {2 * len(entries):2d}")
print(f"after Hermiticity:              {hermitian_basis.shape[1]:2d}")
print(f"after glide symmetry:           {glide_basis.shape[1]:2d}")
print(f"required real-space entries:    {len(model_glide.scf_space.required_realspace_entries()):2d}")
```

```{code-cell} ipython3
model_glide.scf_space.required_realspace_entries()
```

The backend does not need all active density entries.
It computes the required entries above, compresses them to reduced SCF parameters, and reconstructs the constrained active mean-field input when needed.

## Converged SCF comparison

Now solve the same interacting problem twice: once with no symmetry constraint and once with the glide constraint.

```{code-cell} ipython3
integration = meanfi.UniformGrid(nk=9)
scf = meanfi.AndersonMixing(M=3, max_iterations=80)

free_result = meanfi.solver(
    model_free,
    model_free.random_meanfield(rng=10, scale=0.05),
    integration=integration,
    scf=scf,
    scf_tol=1e-6,
)
glide_result = meanfi.solver(
    model_glide,
    model_glide.random_meanfield(rng=10, scale=0.05),
    integration=integration,
    scf=scf,
    scf_tol=1e-6,
)

print(f"unconstrained SCF variables: {model_free.scf_space.num_params}")
print(f"glide-constrained variables: {model_glide.scf_space.num_params}")
print(f"unconstrained residual:      {free_result.info.residual_norm:.2e}")
print(f"glide residual:              {glide_result.info.residual_norm:.2e}")
```

The plot below measures the glide mismatch of the converged mean-field Hamiltonian:

:::{math}
\|H(k_x,k_y) - U_g(k_x) H(k_x,-k_y) U_g(k_x)^\dagger\|_F,
\qquad
U_g(k_x)=U_0+e^{-ik_x}U_1.
:::

The unconstrained solution is allowed to break the glide.
The constrained solution stays in the glide-preserving SCF subspace.

```{code-cell} ipython3
:tags: [hide-input]

def glide_matrix(kx):
    return U0 + np.exp(-1j * kx) * U1


def glide_mismatch(model, meanfield):
    h_of_k = meanfi.tb_to_kfunc(model.hamiltonian_from_meanfield(meanfield))
    grid = np.linspace(-np.pi, np.pi, 81)
    mismatch = np.empty((grid.size, grid.size))
    for y_index, ky in enumerate(grid):
        for x_index, kx in enumerate(grid):
            h_here = h_of_k(np.array([[kx, ky]]))[0]
            h_reflected = h_of_k(np.array([[kx, -ky]]))[0]
            unitary = glide_matrix(kx)
            mismatch[y_index, x_index] = np.linalg.norm(
                h_here - unitary @ h_reflected @ unitary.conj().T
            )
    return grid, mismatch


grid, free_mismatch = glide_mismatch(model_free, free_result.mf)
_, glide_mismatch_values = glide_mismatch(model_glide, glide_result.mf)
vmax = float(free_mismatch.max())

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
for ax, values, title in [
    (axes[0], free_mismatch, "unconstrained SCF"),
    (axes[1], glide_mismatch_values, "glide-constrained SCF"),
]:
    image = ax.imshow(
        values,
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )
    ax.axhline(0.0, color="white", lw=1.0, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(r"$k_x$")
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
axes[0].set_ylabel(r"$k_y$")
fig.colorbar(image, ax=axes, label="glide mismatch")
plt.show()
```

The white line is the glide-invariant line.
The important part is not that the symmetry changes a band plot; it changes which mean-field density variables the SCF loop is allowed to use.
