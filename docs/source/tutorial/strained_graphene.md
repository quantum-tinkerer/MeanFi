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

# Strained graphene superlattice

This example combines a large sparse Kwant model with rational matrix functions.
We use a $16\times16$ graphene supercell and a fixed $2\times2$ momentum grid for a
quick qualitative calculation inspired by
[Manesco and Lado](https://doi.org/10.1088/2053-1583/ac0b48).
The lattice construction and band plotting are in
[`strained_graphene_kwant.py`](./scripts/strained_graphene_kwant.py).

## Build the sparse model

```{code-cell} ipython3
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from meanfi.interop import kwant as utils
from scripts.strained_graphene_kwant import create_system, plot_bands

sigma_z = np.diag([1, -1])
bare_builder, lattice, k_path = create_system(n=16)
h_0, geometry = utils.builder_to_tb(
    bare_builder, params={"xi": 7}, sparse=True, return_data=True
)


def onsite_interaction(site, U):
    return U * np.ones((2, 2))


interaction_builder = utils.build_interacting_syst(
    bare_builder, lattice, onsite_interaction
)
h_int = utils.builder_to_tb(interaction_builder, params={"U": 0.8}, sparse=True)
filling = h_0[(0, 0)].shape[0] // 2
model = meanfi.Model(h_0, h_int, filling=filling, kT=0.01)
integration = meanfi.UniformGrid(nk=2**2, matrix_function=meanfi.RationalFOE())
```

`nk` counts total points: `2**2` means two per axis. A prescribed grid does not
estimate integration error. Sparse rational evaluation needs the optional sparse
solver dependencies. We use default EDIIS with `tol=1e-2` for a quick qualitative
calculation; tighter tolerances can noticeably change the magnetization and gap.

## Noninteracting bands

At half filling, the bare model has chemical potential zero. Keep this band
structure for comparison with the interacting solution below.

```{code-cell} ipython3
plot_bands(meanfi.tb_to_kfunc(h_0), k_path)
```

## Solve from an antiferromagnetic guess

```{code-cell} ipython3
def staggered_field(site):
    return sigma_z if site.family == lattice.sublattices[0] else -sigma_z


guess_builder = utils.build_interacting_syst(
    bare_builder, lattice, staggered_field
)
guess = utils.builder_to_tb(guess_builder, sparse=True)
result = meanfi.solver(model, guess, integration=integration, tol=1e-2)
print(f"SCF residual: {result.errors.scf_residual:.2e}")
print(f"Chemical potential: {result.mu:.6f} t")
```

## Mean-field bands

The Hamiltonian is returned without shifting its energy origin. Pass the solved
chemical potential to the band plot.

```{code-cell} ipython3
h = model.hamiltonian_from_meanfield(result.mean_field)
plot_bands(meanfi.tb_to_kfunc(h), k_path, mu=result.mu)
```

## Local magnetization

The collinear guess selects the $z$ direction. We plot the local magnetization
$m_z=n_\uparrow-n_\downarrow$ and its magnitude $|m_z|$, with dot sizes
proportional to the magnitude. The onsite occupations are already in the SCF
result; no additional density calculation is needed.

```{code-cell} ipython3
occupations = np.array([
    result.density.values[result.density.coordinates.index((0, 0), i, i)].real
    for i in range(h_0[(0, 0)].shape[0])
]).reshape(-1, 2)
magnetization = dict(zip(geometry["sites"], occupations[:, 0] - occupations[:, 1]))
magnitude = {site: abs(value) for site, value in magnetization.items()}
peak = max(magnitude.values())
solution_builder = utils.tb_to_builder(
    result.mean_field, geometry["sites"], geometry["periods"]
)
```

The signed panel distinguishes the two spin orientations; the magnitude panel
highlights where the magnetic order is strongest.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for ax, values, title, cmap, minimum in zip(
    axes, [magnetization, magnitude], ["Magnetization", "Magnetization magnitude"],
    ["coolwarm", "viridis"], [-peak, 0]
):
    kwant.plot(
        solution_builder, site_color=values.__getitem__,
        site_size=lambda site: 0.3 * magnitude[site] / peak if peak else 0,
        unit=1, cmap=cmap, ax=ax, show=False,
    )
    ax.collections[0].set_clim(minimum, peak)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5)
plt.show()
```

This small fixed grid illustrates the sparse workflow. Establish convergence
with respect to cell size and momentum sampling before using it quantitatively.
