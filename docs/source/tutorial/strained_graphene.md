---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Strained graphene superlattice

We showcase the interface between `meanfi` and `Kwant` with a strained graphene supercell. In this tutorial, we qualitatively reproduce the results from [[1]](https://doi.org/10.1088/2053-1583/ac0b48).

We first create the atomistic model in `Kwant`. The complete source code of this example can be found in [`strained_graphene_kwant.py`](./scripts/strained_graphene_kwant.py). To reduce the computational cost, we perform the calculations with a $16 \times 16$ supercell whereas in [1](https://doi.org/10.1088/2053-1583/ac0b48) the calculations were performed with a $25 \times 25$ supercell. Thus, the agreement throughout the tutorial is only qualitative.

```{code-cell} ipython3
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from meanfi.kwant_helper import utils
from scripts.pauli import s0, sx, sy, sz
from scripts.strained_graphene_kwant import create_system, plot_bands

sigmas = [sx, sy, sz]
```

We verify the band structure of the Kwant model along a high-symmetry k-path.

```{code-cell} ipython3
h0_builder, lat, k_path = create_system(n=16)
```

```{code-cell} ipython3
fsyst = kwant.wraparound.wraparound(h0_builder).finalized()
params = {"xi": 6}


def hk(k):
    return fsyst.hamiltonian_submatrix(
        params={**params, **dict(k_x=k[0], k_y=k[1])}, sparse=False
    )


plot_bands(hk, k_path)
```

We now use the Kwant model to create the interacting Hamiltonian. Following [[1]](https://doi.org/10.1088/2053-1583/ac0b48), we consider only onsite interactions.

```{code-cell} ipython3
def func_hop(site1, site2):
    return 0 * np.ones((2, 2))


def func_onsite(site, U):
    return U * np.ones((2, 2))


int_builder = utils.build_interacting_syst(
    h0_builder,
    lat,
    func_onsite,
    func_hop,
    max_neighbor=0,
)
```

After we have created the interacting system we can use MeanFi again for getting the solution. We turn both the non-interacting and interacting systems into tight binding dictionaries using the kwant utils. Then we combine them into a mean-field model.

```{code-cell} ipython3
from meanfi.kwant_helper import utils as utils

h0, data = utils.builder_to_tb(h0_builder, params={"xi": 6}, return_data=True)

params_int = dict(U=0.6)
ndof = [*h0.values()][0].shape[0]
filling = ndof // 2
h_int = utils.builder_to_tb(int_builder, params_int)
mf_model = meanfi.Model(h0, h_int, filling=filling)
```

Now getting the solution by providing a guess and the mean-field model to the solver. To accelerate the convergence, we use an antiferromagnetic guess.

```{code-cell} ipython3
def func_hop(site1, site2):
    return 0 * np.ones((2, 2))


def func_onsite(site):
    if site.family == lat.sublattices[0]:
        return sz
    else:
        return -sz


guess_builder = utils.build_interacting_syst(
    h0_builder,
    lat,
    func_onsite,
    func_hop,
    max_neighbor=0,
)

guess = utils.builder_to_tb(guess_builder)
```

Due to the large supercell, lwe use a coarse k-point grid. Furthermore, to speed up the calculation we provide additional `optimizer_kwargs`.

```{code-cell} ipython3
mf_sol = meanfi.solver(
    mf_model,
    guess,
    nk=2,
    optimizer_kwargs={
        "M": 10,
        "f_tol": 1e-4,
        "maxiter": 100,
        "line_search": "armijo",
    },
)
```

We now verify that the mean-field solution results in a gapped phase.

```{code-cell} ipython3
mf_ham = meanfi.add_tb(h0, mf_sol)
hk_mf = meanfi.tb_to_kfunc(mf_ham)
plot_bands(hk_mf, k_path)
```

Now we turn the mean-field corrections into a `kwant.Builder` such that we can visualize observables using Kwant's functionalities. We provide the mean-field solution `mf_sol` as well as the sites and periods of the bulk system to the `tb_to_builder` function.

```{code-cell} ipython3
mf_sol_builder = utils.tb_to_builder(mf_sol, data["sites"], data["periods"])
```

We now plot the magnetization of the system. First, we define the magnetization direction. We do this by arbitrarily choosing the spin direction of one of the sites and defining the magnetization with respect to this direction.

```{code-cell} ipython3
_, reference_value = list(mf_sol_builder.site_value_pairs())[0]

magnetization_direction = []
for sigma in sigmas:
    magnetization_direction.append(np.trace(sigma @ reference_value) * sigma)

reference_magnetization = sum(magnetization_direction)
reference_magnetization = reference_magnetization / np.linalg.norm(
    reference_magnetization
)
```

Now we plot the magnetization of the solution on the sites of the system. We do this by creating functions which calculate the magnetization and its magnitude with respect to the reference magnetization direction.

```{code-cell} ipython3
def magnetisation(site):
    matrix = mf_sol_builder.H[site][1]
    return np.trace(reference_magnetization @ matrix).real


def abs_magnetisation(site):
    matrix = mf_sol_builder.H[site][1]
    projected_magnetization = []
    for sigma in sigmas[1:]:
        projected_magnetization.append(np.trace(sigma @ matrix))
    return np.sqrt(np.sum(np.array(projected_magnetization) ** 2).real)


def systemPlotter(syst, onsite, ax, cmap):
    """
    Plots the system with the onsite potential given by the function onsite.
    """
    sites = [*syst.sites()]
    density = [onsite(site) for site in sites]

    def size(site):
        return 0.3 * np.abs(onsite(site)) / np.max(np.abs(density))

    kwant.plot(
        syst, site_color=onsite, ax=ax, cmap=cmap, site_size=size, show=False, unit=1
    )
    return np.min(density), np.max(density)
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(1, 2, figsize=(10, 5))


titles = ["Magnetisation", "Magnetisation magnitude"]
cmaps = ["coolwarm", "viridis"]
onsites = [magnetisation, abs_magnetisation]
for i, ax in enumerate(axs):
    vmin, vmax = systemPlotter(mf_sol_builder, onsites[i], ax=ax, cmap=cmaps[i])
    ax.axis("off")
    ax.set_title(titles[i])
    ax.set_aspect("equal")
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5)
fig.show()
```
