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

# Strained graphene

We showcase the interface between `meanfi` and `Kwant` with a strained graphene supercell. In this tutorial, we qualitatively reproduce the results from [1](https://doi.org/10.1088/2053-1583/ac0b48).

## Model creation:

We first create the atomistic model in `Kwant`. The complete source code of this example can be found in [`strained_graphene_kwant.py`](./scripts/strained_graphene_kwant.py). To reduce the computational cost, we perform the calculations with a $10 \times 10$ supercell whereas in [1](https://doi.org/10.1088/2053-1583/ac0b48) the calculations were performed with a $25 \times 25$ supercell. Thus, the agreement throughout the tutorial is only qualitative.

```{code-cell} ipython3
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from meanfi.kwant_helper import utils
import scipy
from scripts.strained_graphene_kwant import create_system
```

We verify the band structure of the Kwant model along a high-symmetry k-path.

```{code-cell} ipython3
syst, lat, k_path = create_system(10)
fsyst = kwant.wraparound.wraparound(syst).finalized()

eks = []
params = {"t": 1.0, "mu": 0.0, "delta_mu": 0.0, "xi": 6}
for k in k_path:
    ham_k = fsyst.hamiltonian_submatrix(
        params={**params, **dict(k_x=k[0], k_y=k[1])}, sparse=False
    )
    energies = np.sort(np.linalg.eigvalsh(ham_k))
    eks.append(energies)

nk = len(k_path)
plt.plot(eks, c="k", lw=1)
plt.ylabel(r"$E-E_F\ [eV]$")
plt.ylim(-0.5, 0.5)
plt.xlim(0, 149)
plt.xticks(
    [0, nk // 3, nk // 2, int(2 * nk // 3), nk],
    [r"$\Gamma$", r"$K$", r"$M$", r"$K^{\prime}$", r"$\Gamma$"],
)
plt.axvline(x=50, c="k", ls="--")
plt.axvline(x=75, c="k", ls="--")
plt.axvline(x=100, c="k", ls="--")
plt.axhline(y=0, c="k", ls="--")
plt.tight_layout()
plt.show()
```

We now use the Kwant model to create the interacting Hamiltonian. Following [[1]](https://doi.org/10.1088/2053-1583/ac0b48), we consider only onsite interactions.

```{code-cell} ipython3
def func_hop(site1, site2):
    return 0 * np.ones((2, 2))


def func_onsite(site, U):
    return U * np.ones((2, 2))


int_builder = utils.build_interacting_syst(
    syst,
    lat,
    func_onsite,
    func_hop,
    max_neighbor=0,
)
```

After we have created the interacting system we can use MeanFi again for getting the solution. We turn both the non-interacting and interacting systems into tight binding dictionaries using the kwant utils. Then we combine them into a mean-field model.

```{code-cell} ipython3
from meanfi.kwant_helper import utils as utils
h0 = utils.builder_to_tb(syst, params=params)

params_int = dict(U=2)
ndof = [*h0.values()][0].shape[0]
filling = ndof // 2
h_int = utils.builder_to_tb(int_builder, params_int)
mf_model = meanfi.Model(h0, h_int, filling=filling)
```

Now getting the solution by providing a guess and the mean-field model to the solver. As this strained graphene system is larger than the previous examples we use a smaller sampling of k-points. Furthermore, to speed up the calculation we played around with the `optimizer_kwargs`.

```{code-cell} ipython3
guess = meanfi.guess_tb(frozenset(h_int), ndof=ndof)
mf_sol = meanfi.solver(
    mf_model,
    guess,
    nk=4,
    optimizer_kwargs={
        "M": 10,
        "f_tol": 1e-3,
        "maxiter": 100,
        "verbose": True,
        "line_search": "wolfe",
    },
)
```

Let us now plot the bands of the mean-field solution along the same k-path where we visualized the bands earlier.

```{code-cell} ipython3
eks = []
full_sol = meanfi.add_tb(h0, mf_sol)
sol_ofk = tb_to_kfunc(full_sol)
for k in k_points:
    hk = sol_ofk(momentum_to_lattice(k, wrapped_syst))
    energies = np.linalg.eigvalsh(hk)
    eks.append(np.sort(energies))
```

```{code-cell} ipython3
:tags: [hide-input]

plt.plot(eks, c="k", lw=1)
plt.ylabel(r"$E-E_F\ [eV]$")
plt.ylim(-0.1, 0.1)
plt.xlim(0, 149)
plt.xticks(
    [0, 50, 75, 100, 150], [r"$\Gamma$", r"$K$", r"$M$", r"$K^{\prime}$", r"$\Gamma$"]
)
plt.axvline(x=50, c="k", ls="--")
plt.axvline(x=75, c="k", ls="--")
plt.axvline(x=100, c="k", ls="--")
plt.axhline(y=0, c="k", ls="--")
plt.tight_layout()
plt.show()
```

Now we turn our tight-binding mean-field solution into a kwant builder such that we can calculate and visualize observables using Kwant's functionalities. To do this we simply provide the mean-field solution `mf_sol` as well as the sites and periods of the bulk system to the `tb_to_builder` function. We then wrap the system and finalize it.

```{code-cell} ipython3
mf_sol_builder = utils.tb_to_builder(
    mf_sol, list(bulk.sites()), bulk.symmetry.periods
)

wrapped_syst_mfsol = kwant.wraparound.wraparound(mf_sol_builder)
wrapped_fsyst_mfsol = wrapped_syst.finalized()
```

We want to look at the magnetization of the system. To do this we first need to define the magnetization direction. We do this by arbitrarily choosing the spin direction of one of the sites and defining the magnetization with respect to this direction.

```{code-cell} ipython3
_, reference_value = list(mf_sol_builder.site_value_pairs())[0]

sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

magnetization_p_direction = []
for sigma in [sigma_0, sigma_x, sigma_y, sigma_z]:
    magnetization_p_direction.append(np.trace(sigma@reference_value)*sigma)

reference_magnetization = sum(magnetization_p_direction)
reference_magnetization = reference_magnetization / np.linalg.norm(reference_magnetization)
```
