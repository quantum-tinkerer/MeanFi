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

# Interacting graphene

We use Kwant to build spinful graphene with onsite repulsion $U$ and
nearest-neighbor repulsion $V$, then map its mean-field phases.

## Build the model

```{code-cell} ipython3
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from meanfi.interop import kwant as utils

s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1, -1])

graphene = kwant.lattice.general(
    [(1, 0), (1 / 2, np.sqrt(3) / 2)], [(0, 0), (0, 1 / np.sqrt(3))], norbs=2
)
a, b = graphene.sublattices
bulk = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
bulk[a(0, 0)] = bulk[b(0, 0)] = 0 * s0
bulk[graphene.neighbors()] = s0
h_0 = utils.builder_to_tb(bulk)


def onsite_interaction(site, U):
    return U * sx


def bond_interaction(site1, site2, V):
    return V * np.ones((2, 2))


interaction = utils.build_interacting_syst(
    bulk, graphene, onsite_interaction, bond_interaction, max_neighbor=1
)
h_int = utils.builder_to_tb(interaction, params={"U": 0.2, "V": 1.2})
model = meanfi.Model(h_0, h_int, filling=2)
result = meanfi.solver(model, model.random_meanfield(rng=0, scale=0.05))
```

The onsite matrix couples opposite spins. The bond matrix couples all spin pairs
on neighboring sites. `filling=2` means two electrons per four-orbital cell.

## Measure charge and spin order

A charge density wave (CDW) gives different occupations on the two sublattices.
Its operator is $\sigma_z\otimes I$, with sublattice first and spin second.
Request the onsite density block to evaluate it. Reuse the solved chemical
potential instead of repeating the filling search.

```{code-cell} ipython3
cdw_operator = {(0, 0): np.kron(sz, s0)}
sdw_operators = [{(0, 0): np.kron(sz, spin)} for spin in (sx, sy, sz)]
h = model.hamiltonian_from_meanfield(result.mean_field)
rho = meanfi.density_matrix_at_mu(h, mu=result.mu, keys=[(0, 0)]).to_tb()
print(f"CDW amplitude: {abs(meanfi.expectation_value(rho, cdw_operator)):.3f}")
```

Spin density wave (SDW) order uses $\sigma_z\otimes\boldsymbol\sigma$.
Summing the squares of its three components makes the magnitude independent of
the spin direction chosen by SCF.

## Scan the phase diagram

For this qualitative scan we use `tol=1e-2` to keep execution quick. The default
EDIIS method and tolerance policy are unchanged. Each point is calculated afresh.
The separate band grid includes the Dirac points and measures the gap between
the second and third bands at half filling.

```{code-cell} ipython3
Us = np.linspace(0, 4, 10)
Vs = np.linspace(0, 1.5, 10)
gaps = np.empty((len(Us), len(Vs)))
cdw = np.empty_like(gaps)
sdw = np.empty_like(gaps)

for i, U in enumerate(Us):
    for j, V in enumerate(Vs):
        h_int = utils.builder_to_tb(interaction, params={"U": U, "V": V})
        model = meanfi.Model(h_0, h_int, filling=2)
        result = meanfi.solver(
            model, model.random_meanfield(rng=0, scale=0.05), tol=1e-2
        )
        h = model.hamiltonian_from_meanfield(result.mean_field)
        bands = np.linalg.eigvalsh(meanfi.tb_to_kgrid(h, (60, 60)))
        gaps[i, j] = max(0.0, bands[..., 2].min() - bands[..., 1].max())
        rho = meanfi.density_matrix_at_mu(h, mu=result.mu, keys=[(0, 0)]).to_tb()
        cdw[i, j] = abs(meanfi.expectation_value(rho, cdw_operator))**2
        sdw[i, j] = sum(abs(meanfi.expectation_value(rho, op))**2 for op in sdw_operators)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
for ax, values, title in zip(axes, [gaps, cdw, sdw], ["Gap", "CDW squared", "SDW squared"]):
    image = ax.imshow(values.T, origin="lower", aspect="auto",
                      extent=(Us[0], Us[-1], Vs[0], Vs[-1]))
    ax.set(title=title, xlabel="U", ylabel="V")
    fig.colorbar(image, ax=ax)
plt.show()
```

Large onsite repulsion favors SDW order; large neighbor repulsion favors CDW
order. This coarse scan illustrates the workflow, rather than locating precise
phase boundaries. As with other mean-field calculations, competing initial
guesses can reveal different self-consistent states.
