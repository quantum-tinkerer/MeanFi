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

# Rhombohedral graphene on a momentum disk

Use a continuum Hamiltonian, general interactions and adaptive FermiSimplex at
the parameter points of [Liu and Wang, Fig. 2](https://arxiv.org/pdf/2401.13413v2).
We use one random-guess rule and one tolerance policy throughout. These runs
explore self-consistent states; they do not select the paper's named phases.
Panels (b) and (c) share parameters, so we show that point once. The four
columns are labeled by their Hamiltonian parameters.

## Hamiltonian and disk integration

The [Hamiltonian helper](scripts/rhombohedral_graphene.py) implements the paper's
pentalayer model: ten sublattice sites, two valleys and two spins, giving 40
orbitals. Energies are in eV and momentum is $q=a_0k$. It uses the supplement's
onsite-offset convention.

```{code-cell} ipython3
from dataclasses import replace

import matplotlib.pyplot as plt
import meanfi as mf
import numpy as np
from numba import njit

from scripts.rhombohedral_graphene import Graphene
from scripts.graphene_figure import plot_bands
```

MeanFi integrates over $[0,2\pi]^2$. The concentric equal-area map transforms this
square into the cutoff disk $|q|\leq\Lambda$, with $\Lambda=0.16$:

```{code-cell} ipython3
def disk_coordinates(k1, k2, cutoff):
    """Concentric equal-area map from the computational BZ to a momentum disk."""
    a, b = k1 / np.pi - 1, k2 / np.pi - 1
    if a == 0 and b == 0:
        return 0.0, 0.0
    if abs(a) > abs(b):
        radius, angle = cutoff * a, np.pi / 4 * b / a
    else:
        radius, angle = cutoff * b, np.pi / 2 - np.pi / 4 * a / b
    return radius * np.cos(angle), radius * np.sin(angle)
```

Its constant Jacobian gives a normalized disk average,

$$
\int\frac{dk_1dk_2}{(2\pi)^2}f(q(k))
=\frac{1}{\pi\Lambda^2}\int_{|q|<\Lambda}d^2q\,f(q).
$$

The Cartesian Hamiltonian is affine: $h(q)=h_c+q_xh_x+q_yh_y$.
We compile its disk wrapper with Numba (`pip install numba`):

```{code-cell} ipython3
compiled_disk_coordinates = njit(disk_coordinates)


def disk_hamiltonian(parameters):
    cartesian = parameters.affine_hamiltonian()
    constant = cartesian(0, 0)
    hx = cartesian(1, 0) - constant
    hy = cartesian(0, 1) - constant
    cutoff = parameters.cutoff

    @njit
    def h(k1, k2):
        qx, qy = compiled_disk_coordinates(k1, k2, cutoff)
        return constant + qx * hx + qy * hy

    return h


parameters = Graphene()
h0 = mf.BlochHamiltonian(disk_hamiltonian(parameters))
```

MeanFi infers dimension and orbital count from the function and its matrix.
Other equal-measure domain maps work the same way. A varying Jacobian requires
weighted integration; multiplying the Hamiltonian by it changes the physics.

## General interactions

On each sublattice, the interaction is

$$
\frac U2:n^2:+Vn_Kn_{K'}-J_H\sum_{a=x,y,z}S_K^aS_{K'}^a.
$$

`BilinearTerm(g, A, B)` represents $g:(c^\dagger A c)(c^\dagger B c):$.
The spin operators below use Pauli matrices without a factor of $1/2$.
Because our density is a disk average, the paper's couplings must be multiplied
by $w=(\sqrt3/2)\Lambda^2/(4\pi)$. Both valleys are already in the Hamiltonian.

The complete interaction construction is:

```{code-cell} ipython3
IDENTITY = np.eye(2)
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
VALLEYS = (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))


def graphene_interaction(*, sites, u, v, hund, measure):
    """Embed each sublattice's four-flavor operators in the full orbital space."""
    plus, minus = [np.kron(p, IDENTITY) for p in VALLEYS]
    terms = []
    for site in range(sites):
        projector = np.diag(np.arange(sites) == site)

        def embed(operator):
            return np.kron(projector, operator)

        number = embed(np.eye(4))
        terms.append(mf.BilinearTerm(measure * u / 2, number, number))
        terms.append(mf.BilinearTerm(measure * v, embed(plus), embed(minus)))
        for spin in PAULI:
            terms.append(
                mf.BilinearTerm(
                    -measure * hund,
                    embed(np.kron(VALLEYS[0], spin)),
                    embed(np.kron(VALLEYS[1], spin)),
                )
            )
    return mf.BilinearInteraction(terms)
```

```{code-cell} ipython3
interaction = graphene_interaction(
    sites=10, u=40.0, v=-8.0, hund=parameters.hund, measure=parameters.measure,
)
model = mf.Model(h0, interaction, filling=20)
```

## Solve the parameter points

The same loose tolerances apply to every solve. The fixed random seed makes
runs reproducible; it does not prescribe a spin, valley or layer ordering.
Different initial guesses can converge to different self-consistent states.

```{code-cell} ipython3
parameter_points = [(-0.012, 10.0), (-0.015, 10.0), (-0.015, 5.0), (-0.019, 5.0)]
tol = replace(
    mf.default_solver_tolerances(5e-3),
    charge_integration=2e-2,
    filling_residual=5e-3,
)
solutions = []
for delta, hund in parameter_points:
    p = Graphene(delta=delta, hund=hund)
    h0 = mf.BlochHamiltonian(disk_hamiltonian(p))
    interaction = graphene_interaction(
        sites=10, u=40.0, v=-8.0, hund=hund, measure=p.measure,
    )
    model = mf.Model(h0, interaction, filling=20)
    guess = model.random_meanfield(rng=81, scale=0.025)
    result = mf.solver(
        model, guess, integration=mf.FermiSimplex(), scf=mf.EnergyDIIS(), tol=tol,
    )
    solutions.append((p, result))

plot_bands(solutions)
plt.show()
```

The plots show the full Hamiltonian's bands nearest half filling, colored by
outer-surface polarization, without assuming spin or valley conservation.
`result.mean_field`, `result.density` and `result.internal_energy` are available
as usual. EDIIS uses relative energies during iteration; absolute energy is
computed once on the final mesh.
