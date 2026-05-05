---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Mean-field approximation

The Hartree-Fock approximation replaces the quartic interaction by a bilinear operator weighted by expectation values of the density matrix.
For density-density interactions, the normal-state mean-field approximation is

:::{math}
:label: mf_approx
\hat{V}_{\mathrm{MF}}
= \frac12 \sum_{ij} v_{ij}
\left[
\langle c_i^\dagger c_i \rangle c_j^\dagger c_j
- \langle c_i^\dagger c_j \rangle c_j^\dagger c_i
\right],
:::

up to constant energy shifts that do not affect the self-consistent solve.

## Tight-binding correction

In the translationally invariant tight-binding setting, the mean-field correction becomes another tight-binding dictionary.
The normal-state correction has two parts:

- a direct onsite Hartree contribution,
- an exchange contribution on the same keys as the interaction support.

In code, this is the logic implemented by {autolink}`~meanfi.meanfield`.

Writing the density matrix as $\rho_{mn}(R)$ and the interaction as $v_{mn}(R)$, the correction has the form

:::{math}
:label: mf_infinite
V_{\mathrm{MF},mn}(R)
= \delta(R)\delta_{mn} \sum_i \rho_{ii}(0) v_{in}(0)
- \rho_{mn}(R) v_{mn}(R).
:::

## Self-consistency

The unknown density matrix appears on both sides of the problem:

1. it defines the mean-field correction,
2. the mean-field correction defines the Hamiltonian,
3. the Hamiltonian defines a new density matrix.

So the goal is to find a fixed point of that map while also satisfying the target filling.

The algorithm pages describe how `MeanFi` turns this into:

- a reduced real parameter vector,
- an outer SCF fixed-point problem,
- and an inner fixed-filling density evaluation.
