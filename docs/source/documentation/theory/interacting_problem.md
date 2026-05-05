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
# Interacting problem

`MeanFi` starts from a single-particle tight-binding Hamiltonian plus a density-density interaction,

:::{math}
:label: hamiltonian
\hat{H} = \hat{H_0} + \hat{V}
= \sum_{ij} h_{ij} c^\dagger_i c_j
+ \frac12 \sum_{ij} v_{ij} c_i^\dagger c_j^\dagger c_j c_i.
:::

Here $c_i^\dagger$ and $c_i$ create and annihilate fermions in the single-particle basis states labelled by $i$.
The non-interacting part $\hat{H_0}$ is easy to solve directly.
The interacting part $\hat{V}$ is quartic in fermion operators, which makes the exact many-body problem expensive very quickly.

## Translationally invariant tight-binding form

For translationally invariant systems, the single-particle label separates into:

- internal indices inside the unit cell,
- a Bravais-lattice translation vector $R$.

That means matrix elements depend only on relative displacement:

:::{math}
\rho_{mn}(R) = \langle c^\dagger_{m,0} c_{n,R}\rangle.
:::

This is the tight-binding dictionary representation used throughout the code:

- each key is a lattice vector $R$,
- each value is the matrix acting on the internal degrees of freedom.

The density matrix and the mean-field correction are represented in exactly the same way.

## Filling and chemical potential

Most self-consistent calculations in `MeanFi` are done at fixed filling.
In that case, each density evaluation also has to solve for a chemical potential $\mu$ such that the resulting density has the requested trace per unit cell.

However, the package also exposes fixed-$\mu$ density calculations through {autolink}`~meanfi.density_matrix_at_mu`.
So the filling constraint is central to the default self-consistent workflow, but it is not the only density-evaluation mode available in the package.
