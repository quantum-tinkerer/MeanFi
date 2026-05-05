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
# Matrix-function backends

Once a k-space node is chosen, `MeanFi` still has to compute the density matrix contribution at that node.
This is the role of the matrix-function backend.

## Single-$k$ viewpoint

At a fixed $k$, the backend evaluates the density matrix of the shifted Hamiltonian

:::{math}
H(k) - \mu Q
:::

or the corresponding normal-state version, and then extracts the relevant density block.

Conceptually, this is the work performed by the `density_block(...)` layer.

## Exact diagonalization

The direct path diagonalizes the sampled matrix explicitly and evaluates the occupation function from the eigenvalues.
It is straightforward and robust, but dense diagonalization becomes expensive as the matrix size grows.

## Rational FOE

The rational FOE path approximates the same matrix function without full diagonalization.
It is especially useful in sparse finite-temperature calculations where exact diagonalization would be much heavier.

## Dense versus sparse behavior

Dense and sparse backends are not just storage choices.
They also affect which matrix-function strategies are practical:

- dense problems default to direct diagonalization,
- sparse finite-temperature problems default to `RationalFOE("aaa")`,
- some rational paths support more features on sparse matrices than on dense ones.

That is why the integration-family choice and the matrix-function choice are documented separately.
