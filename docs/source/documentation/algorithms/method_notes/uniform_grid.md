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
# `UniformGrid`

`UniformGrid` samples the Brillouin zone on a fixed tensor-product grid.

It approximates the Brillouin-zone integral by a Riemann sum:

:::{math}
\rho(R) \approx \frac{1}{N_k} \sum_{k \in \mathcal{G}} e^{i k \cdot R}\,\rho(k),
:::

where $\mathcal{G}$ is the chosen uniform k-grid.

## Main idea

If `nk` is the number of points per momentum direction and the problem is $d$-dimensional, then the grid contains

:::{math}
N_k = nk^d
:::

points.

Every point is evaluated explicitly.
There is no adaptive refinement and no k-space error estimator beyond changing the grid resolution by hand.

## Why it is still useful

`UniformGrid` remains useful because it is:

- simple,
- predictable,
- easy to compare against reference calculations,
- currently the explicit path for zero-temperature BdG calculations.

## Cost versus error scaling

Its cost is directly proportional to the number of sampled points:

:::{math}
\text{cost} \sim nk^d \times C_k.
:::

This becomes expensive quickly with dimension, but the method has the advantage that the sampling pattern is fully transparent.
For a plain fixed-grid Riemann sum on a smooth integrand, the generic algebraic behavior is

:::{math}
\varepsilon \sim nk^{-1},
:::

then

:::{math}
\;nk \sim \varepsilon^{-1},
\qquad
\text{cost} \sim \varepsilon^{-d} C_k.
:::

## Backend pairing

`UniformGrid` can still be paired with:

- direct diagonalization,
- or rational FOE,

depending on the matrix structure and temperature.
