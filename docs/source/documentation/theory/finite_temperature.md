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
# Finite temperature

The density matrix is computed from the Fermi-Dirac occupation function

:::{math}
f(\varepsilon) = \frac{1}{e^{\varepsilon / kT} + 1}.
:::

At finite temperature, this smooths the occupation near the Fermi level.

## Why `kT` matters numerically

A nonzero `kT` helps in two ways:

- the density matrix becomes a smoother function of the spectrum,
- the fixed-filling solve becomes better conditioned because the filling changes more smoothly with $\mu$.

This is why the finite-temperature algorithms are built around adaptive quadrature and matrix-function evaluation at each sampled $k$ point.

## Zero temperature versus finite temperature

`MeanFi` currently uses two different numerical viewpoints:

- for `kT > 0`, the density matrix is treated through finite-temperature matrix functions and adaptive quadrature or fixed grids,
- for `kT = 0`, the normal-state default uses a dedicated adaptive simplicial backend, while superconducting calculations require an explicit `UniformGrid(...)`.

So `kT` is not just a physical parameter here.
It also determines which families of numerical methods are available by default.
