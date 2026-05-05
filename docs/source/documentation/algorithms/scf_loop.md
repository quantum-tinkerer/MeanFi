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
# SCF loop

The outer solve in `MeanFi` is a fixed-point problem for the density matrix, or equivalently for a reduced real parameter vector derived from it.

## Fixed-point map

Let $\theta$ denote the reduced real parametrization of the current density state.
One SCF iteration does the following:

1. reconstruct the density matrix or BdG density state from $\theta$,
2. build the corresponding mean-field correction,
3. solve the fixed-filling density problem for that Hamiltonian,
4. convert the new density result back to reduced parameters.

This defines a map

:::{math}
\mathcal{G}(\theta) = \theta_{\mathrm{new}},
:::

and the SCF solve seeks

:::{math}
\mathcal{F}(\theta) = \mathcal{G}(\theta) - \theta = 0.
:::

## Separation of responsibilities

The outer SCF method does **not** solve for the chemical potential directly.
That work is delegated to the inner fixed-filling density evaluation.
This separation keeps the outer solver focused on the mean-field fixed point rather than mixing it with the filling constraint.

## SCF methods

The public `solver(...)` entry point accepts an explicit SCF method through `scf=...`.
Current built-in methods include:

- `AndersonMixing(...)`
- `LinearMixing(...)`

The default solver path uses Anderson mixing with conservative settings.

## Output

Once the fixed point converges, `MeanFi` converts the converged density result back into a mean-field correction and returns that as `result.mf`.
