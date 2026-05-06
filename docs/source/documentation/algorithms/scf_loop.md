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

The outer solve in `MeanFi` is a fixed-point problem for the self-consistent state.
That state may be represented directly as a density object or through a reduced real parametrization described in [Parametrization and symmetry reduction](./parametrization.md).

## Fixed-point map

One SCF iteration has the structure

:::{math}
\rho_n
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[\rho_n]
\;\longrightarrow\;
\rho_{n+1},
:::

where the density update already includes the fixed-filling solve and the Brillouin-zone density evaluation.
So the SCF problem is the fixed-point equation

:::{math}
\rho = \mathcal{G}(\rho).
:::

If the state is represented internally by a reduced real parametrization $\theta$, the same structure becomes

:::{math}
\theta_{n+1} = \mathcal{G}(\theta_n),
\qquad
\theta = \mathcal{G}(\theta).
:::

## SCF methods

The public `solver(...)` entry point accepts an explicit SCF method through `scf=...`.
Current built-in methods include:

- `AndersonMixing(...)`
- `LinearMixing(...)`

The default solver path uses Anderson mixing with conservative settings.

## Output

Once the fixed point converges, `MeanFi` converts the converged density result back into a mean-field correction and returns that as `result.mf`.
