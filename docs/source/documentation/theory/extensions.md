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
# Theory extensions

This page collects the two main extensions of the core normal-state theory: superconducting pairing and finite temperature.
Both preserve the same overall structure as the main theory page.
The numerical task is still to evaluate a density object for an effective quadratic Hamiltonian and update it self-consistently.

(theory-superconducting)=
## Superconducting extension

When `superconducting=True`, the mean-field problem does not stop at the normal density matrix.
It also includes anomalous pairing averages of the form

:::{math}
F_{mn}(R) = \langle c_{m,0} c_{n,R}\rangle,
:::

so the quadratic mean-field Hamiltonian contains both normal bilinears and pairing terms.
This does not change the conceptual role of mean field: the interacting problem is still replaced by a quadratic Hamiltonian whose coefficients are determined by the state.
What changes is that the state now carries both normal and anomalous information.

:::{note}
Numerical task:
extend the self-consistent state to include anomalous densities and solve the corresponding fixed-filling problem.
See {ref}`Algorithm overview: build the self-consistent quadratic Hamiltonian <algo-build>` and {ref}`Algorithm overview: solve the filling constraint <algo-filling>`.
:::

(theory-finite-temperature)=
## Finite temperature

At finite temperature, the density is determined by the Fermi-Dirac occupation function

:::{math}
f(\varepsilon) = \frac{1}{e^{\varepsilon / kT} + 1}.
:::

Instead of a sharp occupation of states, the density becomes a smooth spectral function of the effective quadratic Hamiltonian.
This changes both the physics near the Fermi level and the numerical character of the density evaluation.

:::{note}
Numerical task:
evaluate a temperature-smeared density, which changes both the filling solve and the single-$k$ matrix-function step.
See {ref}`Algorithm overview: solve the filling constraint <algo-filling>` and {ref}`Algorithm overview: evaluate the density at one sampled $k$ <algo-single-k>`.
:::

## Combined viewpoint

Superconductivity and finite temperature are not separate theories in `MeanFi`.
They are extensions of the same self-consistent density-evaluation problem introduced on the main theory page.
The detailed numerical choices are summarized in the [Algorithm overview](../algorithm.md) and documented more fully in the [Algorithm reference](../algorithms/index.md).
