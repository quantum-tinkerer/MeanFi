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
# Theory

`MeanFi` is built for interacting fermionic systems, especially electron models written in a tight-binding basis.
The theoretical problem starts from a quadratic single-particle Hamiltonian plus a quartic interaction,

:::{math}
\hat H = \hat H_0 + \hat V,
\qquad
\hat H_0 = \sum_{ij} h_{ij} c_i^\dagger c_j.
:::

The quadratic part can be diagonalized directly.
The difficulty is the interaction, because it couples many-body states rather than single-particle amplitudes.
That is the point where an approximation becomes necessary.

(theory-interaction)=
## General interaction and restricted structure

In a general fermionic model, the interaction has the four-index form

:::{math}
\hat V
=
\frac12
\sum_{ijkl}
V_{ijkl}\,
c_i^\dagger c_j^\dagger c_l c_k.
:::

This is the broad theoretical setting: the Hamiltonian is quadratic plus quartic, and the quartic term is the source of the many-body complexity.
Numerically, one must replace that many-body object by something that can be evaluated in an effective single-particle language.

:::{note}
Numerical task:
build the self-consistent quadratic Hamiltonian from the current state.
See {ref}`Algorithm overview: build the self-consistent quadratic Hamiltonian <algo-build>`.
:::

`MeanFi` currently focuses on density-density interactions,

:::{math}
\hat V
=
\frac12 \sum_{ij} v_{ij}\, n_i n_j,
\qquad
n_i = c_i^\dagger c_i.
:::

This simplifies the coefficient structure from a four-index tensor to a two-index coupling matrix $v_{ij}$, but the operator is still quartic in the fermion fields.
The modeling restriction is therefore in how the interaction is represented, not in whether the problem remains genuinely interacting.

:::{note}
Numerical task:
use the restricted interaction structure to determine which matrix entries contribute to the mean-field update.
See {ref}`Algorithm overview: build the self-consistent quadratic Hamiltonian <algo-build>`.
:::

(theory-tight-binding)=
## Tight-binding and real-space notation

For translationally invariant systems, the single-particle label separates into an internal orbital index and a Bravais-lattice translation $R$.
The same Hamiltonian can then be written in real-space tight-binding form,

:::{math}
\hat H_0
=
\sum_{R}\sum_{mn}
h_{mn}(R)\,
c_{m,0}^\dagger c_{n,R}.
:::

This is the notation that connects directly to how models are specified in `MeanFi`: each lattice vector $R$ carries a matrix over the internal degrees of freedom.
Once the model is in this form, the main numerical task is no longer building the operator, but evaluating the state it defines.

:::{note}
Numerical task:
evaluate the density by combining Brillouin-zone integration with single-$k$ density evaluation.
See {ref}`Algorithm overview: integrate the density over the Brillouin zone <algo-bz>`.
:::

(theory-mean-field)=
## Mean-field approximation

The mean-field step replaces the interacting many-body Hamiltonian by a self-consistent quadratic one.
In Wick-decoupled form, the quartic interaction is approximated by bilinears weighted by expectation values of operator pairs,

:::{math}
\hat V_{\mathrm{MF}}
\sim
\frac12
\sum_{ijkl}
V_{ijkl}
\left(
\langle c_i^\dagger c_k \rangle c_j^\dagger c_l
+
\langle c_j^\dagger c_l \rangle c_i^\dagger c_k
-
\langle c_i^\dagger c_l \rangle c_j^\dagger c_k
-
\langle c_j^\dagger c_k \rangle c_i^\dagger c_l
\right)
\;+\;
\frac12
\sum_{ijkl}
V_{ijkl}
\left(
\langle c_i^\dagger c_j^\dagger \rangle c_l c_k
+
\langle c_l c_k \rangle c_i^\dagger c_j^\dagger
\right).
:::

The contractions $\langle c_i^\dagger c_j \rangle$ define the normal density matrix, and the contractions $\langle c_i^\dagger c_j^\dagger \rangle$ or $\langle c_i c_j \rangle$ define anomalous pairing densities when superconductivity is allowed.
Even when superconductivity is allowed structurally, the central idea is the same: the interacting problem is replaced by a quadratic Hamiltonian whose coefficients depend on expectation values of the state itself.
That makes the main numerical question a self-consistent one rather than an exact many-body diagonalization.

:::{note}
Numerical task:
iterate the map from expectation values to a quadratic Hamiltonian and back again until it becomes self-consistent.
See {ref}`Algorithm overview: build the self-consistent quadratic Hamiltonian <algo-build>`.
:::

(theory-density-matrix)=
## Density matrix

The natural object for those expectation values is the density matrix,

:::{math}
\rho_{ij} = \langle c_i^\dagger c_j \rangle,
\qquad
\rho_{mn}(R) = \langle c_{m,0}^\dagger c_{n,R} \rangle
\quad
\text{in the translationally invariant basis.}
:::

For density-density Hartree-Fock problems, this object is enough to build the normal mean-field correction.
More generally, it is the single-particle summary of the state from which observables and self-consistent updates are assembled.
Numerically, this is the object that has to be evaluated from the effective quadratic Hamiltonian.

:::{note}
Numerical task:
compute the density matrix from the effective quadratic Hamiltonian, using Brillouin-zone integration and a single-$k$ density evaluator.
See {ref}`Algorithm overview: integrate the density over the Brillouin zone <algo-bz>` and {ref}`Algorithm overview: evaluate the density at one sampled $k$ <algo-single-k>`.
:::

(theory-filling)=
## Self-consistency and filling

The density matrix is both input and output of the mean-field problem:

:::{math}
\rho
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[\rho]
\;\longrightarrow\;
\rho'.
:::

A mean-field solution is a fixed point of this map, together with whatever filling constraint is imposed.
In the fixed-filling setting used by default in `MeanFi`, the density must satisfy

:::{math}
N(\mu) = \nu,
:::

so the chemical potential is part of the self-consistent density-evaluation problem rather than an external input.
In translationally invariant systems, evaluating $\rho$ therefore requires both solving the filling constraint and integrating the occupied single-particle states over momentum space or, at finite temperature, applying the corresponding occupation function to the spectrum.

:::{note}
Numerical task:
solve the filling constraint together with the density evaluation.
See {ref}`Algorithm overview: solve the filling constraint <algo-filling>`, {ref}`Algorithm overview: integrate the density over the Brillouin zone <algo-bz>`, and {ref}`Algorithm overview: evaluate the density at one sampled $k$ <algo-single-k>`.
:::

```{toctree}
:hidden:
:maxdepth: 1

extensions.md
```

The theory section is intentionally short:

- [Theory extensions](./extensions.md): superconductivity and finite temperature as extensions of the same density-evaluation problem

For the numerical side of each theoretical step, continue with the [Algorithm overview](../algorithm.md).
