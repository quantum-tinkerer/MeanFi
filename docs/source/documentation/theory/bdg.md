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
# BdG extension

When `superconducting=True`, `MeanFi` does not stop at the normal density matrix.
It also solves for anomalous averages of the form

:::{math}
F_{mn}(R) = \langle c_{m,0} c_{n,R}\rangle,
:::

which describe pairing correlations.

## Electron-first BdG structure

The superconducting problem is represented in electron-first BdG form.
At each lattice key $R$, the mean-field correction is a block matrix

:::{math}
\mathcal{V}_{\mathrm{BdG}}(R) =
\begin{pmatrix}
V_{\mathrm{normal}}(R) & \Delta(R) \\
\Delta^\dagger(-R) & -V_{\mathrm{normal}}^T(-R)
\end{pmatrix}.
:::

This structure is enforced explicitly in the code and reduces the number of independent parameters that have to be tracked.

## Normal and anomalous channels

The superconducting correction has two logically different pieces:

- the normal block, which is still built from density-density Hartree-Fock terms,
- the anomalous block, which is built from pairing amplitudes and the same interaction support.

In `MeanFi`, the normal block uses the same normal-state mean-field machinery as before, while the anomalous block is assembled separately from the anomalous density.

## Why fixed filling is harder in BdG

In the normal problem, changing $\mu$ simply shifts the single-particle spectrum.
In BdG, the charge operator is no longer the identity in the effective single-particle space.
Instead the Hamiltonian is shifted by

:::{math}
\mathcal{H}_{\mathrm{BdG}} - \mu Q,
\qquad
Q = \mathrm{diag}(+1,\dots,+1,-1,\dots,-1).
:::

Because $Q$ generally does not commute with the BdG Hamiltonian, changing $\mu$ means the matrix has to be re-evaluated and re-diagonalized or re-processed by the chosen backend.
That is one reason superconducting fixed-filling calculations are heavier than normal-state ones.
