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
# Algorithm overview

## Self-consistency loop

In order to calculate the mean-field interaction in {eq}`mf_infinite`, we require the ground-state density matrix $\rho_{mn}(R)$.
However, the density matrix in {eq}`density` is a functional of the mean-field interaction $\hat{V}_{\text{MF}}$ itself.
Therefore, we need to solve for both self-consistently.

We define a single iteration of a self-consistency loop:

$$
\text{SCF}(\hat{V}_{\text{init, MF}}) \to \hat{V}_{\text{new, MF}},
$$

such that it performs the following operations given an initial mean-field interaction $\hat{V}_{\text{init, MF}}$:

1. Calculate the total Hamiltonian $\hat{H}(R) = \hat{H_0}(R) + \hat{V}_{\text{init, MF}}(R)$ in real-space.
2. Fourier transform the total Hamiltonian to the momentum space $\hat{H}(R) \to \hat{H}(k)$.
3. Calculate the ground-state density matrix $\rho_{mn}(k)$ in momentum space.
    1. Diagonalize the Hamiltonian $\hat{H}(k)$ to obtain the eigenvalues and eigenvectors.
    2. Calculate the fermi level $\mu$ given the desired filling of the unit cell.
    3. Calculate the density matrix $\rho_{mn}(k)$ using the eigenvectors and the fermi level $\mu$ (currently we do not consider thermal effects so $\beta \to \infty$).
4. Inverse Fourier transform the density matrix to real-space $\rho_{mn}(k) \to \rho_{mn}(R)$.
5. Calculate the new mean-field interaction $\hat{V}_{\text{new, MF}}(R)$ via {eq}`mf_infinite`.

## Self-consistency criterion

To define the self-consistency condition, we first introduce an invertible function $f$ that uniquely maps $\hat{V}_{\text{MF}}$ to a real-valued vector which minimally parameterizes it:

$$
f : \hat{V}_{\text{MF}} \to f(\hat{V}_{\text{MF}}) \in \mathbb{R}^N.
$$

Currently, $f$ parameterizes the mean-field interaction by taking only the upper triangular elements of the matrix $V_{\text{MF}, nm}(R)$ (the lower triangular part is redundant due to the Hermiticity of the Hamiltonian) and splitting it into a real and imaginary parts to form a real-valued vector.

With this function, we define the self-consistency criterion as a fixed-point problem:

$$
f(\text{SCF}(\hat{V}_{\text{MF}})) = f(\hat{V}_{\text{MF}}).
$$

To solve this fixed-point problem, we utilize a root-finding function `scipy.optimize.anderson` which uses the Anderson mixing method to find the fixed-point solution.
However, our implementation also allows to use other custom fixed-point solvers by either providing it to `solver` or by re-defining the `solver` function.
