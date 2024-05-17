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

## Self-consistent mean-field loop

To calculate the mean-field interaction in {eq}`mf_infinite`, we require the ground-state density matrix $\rho_{mn}(R)$.
However, {eq}`density` is a function of the mean-field interaction $\hat{V}_{\text{MF}}$ itself.
Therefore, we need to solve for both self-consistently.

A single iteration of this self-consistency loop is a function that computes a new mean-field correction from a given one:

$$
\text{MF}(\hat{V}_{\text{init, MF}}) \to \hat{V}_{\text{new, MF}},
$$

which is defined in {autolink}`~meanfi.model.Model.mfield` method.
It performs the following steps:
1. Calculate the total Hamiltonian $\hat{H}(R) = \hat{H_0}(R) + \hat{V}_{\text{init, MF}}(R)$ in real-space.
2. ({autolink}`~meanfi.mf.density_matrix`) Compute the ground-state density matrix $\rho_{mn}(R)$:
    1. ({autolink}`~meanfi.tb.transforms.tb_to_kgrid`) Fourier transform the total Hamiltonian to momentum space $\hat{H}(R) \to \hat{H}(k)$.
    2. ({autolink}`numpy.linalg.eigh`) Diagonalize the Hamiltonian $\hat{H}(R)$ to obtain the eigenvalues and eigenvectors.
    3. ({autolink}`~meanfi.mf.fermi_on_kgrid`) Calculate the fermi level given the desired filling of the unit cell.
    4.  ({autolink}`~meanfi.mf.density_matrix_kgrid`) Calculate the density matrix $\rho_{mn}(k)$ using the eigenvectors and the fermi level.
    5. ({autolink}`~meanfi.tb.transforms.kgrid_to_tb`) Inverse Fourier transform the density matrix to real-space $\rho_{mn}(k) \to \rho_{mn}(R)$.
3. ({autolink}`~meanfi.mf.meanfield`) Calculate the new mean-field correction $\hat{V}_{\text{new, MF}}(R)$ using {eq}`mf_infinite`.

## Self-consistency criteria

To define the self-consistency condition, we first introduce an invertible function $f$ that uniquely maps $\hat{V}_{\text{MF}}$ to a real-valued vector which minimally parameterizes it:

$$
f : \hat{V}_{\text{MF}} \to f(\hat{V}_{\text{MF}}) \in \mathbb{R}^N.
$$

In the code, $f$ corresponds to the {autolink}`~meanfi.params.rparams.tb_to_rparams` function (inverse is {autolink}`~meanfi.params.rparams.rparams_to_tb`).
Currently, $f$ parameterizes the mean-field interaction by taking only the upper triangular elements of the matrix $V_{\text{MF}, nm}(R)$ (the lower triangular part is redundant due to the Hermiticity of the Hamiltonian) and splitting it into real and imaginary parts to form a real-valued vector.

With this, we define the self-consistency criterion as a fixed-point problem:

$$
f(\text{MF}(\hat{V}_{\text{MF}})) = f(\hat{V}_{\text{MF}}).
$$

Instead of solving the fixed point problem, we rewrite it as the difference of the two successive self-consistent mean-field iterations in {autolink}`~meanfi.solvers.cost_mf`.
That re-defines the problem into a root-finding problem which is more consistent with available numerical solvers such as {autolink}`~scipy.optimize.anderson`.
That is exactly what we do in the {autolink}`~meanfi.solvers.solver` function, although we also provide the option to use a custom optimizer.
