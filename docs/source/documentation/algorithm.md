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
## Algorithm Overview

### Self-Consistent Mean-Field Loop

To calculate the mean-field interaction in {eq}`mf_infinite`, we require the ground-state density matrix $\rho_{mn}(R)$. However, {eq}`density` depends on both the chemical potential $\mu$ and the mean-field interaction $\hat{V}_{\text{MF}}$ itself. Hence, these must be determined self-consistently.

A single iteration of the self-consistency loop can be viewed as a function that, given an initial density matrix $\rho_{mn}(R)_{\text{init}}$ and a chemical potential $\mu$, computes a new density matrix $\rho_{mn}(R)_{\text{new}}$:

$$
\text{MF}(\rho_{mn}(R)_{\text{init}},\mu) \to \rho_{mn}(R)_{\text{new}}.
$$

This iteration is defined in the {autolink}`~meanfi.model.Model.density_matrix` method and proceeds as follows:

1. **Mean-field calculation** ({autolink}`~meanfi.mf.meanfield`): Compute the new mean-field potential $\hat{V}_{\text{new, MF}}(R)$ using {eq}`mf_infinite`.
2. **Density matrix update** ({autolink}`~meanfi.mf.density_matrix`):
   1. **Momentum-space Hamiltonian** ({autolink}`~meanfi.tb.transforms.tb_to_kfunc`): Transform the real-space tight-binding Hamiltonian into a function of $k$ that returns $\hat{H}(k)$.
   2. **Momentum-space density** (`density_matrix_k`): Define the momentum-space density matrix $\rho_{mn}(k)$ given $\hat{H}(k)$ and $\mu$.
   3. **Fourier transform integrand** (`integrand`): Provide the integrand that transforms $\rho_{mn}(k)$ back to real space $\rho_{mn}(R)$. A provisional bandwidth estimation is included for constraining $\mu$, though further refinements may be needed.
   4. **Integration** (`scipy.integrate.cubature`): Integrate over momentum space to obtain the new real-space density matrix $\rho_{mn}(R)_{\text{new}}$.

### Self-Consistency Criteria

To define the self-consistency condition clearly, we introduce a mapping function $f$ that transforms a Hermitian tight-binding dictionary into a real vector of parameters:

$$
 f : \rho_{mn}(R) \;\longmapsto\; f(\rho_{mn}(R)) \in \mathbb{R}^N.
$$

In the code, $f$ is represented by {autolink}`~meanfi.params.rparams.tb_to_rparams`, and its inverse is {autolink}`~meanfi.params.rparams.rparams_to_tb`. Currently, $f$ takes only the upper-triangular elements of the density matrix and separates them into real and imaginary parts, creating a real-valued vector.

#### Fixed-Point Formulation

A solution to the mean-field problem requires two conditions:

1. **Fixed point**: $\rho_{mn}(R)$ must be a fixed point of the mean-field iteration:

   $$
   f(\text{MF}(\rho_{mn}(R),\mu)) = f(\rho_{mn}(R)).
   $$

2. **Filling constraint**: The trace of $\rho_{mn}(0)$ equals the target filling $\nu$, i.e., $\mathrm{Tr}[\rho_{mn}(0)] = \nu$.

Since $\mu$ must also be adjusted to satisfy the filling constraint, one naive approach is to do a fixed-point iteration for $\rho_{mn}(R)$ at various values of $\mu$ until the desired $\nu$ is reached. However, such bisection-like methods can be slow.

## Proposed Approach for Solving

### 1. Augmented Parameter Vector

Rather than handling $\mu$ and $\rho_{mn}(R)$ separately, we can combine them. For clarity, let us denote

$$
\boldsymbol{\theta} = f(\rho_{mn}(R)) \in \mathbb{R}^N,
$$

the parameterized representation of the density matrix. We then define an augmented vector:

$$
\mathbf{\Xi} = \begin{pmatrix}\boldsymbol{\theta}\\ \mu \end{pmatrix} \in \mathbb{R}^{N+1}.
$$

### 2. Unified Objective Function

We define a function $\mathcal{F}: \mathbb{R}^{N+1} \to \mathbb{R}^{N+1}$ by:

$$
\mathcal{F}(\mathbf{\Xi}) = \begin{pmatrix}
  f\bigl(\text{MF}(f^{-1}(\boldsymbol{\theta}), \mu)\bigr) \; - \; \boldsymbol{\theta}\\
  \mathrm{Tr}\bigl[f^{-1}(\boldsymbol{\theta})(0)\bigr] \; - \; \nu
\end{pmatrix}.
$$

Here, $f^{-1}$ is the inverse of our parameterization. In the code, this function is defined by {autolink}`~meanfi.solvers.cost_density`.

### 3. Root-Finding as a Single Problem

Our goal is to solve:

$$
\mathcal{F}(\mathbf{\Xi}^*) = 0,
$$

which means both the fixed-point condition and the filling constraint are satisfied simultaneously. This is a single root-finding problem in $(N+1)$-dimensional space, avoiding the need for separate bisection in $\mu$.
One can use {autolink}`~scipy.optimize.root` methods for that.

### 4. Numerical Methods

To solve $\mathcal{F}(\mathbf{\Xi}) = 0$ efficiently, one can use methods such as Broyden’s method or Anderson mixing, which are quasi-Newton approaches. They require iteratively updating an approximate Jacobian or its inverse to find:

$$
\mathbf{\Xi}_{k+1} = \mathbf{\Xi}_k - \mathbf{G}_k \mathcal{F}(\mathbf{\Xi}_k),
$$

where $\mathbf{G}_k$ is an evolving approximation to the inverse Jacobian. Such methods are implemented in {autolink}`~meanfi.solvers.solver`.

### Chemical potential instability

The chemical potential $\mu$ can be unstable during the root-finding process.
This happens whenever chemical potential falls outside bandwidth, making a partial derivative of the density matrix with respect to $\mu$ zero.
Currently, a simple linear cost function is added to the objective function to ensure $\mu$ remains within the bandwidth.
