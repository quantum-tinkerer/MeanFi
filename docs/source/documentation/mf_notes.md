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
# Theory overview

## Interacting problems

In physics, one often encounters problems where a system of multiple particles interacts with each other.
In this package, we consider a general electronic system with density-density interparticle interaction:

:::{math}
:label: hamiltonian
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} c_i^\dagger c_j^\dagger c_j c_i
:::

where $c_i^\dagger$ and $c_i$ are the creation and annihilation operators respectively for fermion in state $i$.
The first term $\hat{H_0}$ is the non-interacting Hamiltonian which by itself is straightforward to solve on a single-particle basis by direct diagonalizations made easy through packages such as [kwant](https://kwant-project.org/).
The second term $\hat{V}$ is the density-density interaction term between two particles, for example Coulomb interaction.
To solve the interacting problem exactly, one needs to diagonalize the full Hamiltonian $\hat{H}$ in the many-particle basis which grows exponentially with the number of particles.
Such a task is often infeasible for large systems and one needs to resort to approximations.

## Mean-field approximation

The first-order perturbative approximation to the interacting Hamiltonian is the Hartree-Fock approximation also known as the mean-field approximation.
The mean field approximates the quartic term $\hat{V}$ in {eq}`hamiltonian` as a sum of bilinear terms weighted by the expectation values of the remaining operators:
:::{math}
:label: mf_approx
\hat{V} \approx \hat{V}_{\text{MF}} \equiv \sum_{ij} v_{ij} \left[
\braket{c_i^\dagger c_i} c_j^\dagger c_j - \braket{c_i^\dagger c_j} c_j^\dagger c_i \right],
:::
where we neglect the constant offset terms and the superconducting pairing (for now).
The expectation value terms  $\langle c_i^\dagger c_j \rangle$ are due to the ground state density matrix and act as an effective field on the system.
The ground state density matrix reads:
:::{math}
:label: density
\rho_{ij} \equiv \braket{c_i^\dagger c_j } = \text{Tr}\left(e^{-\beta \left(\hat{H_0} + \hat{V}_{\text{MF}} - \mu \hat{N} \right)} c_i^\dagger c_j\right),
:::
where $\beta = 1/ (k_B T)$ is the inverse temperature, $\mu$ is the chemical potential, and $\hat{N} = \sum_i c_i^\dagger c_i$ is the number operator.
Currently, we neglect thermal effects so $\beta \to \infty$.

## Finite tight-binding grid

To simplify the mean-field Hamiltonian, we assume a finite, normalised, orthogonal tight-binding grid defined by the single-particle basis states:

$$
\ket{n} = c^\dagger_n\ket{\text{vac}}
$$

where $\ket{\text{vac}}$ is the vacuum state.
We project our mean-field interaction in {eq}`mf_approx` onto the tight-binding grid:

:::{math}
:label: mf_finite
V_{\text{MF}, nm} = \braket{n | \hat{V}_{\text{MF}} | m} =  \sum_{i} \rho_{ii} v_{in} \delta_{nm} - \rho_{mn} v_{mn},
:::
where $\delta_{nm}$ is the Kronecker delta function.

## Infinite tight-binding grid

In the limit of a translationally invariant system, the index $n$ that labels the basis states partitions into two independent variables: the unit cell internal degrees of freedom (spin, orbital, sublattice, etc.) and the position of the unit cell $R_n$:

$$
n \to n, R_n.
$$

Because of the translational invariance, the physical properties of the system are independent of the absolute unit cell position $R_n$ but rather depend on the relative position between the two unit cells $R_{nm} = R_n - R_m$:

$$
\rho_{mn} \to \rho_{mn}(R_{mn}).
$$

That allows us to re-write the mean-field interaction in {eq}`mf_finite` as:

:::{math}
:label: mf_infinite
V_{\text{MF}, nm} (R) =  \sum_{i} \rho_{ii} (0) v_{in} (0) \delta_{nm} \delta(R) - \rho_{mn}(R) v_{mn}(R),
:::

where now indices $i, n, m$ label the internal degrees of freedom of the unit cell and $R$ is the relative position between the two unit cells in terms of the lattice vectors.
