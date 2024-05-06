# Algorithm overview

## Interacting problems

In physics, one often encounters problems where a system of multiple particles interact with each other.
In this package, we consider a general electronic system with density-density interparticle interaction:

:::{math}
:label: hamiltonian
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} c_i^\dagger c_j^\dagger c_j c_i
:::

where $c_i^\dagger$ and $c_i$ are creation and annihilation operators respectively for fermion in state $i$.
The first term $\hat{H}_0$ is the non-interacting Hamiltonian which by itself is straightforward to solve in a single-particle basis by direct diagonalizations made easy through packages such as `kwant`.
The second term $\hat{V}$ is density-density interaction term between two particles, for example Coulomb interaction.
In order to solve the interacting problem exactly, one needs to diagonalize the full Hamiltonian $\hat{H}$ in the many-particle basis which grows exponentially with the number of particles.
Such a task is often infeasible for large systems and one needs to resort to approximations.

## Mean-field Hamiltonian

The first-order perturbative approximation to the interacting Hamiltonian is the Hartree-Fock approximation also known as the mean-field approximation.
The mean-field approximates the quartic term $\hat{V}$ in {eq}`hamiltonian` as a sum of bilinear terms weighted by the expectation values the remaining operators:

$$
\hat{V} \approx \hat{V}^{\text{MF}} \equiv \sum_{ij} v_{ij} \left[
\braket{c_i^\dagger c_i} c_j^\dagger c_j - \braket{c_i^\dagger c_j} c_j^\dagger c_i \right]
$$

we neglect the superconducting pairing and constant offset terms.
The expectation value terms  $\langle c_i^\dagger c_j \rangle$ are due to the ground-state density matrix:

$$
\rho_{ij} \equiv \langle c_i^\dagger c_j \rangle,
$$

and therefore act as an effective field acting on the system.

<!-- :::{admonition} Derivation of the mean-field Hamiltonian with Wicks theorem
:class: dropdown info
```{include} mf_details.md
```
::: -->

### Tight-binding grid

To simplify the mean-field Hamiltonian, we assume a normalised orthogonal tight-binding grid defined by the single-particle basis states:

$$
\ket{n} = c^\dagger_n\ket{\text{vac}}
$$

where $\ket{\text{vac}}$ is the vacuum state.
We project our mean-field interaction onto the tight-binding grid:

$$
V^\text{MF}_{nm} = \langle n | \hat{V}^{\text{MF}} | m \rangle =  \sum_{i} \rho_{ii} v_{in} \delta_{nm} - \rho_{mn} v_{mn},
$$
where $\delta_{nm}$ is the Kronecker delta function.
