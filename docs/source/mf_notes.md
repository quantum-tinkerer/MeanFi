# Algorithm overview

## Interacting problems

In physics, one often encounters problems where a system of multiple particles interact with each other.
By using the second quantization notation, a general Hamiltonian of such system reads:

:::{math}
:label: hamiltonian
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ijkl} v_{ijkl} c_i^\dagger c_j^\dagger c_l c_k
:::

where $c_i^\dagger$ and $c_i$ are creation and annihilation operators respectively for fermion in state $i$.
The first term $\hat{H}_0$ is the non-interacting Hamiltonian which by itself is straightforward to solve in a single-particle basis by direct diagonalizations made easy through packages such as `kwant`.
The second term $\hat{V}$ is the interaction term between two particles.
In order to solve the interacting problem exactly, one needs to diagonalize the full Hamiltonian $\hat{H}$ in the many-particle basis which grows exponentially with the number of particles.
Such a task is often infeasible for large systems and one often needs to resort to approximations.

## Mean-field approximation

In many interacting systems, there exist constant order parameters $\langle A \rangle$ that describe the phase of the system.
Here we define $\hat{A}$ as some operator and $\langle \rangle$ denotes the expectation value with respect to the ground state of the system.
Famous examples of such order parameter is the magnetization in a ferromagnet and the superconducting order parameter in a superconductor.
If we are interested in properties of the system close to the ground state, we can re-write the operator $\hat{A}$ around the order parameter:

$$
\hat{A} = \langle A \rangle + \delta \hat{A},
$$

where operator $\delta \hat{A}$ describes the fluctuations around the order parameter.
Let us consider an additional operator $\hat{B}$ and say we are interested in the product of the two operators $\hat{A}\hat{B}$.
If we assume that the fluctuations $\delta$ are small, we can approximate the product of operators into a sum of single operators and the product of the expectation values:

$$
\hat{A}\hat{B} \approx \langle A \rangle \hat{B} + \hat{A} \langle B \rangle - \langle A \rangle \langle B \rangle
$$

where we neglect $\delta^2$ terms.
This approximation is known as the mean-field approximation.

## Mean-field Hamiltonian

We apply the mean-field approximation to the quartic interaction term in {eq}`hamiltonian`:

$$
V \approx \frac12 \sum_{ijkl} v_{ijkl} \left[
\langle c_i^\dagger c_k \rangle c_j^\dagger c_l - \langle c_i^\dagger c_l \rangle c_j^\dagger c_k - \langle c_j^\dagger c_k \rangle c_i^\dagger c_l + \langle c_j^\dagger c_l \rangle c_i^\dagger c_k \right]
$$

where we make use of Wicks theorem to simplify the expression and neglect the superconducting pairing and constant offset terms.

:::{admonition} Derivation of the mean-field Hamiltonian with Wicks theorem
:class: dropdown info
```{include} mf_details.md
```
:::

The expectation value terms $\langle c_i^\dagger c_j \rangle$ are the density matrix elements of the ground state of the system:
$$
\rho_{ij} = \langle c_i^\dagger c_j \rangle.
$$

### Tight-binding grid

We project $\hat{V}$ onto a tight-binding grid:

$$
V_{nm} = \langle n | \hat{V} | m \rangle = \\
\frac12 \left[ \sum_{ik} v_{inkm} F_{ik} - \sum_{jk} v_{njkm} F_{jk} - \sum_{il} v_{inml} F_{il} + \sum_{jl} v_{njml} F_{jl} \right] = \\
-\sum_{ij} F_{ij} \left(v_{inmj} - v_{injm} \right)
$$

where I used the $v_{ijkl} = v_{jilk}$ symmetry from Coulomb.

For density-density interactions (like Coulomb repulsion) the interaction tensor reads:

$$
v_{ijkl} = v_{ij} \delta_{ik} \delta_{jl},
$$

which simplifies the interaction to:

$$
V_{nm} = - \sum_{ij} F_{ij} \left(v_{inmj} - v_{injm} \right) = \\
-\sum_{ij}F_{ij} v_{in} \delta_{im} \delta_{nj} + \sum_{ij}F_{ij} v_{in} \delta_{ij} \delta_{nm} = \\
-F_{mn} v_{mn} + \sum_{i} F_{ii} v_{in} \delta_{nm}
$$

the first term is the exchange interaction whereas the second one is the direct interaction.
