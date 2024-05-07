# Algorithm overview

## The mean-field Hamiltonian

### Interacting problems

In physics, one often encounters problems where a system of multiple particles interact with each other.
In this package, we consider a general electronic system with density-density interparticle interaction:

:::{math}
:label: hamiltonian
\hat{H} = \hat{H^0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} c_i^\dagger c_j^\dagger c_j c_i
:::

where $c_i^\dagger$ and $c_i$ are creation and annihilation operators respectively for fermion in state $i$.
The first term $\hat{H^0}$ is the non-interacting Hamiltonian which by itself is straightforward to solve in a single-particle basis by direct diagonalizations made easy through packages such as `kwant`.
The second term $\hat{V}$ is density-density interaction term between two particles, for example Coulomb interaction.
In order to solve the interacting problem exactly, one needs to diagonalize the full Hamiltonian $\hat{H}$ in the many-particle basis which grows exponentially with the number of particles.
Such a task is often infeasible for large systems and one needs to resort to approximations.

### Mean-field approximaton

The first-order perturbative approximation to the interacting Hamiltonian is the Hartree-Fock approximation also known as the mean-field approximation.
The mean-field approximates the quartic term $\hat{V}$ in {eq}`hamiltonian` as a sum of bilinear terms weighted by the expectation values the remaining operators:
:::{math}
:label: mf_approx
\hat{V} \approx \hat{V}^{\text{MF}} \equiv \sum_{ij} v_{ij} \left[
\braket{c_i^\dagger c_i} c_j^\dagger c_j - \braket{c_i^\dagger c_j} c_j^\dagger c_i \right]
:::
we neglect the superconducting pairing and constant offset terms.
The expectation value terms  $\langle c_i^\dagger c_j \rangle$ are due to the ground-state density matrix and therefore act as an effective field acting on the system.
The ground-state density matrix reads:
:::{math}
:label: density
\rho_{ij} \equiv \braket{c_i^\dagger c_j } = \text{Tr}\left(e^{-\beta \left(\hat{H^0} + \hat{V}^{\text{MF}} - \mu \hat{N} \right)} c_i^\dagger c_j\right),
:::
where $\beta = 1/ (k_B T)$ is the inverse temperature, $\mu$ is the chemical potential, and $\hat{N} = \sum_i c_i^\dagger c_i$ is the number operator.

### Finite tight-binding grid

To simplify the mean-field Hamiltonian, we assume a finite, normalised orthogonal tight-binding grid defined by the single-particle basis states:

$$
\ket{n} = c^\dagger_n\ket{\text{vac}}
$$

where $\ket{\text{vac}}$ is the vacuum state.
We project our mean-field interaction in {eq}`mf_approx` onto the tight-binding grid:

:::{math}
:label: mf_finite
V^\text{MF}_{nm} = \braket{n | \hat{V}^{\text{MF}} | m} =  \sum_{i} \rho_{ii} v_{in} \delta_{nm} - \rho_{mn} v_{mn},
:::
where $\delta_{nm}$ is the Kronecker delta function.

### Infinite tight-binding grid

In the limit of a translationally invariant system, the index $n$ that labels the basis states partitions into two independent variables: the unit cell internal degrees of freedom (spin, orbital, sublattice, etc.) and the position of the unit cell $R_n$:

$$
n \to n, R_n.
$$

Because of the translationaly invariance, the physical properties of the system are independent of the absolute unit cell position $R_n$ and rather depend on the relative position between the two unit cells $R_{nm} = R_n - R_m$:

$$
\rho_{mn} \to \rho_{mn}(R_{mn}).
$$

That allows us to re-write the mean-field interaction in {eq}`mf_finite` as:

:::{math}
:label: mf_infinite
V^\text{MF}_{nm} (R) =  \sum_{i} \rho_{ii} (0) v_{in} (0) \delta_{nm} \delta(R) - \rho_{mn}(R) v_{mn}(R),
:::

where now indices $i, n, m$ label the internal degrees of freedom of the unit cell and $R$ is the relative position between the two unit cells in terms of the lattice vectors.

## Numerical implementation

### Self-consistency loop

In order to calculate the mean-field interaction in {eq}`mf_infinite`, we require the ground-state density matrix $\rho_{mn}(R)$.
However, the density matrix in {eq}`density` is a functional of the mean-field interaction $\hat{V}^{\text{MF}}$ itself.
Therefore, we need to solve for both self-consistently.

We define a single iteration of a self-consistency loop:

$$
\text{SCF}(\hat{V}^{\text{init, MF}}) \to \hat{V}^{\text{new, MF}},
$$

such that it performs the following operations given an initial mean-field interaction $\hat{V}^{\text{init, MF}}$:

1. Calculate the total Hamiltonian $\hat{H}(R) = \hat{H^0}(R) + \hat{V}^{\text{init, MF}}(R)$ in real-space.
2. Fourier transform the total Hamiltonian to the momentum space $\hat{H}(R) \to \hat{H}(k)$.
3. Calculate the ground-state density matrix $\rho_{mn}(k)$ in momentum space.
    1. Diagonalize the Hamiltonian $\hat{H}(k)$ to obtain the eigenvalues and eigenvectors.
    2. Calculate the fermi level $\mu$ given the desired filling of the unit cell.
    3. Calculate the density matrix $\rho_{mn}(k)$ using the eigenvectors and the fermi level $\mu$ (currently we do not consider thermal effects so $\beta \to \infty$).
4. Inverse Fourier transform the density matrix to real-space $\rho_{mn}(k) \to \rho_{mn}(R)$.
5. Calculate the new mean-field interaction $\hat{V}^{\text{new, MF}}(R)$ via {eq}`mf_infinite`.

### Self-consistency criterion

To define the self-consistency condition, we first introduce an invertible function $f$ that uniquely maps $\hat{V}^{\text{MF}}$ to a real-valued vector which minimally parameterizes it:

$$
f : \hat{V}^{\text{MF}} \to f(\hat{V}^{\text{MF}}) \in \mathbb{R}^N.
$$

Currently, $f$ parameterizes the mean-field interaction by taking only the upper triangular elements of the matrix $V^\text{MF}_{nm}(R)$ (the lower triangular part is redundant due to the Hermiticity of the Hamiltonian) and splitting it into a real and imaginary parts to form a real-valued vector.

With this function, we define the self-consistency criterion as a fixed-point problem:

$$
f(\text{SCF}(\hat{V}^{\text{MF}})) = f(\hat{V}^{\text{MF}}).
$$

To solve this fixed-point problem, we utilize a root-finding function `scipy.optimize.anderson` which uses the Anderson mixing method to find the fixed-point solution.
However, our implementation also allows to use other custom fixed-point solvers by either providing it to `solver` or by re-defining the `solver` function.
