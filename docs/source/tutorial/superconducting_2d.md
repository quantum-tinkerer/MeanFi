---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 2D Superconductor

This is an advanced tutorial that demonstrates how to set up and solve a constrained mean-field problem.
As an example, we consider a 2D model with onsite attractive interactions: a toy model for superconductivity.
Because superconductors are particle-hole symmetric, our goal is to constrain the mean-field solution to respect this symmetry.

For the model we consider the Hamiltonian of two fermions $c_{i \sigma}$ per site in a 2D square lattice:

\begin{equation}
H_0 = \sum_{\langle i, j \rangle, \sigma} c_{i \sigma}^\dagger c_{j \sigma} - \mu \sum_{i, \sigma} c_{i \sigma}^\dagger c_{i \sigma}
\end{equation}

where $\sigma = \uparrow, \downarrow$ denotes the spin degree of freedom, $t$ is the hopping amplitude between nearest neighbors, and $\mu$ is the chemical potential.
The attractive onsite interaction is given by:

\begin{equation}
H_{int} = U \sum_{i} n_{i \uparrow} n_{i \downarrow}
\end{equation}

with $U < 0$.

In the Bogoliubov-de Gennes (BdG) formalism, we rewrite the Hamiltonian in Nambu space by introducing the particle-hole spinor $\Psi_i = (c_{i \uparrow}, c_{i \downarrow}, c_{i \uparrow}^\dagger, c_{i \downarrow}^\dagger)^T$.
The non-interacting Hamiltonian in this basis reads:

\begin{equation}
H_0 = \sum_{\langle i, j \rangle} \Psi_i^\dagger \tau_z \Psi_j - \mu \sum_{i} \Psi_i^\dagger \tau_z \Psi_i
\end{equation}

where $\tau_z$ is the Pauli matrix acting in particle-hole space.
The interaction term can be decoupled in the mean-field approximation, leading to a superconducting order parameter $\Delta_i = U \langle c_{i \downarrow} c_{i \uparrow} \rangle$.
The mean-field Hamiltonian then becomes:

\begin{equation}
H_{MF} = H_0 + \sum_{i} \left( \Delta_i c_{i \uparrow}^\dagger c_{i \downarrow}^\dagger + \Delta_i^* c_{i \downarrow} c_{i \uparrow} \right)
\end{equation}

Our goal is to solve for the superconducting order parameter $\Delta_i$ self-consistently while ensuring that the mean-field solution respects particle-hole symmetry.

## Setup non-interacting Hamiltonian

We start by defining the non-interacting part of the Hamiltonian on a square lattice using Kwant.
This is convenient for building tight-binding models on various lattice geometries, and here we expemlify it on a square lattice with four degrees of freedom per site: two for spin and two for particle-hole space.

```{code-cell} ipython3
import kwant
import numpy as np
import matplotlib.pyplot as plt

# Create square lattice
nph = nspin = 2  # particle-hole and spin degrees of freedom
square_lattice = kwant.lattice.square(norbs=nph * nspin)

def square_shape(pos):
    x, y = pos
    return True

# Build Kwant system with translational symmetry
syst = kwant.Builder(kwant.TranslationalSymmetry(*square_lattice.prim_vecs))

# Pauli matrices in particle-hole space
tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)

# Onsite terms
syst[square_lattice.shape(square_shape, (0, 0))] = 0 * np.kron(tau_z, np.eye(nspin))

# Hopping terms
syst[square_lattice.neighbors(1)] = np.kron(tau_z, np.eye(nspin))
```

For simplicity, we set the chemical potential $\mu = 0$ and the hopping amplitude $t = 1$.

At this point, we may visualize the unit cell:

```{code-cell} ipython3
kwant.plot(syst)
```

and confirm that the dispersion relation of $H_0$ is $E_0(\mathbf{k}) = 2 (\cos k_x + \cos k_y)$ by using `kwant.wraparound`:

```{code-cell} ipython3
wrapped_syst = kwant.wraparound.wraparound(syst).finalized()
hk = lambda k_x, k_y: wrapped_syst.hamiltonian_submatrix(
    params={"k_x": k_x, "k_y": k_y}
)

ks = np.linspace(0, 2 * np.pi, 20, endpoint=True)
hams = np.array([[hk(kx, ky) for ky in ks] for kx in ks])

evals_h0 = np.linalg.eigvalsh(hams)

cmap = plt.get_cmap("twilight")
norms = plt.Normalize(vmin=0, vmax=2 * np.pi)
fig, ax = plt.subplots()
for i, ky in enumerate(ks):
    ax.plot(ks, evals_h0[:, i, :], c=cmap(norms(ky)), linewidth=0.5)
cbar_ax = fig.add_axes([-0.03, 0.25, 0.015, 0.55])
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norms, cmap=cmap), cax=cbar_ax)
cb1.set_label(r"$k_y / a$")
ax.set_xticks([0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2\\pi$"])
ax.set_xlim(0, 2 * np.pi)
ax.set_ylabel("$E$")
ax.set_xlabel("$k_x / a$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

At half-filling, the Fermi surface is given by:

```{code-cell} ipython3
fermi_surface = np.min(np.abs(evals_h0), axis=2)

plt.figure()
plt.contourf(*np.meshgrid(ks, ks), fermi_surface, levels=1, cmap="Blues", alpha=0.5)
plt.colorbar()
plt.title("Fermi Surface")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.show()
```

To proceed with the mean-field calculation, we convert the Kwant builder into a MeanFi tight-binding model:

```{code-cell} ipython3
from meanfi.kwant_helper import utils

h_0 = utils.builder_to_tb(syst)
```

## Define the interacting Hamiltonian

The next step is to define the interacting part of the Hamiltonian: an onsite attractive interaction between the fermionic degrees of freedom.
In the BdG formalism, this corresponds to an attractive interaction between electrons and holes, but a repulsive interaction between electrons and electrons, holes and holes: $U (\tau_0 - \tau_x) \otimes \mathbb{1}_{spin}$.

```{code-cell} ipython3
# Define onsite interaction term
def onsite_int(site, U):
    return U * np.kron((tau_0 - tau_x), np.ones((nspin, nspin)))


builder_int = utils.build_interacting_syst(
    builder=syst, lattice=square_lattice, func_onsite=onsite_int, max_neighbor=1
)
params = {"U": -2}
h_int = utils.builder_to_tb(builder_int, params)
```

## Symmetry-constrained mean-field solution

Because we search for a superconducting solution, meanfi's default mean-field solver cannot be used directly, as it does not guarantee that the solution respects particle-hole symmetry.
To enforce this symmetry, we construct a basis of Hamiltonian terms that respect particle-hole symmetry and use it to parametrize the mean-field solution.

```{code-cell} ipython3
nsites = len(wrapped_syst.sites) # number of sites in the unit cell, here 1

# Hamiltonian basis using kwant's ordering of degrees of freedom (site, ph, spin)
ham_terms = np.array(
    [np.kron(np.diag(delta), np.kron(tau_x, np.eye(nspin))) for delta in np.eye(nsites)]
)
```

To ensure that the basis is orthonormal, we perform a QR decomposition:

```{code-cell} ipython3
def hamiltonian_basis(ham_terms):
    ham_vectors = np.array([matrix.flatten() for matrix in ham_terms])
    ham_vectors = np.linalg.qr(ham_vectors.T, mode="reduced")[0].T

    overlap = ham_vectors @ ham_vectors.conj().T
    np.testing.assert_allclose(overlap, np.eye(overlap.shape[0]), atol=1e-8)
    ham_terms = ham_vectors.reshape(*ham_terms.shape)
    return ham_vectors.reshape(*ham_terms.shape)


ham_basis = hamiltonian_basis(ham_terms)
```

### Mean-field guess

Before starting the solver, we need to provide an initial guess for the mean-field solution.
We construct a random guess in the Hamiltonian basis defined above:

```{code-cell} ipython3
# Construct mean-field guess
scale = 1
random_coeffs = scale * np.random.rand(len(ham_basis))
mf_guess = {(0, 0): np.tensordot(random_coeffs, ham_basis, 1)}
```

### Define solver

We define a function to compute the mean-field solution constrained to the Hamiltonian basis defined above.

```{code-cell} ipython3
from functools import partial

from scipy.optimize import anderson
from meanfi.mf import density_matrix, meanfield
from meanfi.model import Model
from meanfi.tb.tb import add_tb

# Compute mean field solution
# This codes assumes only onsite interactions, hence the explicit (0,0) key.
# This code assumes a superconducting order parameter as in ham_basis,
# hence the use of only the off-diagonal block of the density matrix.

def permutate_sites(operator):
    reshaped = operator.reshape(nsites, nph, nspin, nsites, nph, nspin)
    permuted = reshaped.transpose(1, 2, 0, 4, 5, 3)
    return permuted.reshape(*operator.shape)

def cost_density_symmetric(rho_params, model, ham_basis, nk):
    rho = {(0, 0): np.tensordot(rho_params, ham_basis, 1)}
    rho_new = model.density_matrix(rho, nk=nk)[(0, 0)]
    permuted_rho = permutate_sites(rho_new)
    block_rho = permuted_rho[
        : len(permuted_rho) // 4 :, len(permuted_rho) // 2 : 3 * len(permuted_rho) // 4
    ]
    rho_params_new = -2 * np.diag(block_rho)
    # rho_params_new = np.einsum('ij, kji -> k', rho_new, ham_basis)
    return rho_params_new - rho_params


def compute_sol(
    h_0,
    h_int,
    nk,
    ham_basis,
    mf_guess,
    filling,
    optimizer_kwargs
):
    model = Model(h_0, h_int, filling)

    rho_guess = density_matrix(add_tb(model.h_0, mf_guess), filling, nk)[0][(0, 0)]

    # permute rho to order (particle-hole, spin, site) to extract coefficients and avoid einsum
    permuted_rho = permutate_sites(rho_guess)
    block_rho = permuted_rho[
        : len(permuted_rho) // 4 :, len(permuted_rho) // 2 : 3 * len(permuted_rho) // 4
    ]
    rho_params = -2 * np.diag(block_rho)
    # rho_params = np.einsum('ij, kji -> k', rho_guess, ham_basis)
    f = partial(cost_density_symmetric, model=model, ham_basis=ham_basis, nk=nk)
    rho_params = anderson(f, rho_params, **optimizer_kwargs)
    rho_result = {(0, 0): np.tensordot(rho_params, ham_basis, 1)}

    mf_result = meanfield(rho_result, model.h_int)
    return mf_result
```

### Get results

Finally, we compute the mean-field solution constrained to the particle-hole symmetric Hamiltonian basis over a $20 \times 20$ k-point grid:

```{code-cell} ipython3
nk = 20
h_int_solution = compute_sol(
    h_0,
    h_int,
    nk=nk,
    ham_basis=ham_basis,
    mf_guess=mf_guess,
    filling=2,
    optimizer_kwargs={"verbose": True}
)
h_mf = add_tb(h_0, h_int_solution)
```

We can now visualize the mean-field band structure and confirm that a superconducting gap has opened at the Fermi surface:

```{code-cell} ipython3
from meanfi.tb.transforms import tb_to_kgrid

fig, ax = plt.subplots()
ks = np.linspace(0, 2 * np.pi, nk, endpoint=True)
hamiltonians = tb_to_kgrid(h_mf, nk)

evals_hmf = np.linalg.eigvalsh(hamiltonians)

# divergent colormap
cmap = plt.get_cmap("twilight")
norms = plt.Normalize(vmin=0, vmax=2 * np.pi)
for i, ky in enumerate(ks):
    ax.plot(ks, evals_hmf[:, i, :], c=cmap(norms(ky)), linewidth=0.5)
ax.axhline(color="k", linestyle="--", linewidth=1)
ax.set_xticks([0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2\\pi$"])
ax.set_xlim(0, 2 * np.pi)
ax.set_ylabel("$E - E_F$")
ax.set_xlabel("$k_x / a$")
ax.annotate(
    "a)",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(-0.1, 1),
    textcoords="axes fraction",
    fontweight="bold",
    va="top",
    ha="right",
)
cbar_ax = fig.add_axes([-0.03, 0.25, 0.015, 0.55])
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norms, cmap=cmap), cax=cbar_ax)
cb1.set_label(r"$k_y / a$")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

The superconducting gap is:

```{code-cell} ipython3
gap = np.min(np.abs(evals_hmf))
gap
```
