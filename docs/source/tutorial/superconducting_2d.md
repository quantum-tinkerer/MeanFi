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

# 2D Superconducting System

This tutorial demonstrates how to set up and solve a 2D model with onsite attractive interactions using mean-field theory.

The Hamiltonian is defined on a square lattice with two spins per site and takes the form:

\begin{equation}
H_0 = \sum_{\langle i, j \rangle, \sigma} c_{i \sigma}^\dagger c_{j \sigma} - \mu \sum_{i, \sigma} c_{i \sigma}^\dagger c_{i \sigma}
\end{equation}

The interaction term is given by:

\begin{equation}
H_{int} = U \sum_{i} n_{i \uparrow} n_{i \downarrow}
\end{equation}

...

BdG formalism...

## Setup non-interacting Hamiltonian

```{code-cell} ipython3
:tags: [hide-input]

from functools import partial

import kwant
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import anderson

from meanfi.tb.transforms import tb_to_kgrid
from meanfi.model import Model
from meanfi.tb.tb import add_tb
from meanfi.kwant_helper import utils
from meanfi.mf import density_matrix, meanfield
```

We start by defining the non-interacting part of the Hamiltonian on a square lattice using Kwant.
This is convenient for building tight-binding models on various lattice geometries, and here we expemlify it on a square lattice with four degrees of freedom per site: two for spin and two for particle-hole space.

```{code-cell} ipython3
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
We confirm the band structure of the non-interacting Hamiltonian by computing the eigenvalues over a grid in momentum space using `kwant.wraparound`.

```{code-cell} ipython3
wrapped_syst = kwant.wraparound.wraparound(syst).finalized()
ham_func = lambda k_x, k_y: wrapped_syst.hamiltonian_submatrix(
    params={"k_x": k_x, "k_y": k_y}
)

ks = np.linspace(0, 2 * np.pi, 20, endpoint=True)
hams = np.array([[ham_func(kx, ky) for ky in ks] for kx in ks])

evals_h_0 = np.linalg.eigvalsh(hams)

cmap = plt.get_cmap("twilight")
norms = plt.Normalize(vmin=0, vmax=2 * np.pi)
for i, ky in enumerate(ks):
    plt.plot(ks, evals_h_0[:, i, :], c=cmap(norms(ky)), linewidth=0.5)
```

At half-filling, the Fermi surface is given by:

```{code-cell} ipython3
fermi_surface = np.min(np.abs(evals_h_0), axis=2)

plt.figure()
plt.contourf(*np.meshgrid(ks, ks), fermi_surface, levels=1, cmap="Blues", alpha=0.5)
plt.colorbar()
plt.title("Fermi Surface")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.show()
```

## Define mean-field problem

Next, we define the mean-field problem by converting the non-interacting Hamiltonian to a meanfi model and we define the onsite attractive interaction.

```{code-cell} ipython3
# Convert kwant builder to meanfi tight-binding model
h_0 = utils.builder_to_tb(syst)


# Define onsite interaction term
# Attractive interaction between electrons and holes
# Repulsive interaction between electrons and electrons, holes and holes
def onsite_int(site, U):
    return U * np.kron((tau_0 - tau_x), np.ones((nspin, nspin)))


builder_int = utils.build_interacting_syst(
    builder=syst, lattice=square_lattice, func_onsite=onsite_int, max_neighbor=1
)
params = {"U": -2}
h_int = utils.builder_to_tb(builder_int, params)
```

## Mean-field solution

```{code-cell} ipython3
nsites = len(wrapped_syst.sites)

# Construct Hamiltonian basis for mean-field decomposition in kwant's basis
ham_terms = np.array(
    [np.kron(np.diag(delta), np.kron(tau_x, np.eye(nspin))) for delta in np.eye(nsites)]
)


def hamiltonian_basis(ham_terms):
    ham_vectors = np.array([matrix.flatten() for matrix in ham_terms])
    ham_vectors = np.linalg.qr(ham_vectors.T, mode="reduced")[0].T

    overlap = ham_vectors @ ham_vectors.conj().T
    np.testing.assert_allclose(overlap, np.eye(overlap.shape[0]), atol=1e-8)
    ham_terms = ham_vectors.reshape(*ham_terms.shape)
    return ham_vectors.reshape(*ham_terms.shape)


ham_basis = hamiltonian_basis(ham_terms)

def permutate_sites(operator):
    reshaped = operator.reshape(nsites, nph, nspin, nsites, nph, nspin)
    permuted = reshaped.transpose(1, 2, 0, 4, 5, 3)
    return permuted.reshape(*operator.shape)

```

```{code-cell} ipython3
# Construct mean-field guess
scale = 1
random_coeffs = scale * np.random.rand(len(ham_basis))
mf_guess = {(0, 0): np.tensordot(random_coeffs, ham_basis, 1)}
charge_op = np.kron(tau_z, np.eye(nspin))
```


```{code-cell} ipython3
# Compute mean field solution
# This codes assumes only onsite interactions, hence the explicit (0,0) key.
# This code assumes a superconducting order parameter as in ham_basis,
# hence the use of only the off-diagonal block of the density matrix.


def cost_density_symmetric(rho_params, model, ham_basis, nk):
    rho = {(0, 0): np.tensordot(rho_params, ham_basis, 1)}
    rho_new = model.density_matrix_iteration(rho, nk=nk)[(0, 0)]
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
    target_charge,
    kT,
    optimizer_kwargs
):
    model = Model(h_0, h_int, target_charge, charge_op, kT)

    rho_guess = density_matrix(
        add_tb(model.h_0, mf_guess), model.charge_op, model.target_charge, model.kT, nk
    )[0][(0, 0)]

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


def compute_gap(full_sol, nk_dense, fermi_energy=0):
    h_kgrid = tb_to_kgrid(full_sol, nk_dense)
    vals = np.linalg.eigvalsh(h_kgrid)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)
```

### Plot results

```{code-cell} ipython3
nk = 20
target_charge = 0
kT = 0

optimizer_kwargs = {"verbose": True}
h_int_solution = compute_sol(
    h_0,
    h_int,
    nk,
    ham_basis,
    mf_guess,
    target_charge,
    kT,
    optimizer_kwargs
)
h_mf = add_tb(h_0, h_int_solution)
```

```{code-cell} ipython3
n = 20
max_temp = 0.2
temperatures = np.linspace(0, max_temp, n)
gaps = np.zeros_like(temperatures)
nk_dense = 10
for i in range(n):
    if i == 0:
        h_mf_kT = h_mf
    h_mf_kT = add_tb(
        h_0,
        compute_sol(
            h_0,
            h_int,
            nk,
            ham_basis,
            h_mf_kT,
            target_charge,
            temperatures[i],
            optimizer_kwargs,
        ),
    )
    gaps[i] = compute_gap(h_mf_kT, nk_dense)

```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2)
ks = np.linspace(0, 2 * np.pi, nk, endpoint=True)
hamiltonians = tb_to_kgrid(h_mf, nk)

vals = np.linalg.eigvalsh(hamiltonians)

# divergent colormap
cmap = plt.get_cmap("twilight")
norms = plt.Normalize(vmin=0, vmax=2 * np.pi)
for i, ky in enumerate(ks):
    # color each momentum slice differently in shades of gray
    ax[0].plot(ks, vals[:, i, :], c=cmap(norms(ky)), linewidth=0.5)
ax[0].axhline(color="k", linestyle="--", linewidth=1)
ax[0].set_xticks([0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2\\pi$"])
ax[0].set_xlim(0, 2 * np.pi)
ax[0].set_ylabel("$E - E_F$")
ax[0].set_xlabel("$k_x / a$")
ax[0].annotate(
    "a)",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(-0.1, 1),
    textcoords="axes fraction",
    fontweight="bold",
    va="top",
    ha="right",
)
# plot colorbar for momenta using a separate axis
cbar_ax = fig.add_axes([-0.03, 0.25, 0.015, 0.55])
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norms, cmap=cmap), cax=cbar_ax)
cb1.set_label(r"$k_y / a$")


def gap_over_temp(T, Tc, gap_0):
    gap = gap_0 * np.tanh(1.74 * np.sqrt(np.maximum(Tc / T - 1, 0)))
    return gap


theory_gaps = gap_over_temp(temperatures, 0.195, gaps[0])

ax[1].margins(x=0)
ax[1].plot(temperatures, theory_gaps, linestyle=":", linewidth=2, label="Theoretical")

ax[1].plot(temperatures, gaps, label="Calculated")
ax[1].legend().set_zorder(101)
ax[1].set_xlabel(r"$k_B T$")
ax[1].set_ylabel("Gap")
ax[1].annotate(
    "b)",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(-0.1, 1),
    textcoords="axes fraction",
    fontweight="bold",
    va="top",
    ha="right",
)

for ax_i in ax:
    ax_i.spines["top"].set_visible(False)
    ax_i.spines["right"].set_visible(False)
```

```{code-cell} ipython3
plt.plot(ks, vals[:, 15, :])
```

```{code-cell} ipython3
gap = np.min(np.abs(vals))
gap
```
