# %%
import kwant
import numpy as np

import meanfi
from meanfi.kwant_helper import utils
from meanfi.tb.transforms import tb_to_kfunc

s0 = np.identity(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1, -1])

# %%
# Step 1. Define the non-interacting tight-binding model.
# We will use the same graphene lattice here as an example that we make through kwant.

# Create graphene lattice
graphene = kwant.lattice.general(
    [(1, 0), (1 / 2, np.sqrt(3) / 2)], [(0, 0), (0, 1 / np.sqrt(3))], norbs=2
)
a, b = graphene.sublattices

# Create bulk system
bulk_graphene = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
# Set onsite energy to zero
bulk_graphene[a.shape((lambda pos: True), (0, 0))] = 0 * s0
bulk_graphene[b.shape((lambda pos: True), (0, 0))] = 0 * s0
# Add hoppings between sublattices
bulk_graphene[graphene.neighbors(1)] = s0
h_0 = utils.builder_to_tb(bulk_graphene)

# %%
# Step 2. Define the interaction tight-binding model. 
# For simplicity, we will only use the on-site interaction term here.
# We will also build this through kwant.

def onsite_int(site, U):
    return U * sx

builder_int = utils.build_interacting_syst(
    builder=bulk_graphene,
    lattice=graphene,
    func_onsite=onsite_int,
    func_hop=None,
    max_neighbor=0,
)
params = dict(U=2.5, V=0)
h_int = utils.builder_to_tb(builder_int, params)

# %%
# Step 3. Define the mean-field problem and set-up the model.
filling = 2 + 1e-2
model = meanfi.Model(h_0, h_int, filling=filling, atol=1e-4, kT=1e-2)

# %%
# Step 4. Make a guess for the mean-field solution and the chemical potential.

int_keys = frozenset(h_int)
ndof = len(list(h_0.values())[0])
mf_guess = meanfi.guess_tb(int_keys, ndof)

mu_guess = 0

# %% 
# Step 5. Solve for the mean-field solution.
mf_sol = meanfi.solver(model, mf_guess, mu_guess, debug=True)

# %% 
# Step 6. Analyze the solution
h_full = meanfi.add_tb(h_0, mf_sol)
hfunc = tb_to_kfunc(h_full)

Nk = 200
kx_vals = np.linspace(-np.pi, np.pi, Nk)
energies = []

for i, kx in enumerate(kx_vals):
    H = hfunc(np.array([kx, np.pi / 1.5]).reshape(1, -1))
    vals, vecs = np.linalg.eigh(H)
    energies.append(vals)
energies = np.array(energies)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(kx_vals, energies)
plt.xlabel(r"$k_x$")
plt.axhline(0, color="black", lw=0.5)
plt.ylabel("Energy (arbitrary units)")
plt.title("Graphene Band Structure (1D slice)")
plt.legend()
plt.tight_layout()
plt.show()
# %%
