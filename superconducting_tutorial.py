# %% Example
# We import all the required functions.
import qsymm
from meanfi.tb.transforms import tb_to_ham_fam
import numpy as np
from meanfi.model import Model
from meanfi.tb.utils import symm_guess_mf
from meanfi.solvers import solver_density_symmetric

tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])

# %% We build a Hamiltonian for our system.
# We build it using blocks of h_0 and Delta, or h_int and Delta.
ndof = 2
ndim = 1

delta = 0

h_sc_0 = {
    (0,): delta * np.kron(tau_x, np.eye(ndof)),
    (1,): np.kron(tau_z, np.eye(ndof)),
    (-1,): np.kron(tau_z, np.eye(ndof)),
}

h_sc_int = {
    (0,): delta * np.kron(tau_x, np.eye(ndof)) + np.kron(np.eye(2), np.eye(ndof))
}

# We recalculate the ndof, since we now have double it for the holes and electrons.
ndof_sc = ndof * 2

# %% We set up the `meanfi` model.
charge_op = np.kron(tau_z, np.eye(ndof))
target_charge = 0
kT = 0

model = Model(h_sc_0, h_sc_int, target_charge, charge_op, kT)

# %% We build the bloch_family.
# Particle-hole symmetry
PHS = qsymm.particle_hole(ndim, np.kron(tau_x, np.eye(ndof)))
symmetries = [PHS]

hams = (h_sc_0, h_sc_int)
# Probably something going wrong here
hoppings = list(h_sc_int.keys())  # list(set().union(*hams))

ham_fam = tb_to_ham_fam(hoppings, ndof_sc, symmetries)

# %% We generate an initial guess with the ham_fam.
mf_guess = symm_guess_mf(ham_fam)

# %% We compute the self-consistent solution and observables.
ham_solution = solver_density_symmetric(model, ham_fam, mf_guess, nk=100)
print(ham_solution)

observable_1 = ...

# Look at onsite hamiltonian first.
# Are the blocks connecting the electrons and holes full?
# Use imshow to visualize the meanfield hamiltonian.
# Look at gap.

# %%
