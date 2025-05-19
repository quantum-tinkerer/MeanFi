# %% Example
# We import all the required functions.
from meanfi.tb.utils import superc_tb
import qsymm
from meanfi.tb.transforms import tb_to_ham_fam
import numpy as np
from meanfi.model import Model
from meanfi.tb.utils import symm_guess_mf
from meanfi.solvers import solver_density_symmetric

tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])

# %% We build a Hamiltonian for our system.
cutoff = 1
ndim = 1
ndof = 1

h_0 = superc_tb(cutoff, ndim, ndof)
h_int = superc_tb(cutoff, ndim, ndof)
# h_0 =
# h_int =

# %% We set up the `meanfi` model.
charge_op = np.kron(tau_z, np.eye(ndof))
target_charge = 0
kT = 0

model = Model(h_0, h_int, target_charge, charge_op, kT)

# %% We build the bloch_family.
# Particle-hole symmetry
PHS = qsymm.particle_hole(ndim, np.kron(tau_x, np.eye(ndof)))
symmetries = [PHS]

# We recalculate the ndof, since we now have double it for the holes and electrons.
ndof = list(h_0.values())[0].shape[-1]

hams = (h_0, h_int)
hoppings = list(h_int.keys())

ham_fam = tb_to_ham_fam(hoppings, ndof, symmetries)

# %% We generate an initial guess with the ham_fam.
mf_guess = symm_guess_mf(ham_fam)

# %% We compute the self-consistent solution and observables.
ham_solution = solver_density_symmetric(model, ham_fam, mf_guess, nk=100)
print(ham_solution)

observable_1 = ...
