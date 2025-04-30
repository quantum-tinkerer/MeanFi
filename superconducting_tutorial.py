# %%
from meanfi.tb.utils import superc_tb
import qsymm
from meanfi.tb.transforms import tb_to_ham_fam
import numpy as np
from meanfi.model import Model
from meanfi.tb.utils import symm_guess_mf
from meanfi.solvers import solver_density_symmetric

cutoff = 1
ndim = 1
ndof = 1

h_0 = superc_tb(cutoff, ndim, ndof)
h_int = superc_tb(cutoff, ndim, ndof)
tau_z = np.array([[1, 0], [0, -1]])
Q = np.kron(tau_z, np.eye(ndof))
target_Q = 0
kT = 1e-2
tau_x = np.kron(np.array([[0, 1], [1, 0]]), np.eye(ndof))

PHS = qsymm.particle_hole(ndim, tau_x)

symmetries = [PHS]

model = Model(h_0, h_int, Q, target_Q, kT)
hams = (h_0, h_int)

ndof = 2  # This is different because we generated the superconducting tb with ndof 1 per particle. (but there are two particles)
hoppings = list(
    h_int.keys()
)  # I believe this should still contain all hoppings, including h_0
ham_fam = tb_to_ham_fam(hoppings, ndof, symmetries)
mf_guess = symm_guess_mf(
    ham_fam
)  # This function causes it to orthogonalize twice now, but since there are other ways of creating the guess, I think it's fine.

print(h_0)
print(h_int)

print(solver_density_symmetric(model, ham_fam, mf_guess, nk=50))
