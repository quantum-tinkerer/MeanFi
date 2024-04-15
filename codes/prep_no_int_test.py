# %%
import numpy as np
from codes.solvers import solver
from codes.tb import utils
from codes.tb.tb import compare_dicts, add_tb
from codes.model import Model

# %%
"""
Run a calculation without interactions and check if: 1) the solution is the non-interacting Hamiltonian, and (2) it takes no time for the solver to find that.
"""
# %%
cutoff = 1  # These should all be random in the test
dim = 2
ndof = 4
filling = 2
random_hopping_vecs = utils.generate_vectors(cutoff, dim)

# %%
h_0_random = utils.generate_guess(random_hopping_vecs, ndof, scale=1)

# %%
h_int_zeros = {}
vectors = random_hopping_vecs
for vector in vectors:
    if vector not in h_int_zeros.keys():
        rand_hermitian = np.zeros((ndof, ndof), dtype=complex)
        if np.linalg.norm(np.array(vector)) == 0:
            rand_hermitian += rand_hermitian.T.conj()
            rand_hermitian /= 2
            h_int_zeros[vector] = rand_hermitian
        else:
            h_int_zeros[vector] = rand_hermitian
            h_int_zeros[tuple(-np.array(vector))] = rand_hermitian.T.conj()

h_int_only_phases = utils.generate_guess(random_hopping_vecs, ndof, scale=0)

# %%
guess = utils.generate_guess(random_hopping_vecs, ndof, scale=1)
model = Model(h_0_random, h_int_only_phases, filling=filling)
mf_sol = solver(model, guess, nk=20)

# %%
print(f"Onsite too large: {np.max(mf_sol[0, 0])}")
N = len(mf_sol.keys()) // 2
sorted_vals = np.array(list(mf_sol.values()))[
    np.lexsort(np.array(list(mf_sol.keys())).T)
]
print(f"Largest value in hoppings: {np.max(sorted_vals[:N].flatten())}")

# %%
assert compare_dicts(add_tb(mf_sol, h_0_random), h_0_random)
