# %% Example
# We import all the required functions.
import qsymm
from meanfi.tb.transforms import tb_to_ham_fam, tb_to_kgrid
import numpy as np
from meanfi.model import Model
from meanfi.tb.utils import symm_guess_mf
from meanfi.tb.tb import add_tb
from meanfi.solvers import solver_density_symmetric
import matplotlib.pyplot as plt

tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)

# %% We build a Hamiltonian for our system.
# We build it using blocks of h_0 and Delta, or h_int and Delta.
ndof = 2
ndim = 1
U = -2

h_sc_0 = {(1,): np.kron(tau_z, np.eye(ndof)), (-1,): np.kron(tau_z, np.eye(ndof))}
h_sc_int = {
    (0,): U
    * (np.kron(tau_0, np.ones((ndof, ndof))) - np.kron(tau_x, np.ones((ndof, ndof))))
}

charge_op = np.kron(tau_z, np.eye(ndof))

# %% We build the bloch_family.
# Particle-hole symmetry
PHS = qsymm.particle_hole(ndim, np.kron(tau_x, np.eye(ndof)))
symmetries = [PHS]

hams = (h_sc_0, h_sc_int)
hoppings = list(h_sc_int.keys())

# We recalculate the ndof, since we now have double it for the holes and electrons.
ndof_sc = ndof * 2

ham_fam = tb_to_ham_fam(hoppings, ndof_sc, symmetries)

# %% We generate an initial guess with the ham_fam.
mf_guess = symm_guess_mf(ham_fam)


# %% We set up the `meanfi` model and compute a solution.
def compute_sol(h_0, h_int, nk, ham_fam, guess, target_charge, kT):
    model = Model(h_0, h_int, target_charge, charge_op, kT)
    h_sol = solver_density_symmetric(
        model, ham_fam, guess, nk=nk
    )  # , optimizer_kwargs={"verbose": True})

    return h_sol


# %% We plot the observables
def plot_bands(ham, nk):
    ks = np.linspace(0, 2 * np.pi, nk, endpoint=False)
    hamiltonians = tb_to_kgrid(ham, nk)

    vals = np.linalg.eigvalsh(hamiltonians)
    plt.plot(ks, vals, c="k")
    plt.xticks([0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2\\pi$"])
    plt.xlim(0, 2 * np.pi)
    plt.ylabel("$E - E_F$")
    plt.xlabel("$k / a$")
    plt.show()


def compute_gap(full_sol, nk_dense, fermi_energy=0):
    h_kgrid = tb_to_kgrid(full_sol, nk_dense)
    vals = np.linalg.eigvalsh(h_kgrid)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)


# %%
nk = 100
target_charge = 0
kT = 0

h_int_solution = compute_sol(h_sc_0, h_sc_int, nk, ham_fam, mf_guess, target_charge, kT)
h_mf = add_tb(h_sc_0, h_int_solution)

plt.title(f"U = {U}, Imaginary part")
plt.imshow(np.imag(h_int_solution[(0,)]), cmap="coolwarm")
plt.colorbar()
plt.show()

plt.title(f"U = {U}, Real part")
plt.imshow(np.real(h_int_solution[(0,)]), cmap="coolwarm")
plt.colorbar()
plt.show()

plot_bands(h_mf, nk)

from tqdm import tqdm

n = 100
temperatures = np.linspace(0, 0.225, n)
gaps = np.zeros_like(temperatures)
nk_dense = 10000
for i in tqdm(range(n)):
    h_mf = add_tb(
        h_sc_0,
        compute_sol(
            h_sc_0, h_sc_int, nk, ham_fam, mf_guess, target_charge, temperatures[i]
        ),
    )
    gaps[i] = compute_gap(h_mf, nk_dense)
plt.plot(temperatures, gaps)
plt.title("Gap over Temperature")
plt.xlabel("kT")
plt.ylabel("Gap")
plt.show()
