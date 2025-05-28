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
        model, ham_fam, guess, nk=nk, optimizer_kwargs={"verbose": False}
    )

    return h_sol


# %%
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

from tqdm import tqdm

n = 100
temperatures = np.linspace(0, 0.21, n)
gaps = np.zeros_like(temperatures)
nk_dense = 10000
for i in tqdm(range(n)):
    h_mf_kT = add_tb(
        h_sc_0,
        compute_sol(
            h_sc_0, h_sc_int, nk, ham_fam, mf_guess, target_charge, temperatures[i]
        ),
    )
    gaps[i] = compute_gap(h_mf_kT, nk_dense)

# %% Plot everything.
plt.rcParams.update(
    {
        "font.size": 10,  # Set font size to 11pt
        "axes.labelsize": 10,  # -> axis labels
        "legend.fontsize": 8,  # -> legends
        "font.family": "lmodern",
        "text.usetex": True,
        "text.latex.preamble": (  # LaTeX preamble
            r"\usepackage{lmodern}"
            r"\usepackage{anyfontsize}"
            r"\usepackage{amsmath}"
            # ... more packages if needed
        ),
    }
)
W = 6.55994375 * 0.95  # Columnwidth used for the figure
# plt.rcParams.update({'figure.figsize': (W, W/(4/3))})
plt.rcParams.update({"figure.figsize": (W, W / (4 / 3) / 2)})
fig, ax = plt.subplots(1, 2)
ax[0].set_title("Real")
ax[0].set_xticks([])
ax[0].set_yticks([])
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
im1 = ax[0].imshow(np.real(h_int_solution[(0,)]), cmap="coolwarm", interpolation="none")
ax[1].set_title("Imaginary")
ax[1].set_xticks([])
ax[1].set_yticks([])
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
im2 = ax[1].imshow(np.imag(h_int_solution[(0,)]), cmap="coolwarm", interpolation="none")
cbar = fig.colorbar(im1, ax=ax[0], orientation="vertical", fraction=0.046, pad=0.04)
cbar = fig.colorbar(im2, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)
plt.tight_layout()
# plt.savefig('hams.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


fig, ax = plt.subplots(1, 2)
ks = np.linspace(0, 2 * np.pi, nk, endpoint=True)
hamiltonians = tb_to_kgrid(h_mf, nk)

vals = np.linalg.eigvalsh(hamiltonians)
ax[0].plot(ks, vals, c="k")
ax[0].axhline(color="k", linestyle="--", linewidth=1)
ax[0].set_xticks([0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2\\pi$"])
ax[0].set_xlim(0, 2 * np.pi)
ax[0].set_ylabel("$E - E_F$")
ax[0].set_xlabel("$k / a$")
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
# plt.savefig('bands.svg', bbox_inches = 'tight', pad_inches = 0)


def gap_over_temp(T, Tc, gap_0):
    gap = gap_0 * np.tanh(1.74 * np.sqrt(np.maximum(Tc / T - 1, 0)))
    return gap


theory_gaps = gap_over_temp(temperatures, 0.195, gaps[0])
print(gaps[0])

ax[1].margins(x=0)
ax[1].plot(temperatures, theory_gaps, linestyle=":", linewidth=2, label="Theoretical")
ax[1].axvline(0.195, 0, 1, color="k", linestyle="--", label=r"$T_c$")
ax[1].axhline(color="k", linestyle="--", linewidth=1)
ax[1].plot(temperatures, gaps, label="Calculated")
ax[1].legend().set_zorder(101)
# plt.title("Gap over Temperature")
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
plt.tight_layout()
# plt.savefig('results.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
