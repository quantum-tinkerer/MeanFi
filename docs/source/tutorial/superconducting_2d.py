# %% Example
# We import all the required functions.
import kwant
import qsymm
from meanfi.tb.transforms import tb_to_ham_fam, tb_to_kgrid
import numpy as np
from meanfi.model import Model
from meanfi.tb.utils import symm_guess_mf
from meanfi.tb.tb import add_tb
from meanfi.solvers import solver_density_symmetric
from meanfi.kwant_helper import utils
import matplotlib.pyplot as plt

tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
# %%
# Create graphene lattice
ndof = 2  # spin
norbs = 2 * ndof  # particle-hole and spin
square_lattice = kwant.lattice.square(norbs=norbs)


def square_shape(pos):
    x, y = pos
    return True  # Infinite system


syst = kwant.Builder(kwant.TranslationalSymmetry(*square_lattice.prim_vecs))

syst[square_lattice.shape(square_shape, (0, 0))] = 0 * np.kron(tau_z, np.eye(ndof))
syst[square_lattice.neighbors(1)] = np.kron(tau_z, np.eye(ndof))
# %%
kwant.plot(syst)
# %%
# tile a few unit cells to visualize
finite_syst = kwant.Builder()


def finite_shape(site):
    x, y = site.pos
    return (0 <= x < 8) and (0 <= y < 8)


finite_syst.fill(syst, finite_shape, (0, 0))
kwant.plot(finite_syst)
# %%
sysf = finite_syst.finalized()
hamiltonian = sysf.hamiltonian_submatrix()
plt.imshow(hamiltonian.real, cmap="bwr")
# %%
evals = np.linalg.eigvalsh(hamiltonian)
plt.plot(evals, marker="o", linestyle="", markersize=3)
# %%
wrapped_syst = kwant.wraparound.wraparound(syst).finalized()
kwant.plot(wrapped_syst)
# %%
ham_func = lambda k_x, k_y: wrapped_syst.hamiltonian_submatrix(
    params={"k_x": k_x, "k_y": k_y}
)

ks = np.linspace(0, 2 * np.pi, 20, endpoint=True)
hams = np.array([[ham_func(kx, ky) for ky in ks] for kx in ks])

evals_h_0 = np.linalg.eigvalsh(hams)

cmap = plt.get_cmap("twilight")
norms = plt.Normalize(vmin=0, vmax=2 * np.pi)
for i, ky in enumerate(ks):
    # color each momentum slice differently in shades of gray
    plt.plot(ks, evals_h_0[:, i, :], c=cmap(norms(ky)), linewidth=0.5)
# %%
plt.plot(ks, evals_h_0[0, :, :], linewidth=0.5)
# %%
fermi_surface = np.min(np.abs(evals_h_0), axis=2)

plt.figure()
plt.contourf(*np.meshgrid(ks, ks), fermi_surface, levels=1, cmap="Blues", alpha=0.5)
plt.colorbar()
plt.title("Fermi Surface")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.show()


# %%
h_0 = utils.builder_to_tb(syst)


# %%
def onsite_int(site, U):
    return U * np.kron((tau_0 - tau_x), np.ones((ndof, ndof)))


builder_int = utils.build_interacting_syst(
    builder=syst, lattice=square_lattice, func_onsite=onsite_int, max_neighbor=1
)
params = {"U": -2}
h_int = utils.builder_to_tb(builder_int, params)

ndim = 2
# %%
charge_op = np.kron(tau_z, np.eye(ndof))

# %% We build the bloch_family.
# Particle-hole symmetry
PHS = qsymm.particle_hole(ndim, np.kron(tau_x, np.eye(ndof)))
symmetries = [PHS]

hams = (h_0, h_int)
hoppings = list(h_int.keys())

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
nk = 20
target_charge = 0
kT = 0

h_int_solution = compute_sol(h_0, h_int, nk, ham_fam, mf_guess, target_charge, kT)
h_mf = add_tb(h_0, h_int_solution)
# %%

n = 20
max_temp = 0.2
temperatures = np.linspace(0, max_temp, n)
gaps = np.zeros_like(temperatures)
nk_dense = 10
for i in range(n):
    h_mf_kT = add_tb(
        h_0,
        compute_sol(h_0, h_int, nk, ham_fam, mf_guess, target_charge, temperatures[i]),
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
# %%
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

# %%
plt.plot(ks, vals[:, 15, :])
# %%
vals.shape
# %%
