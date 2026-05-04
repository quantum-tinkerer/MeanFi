# %% Imports
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np

from meanfi.interop import kwant as utils

# %% Tuning knobs
density_atol = 1e-2
charge_tol = 1e-2
scf_tol = 1e-3
max_iterations = 1000

filling = 2
U0 = 0.2
V0 = 1.2

gap_nk = 100
phase_diagram_nu = 10
phase_diagram_nv = 10

np.random.seed(0)


# %% Pauli matrices
s0 = np.identity(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1, -1])


# %% Build graphene lattice and non-interacting model
graphene = kwant.lattice.general(
    [(1, 0), (1 / 2, np.sqrt(3) / 2)],
    [(0, 0), (0, 1 / np.sqrt(3))],
    norbs=2,
)
a, b = graphene.sublattices

bulk_graphene = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
bulk_graphene[a.shape((lambda pos: True), (0, 0))] = 0 * s0
bulk_graphene[b.shape((lambda pos: True), (0, 0))] = 0 * s0
bulk_graphene[graphene.neighbors(1)] = s0

h_0 = utils.builder_to_tb(bulk_graphene)


# %% Build interaction model
def onsite_int(site, U):
    del site
    return U * sx


def nn_int(site1, site2, V):
    del site1, site2
    return V * np.ones((2, 2))


builder_int = utils.build_interacting_syst(
    builder=bulk_graphene,
    lattice=graphene,
    func_onsite=onsite_int,
    func_hop=nn_int,
    max_neighbor=1,
)

params = dict(U=U0, V=V0)
h_int = utils.builder_to_tb(builder_int, params)

int_keys = frozenset(h_int)
ndof = len(next(iter(h_0.values())))
integration = meanfi.AdaptiveSimplex(density_matrix_tol=density_atol, refinement_depth=2)


# %% Single solve
model = meanfi.Model(h_0, h_int, filling=filling)
guess = meanfi.guess_tb(int_keys, ndof)

result = meanfi.solver(
    model,
    guess,
    integration=integration,
    scf=meanfi.AndersonMixing(M=0, line_search="wolfe", max_iterations=max_iterations),
    scf_tol=scf_tol,
    filling_tol=charge_tol,
)
h_full = meanfi.add_tb(h_0, result.mf)

print("residual_norm =", result.info.residual_norm)
print("mu =", result.density_matrix_result.mu)
print("filling =", result.density_matrix_result.filling)


# %% CDW order parameter
cdw_operator = {(0, 0): np.kron(sz, np.eye(2))}

rho_result = meanfi.density_matrix(
    h_full,
    filling=filling,
    keys=[(0, 0)],
    integration=integration,
    filling_tol=charge_tol,
)
rho_0_result = meanfi.density_matrix(
    h_0,
    filling=filling,
    keys=[(0, 0)],
    integration=integration,
    filling_tol=charge_tol,
)

rho = rho_result.density_matrix
rho_0 = rho_0_result.density_matrix

cdw_order_parameter = meanfi.expectation_value(rho, cdw_operator)
cdw_order_parameter_0 = meanfi.expectation_value(rho_0, cdw_operator)

print("CDW interacting =", np.round(np.abs(cdw_order_parameter), 6))
print("CDW non-interacting =", np.round(np.abs(cdw_order_parameter_0), 6))


# %% Gap helper
def compute_gap(h, fermi_energy=0, nk=100):
    kham = meanfi.tb_to_kgrid(h, nk)
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)


print("gap =", compute_gap(h_full, fermi_energy=0, nk=gap_nk))


# %% Phase diagram sweep
Us = np.linspace(0, 4, phase_diagram_nu)
Vs = np.linspace(0, 1.5, phase_diagram_nv)

gaps = []
mf_sols = []
for U in Us:
    for V in Vs:
        params = dict(U=U, V=V)
        h_int = utils.builder_to_tb(builder_int, params)

        model = meanfi.Model(h_0, h_int, filling=filling)
        guess = meanfi.guess_tb(int_keys, ndof)
        result = meanfi.solver(
            model,
            guess,
            integration=integration,
            scf=meanfi.AndersonMixing(
                M=0,
                line_search="wolfe",
                max_iterations=max_iterations,
            ),
            scf_tol=scf_tol,
            filling_tol=charge_tol,
        )
        mf_sols.append(result.mf)

        gap = compute_gap(meanfi.add_tb(h_0, result.mf), fermi_energy=0, nk=gap_nk)
        gaps.append(gap)

gaps = np.asarray(gaps, dtype=float).reshape((len(Us), len(Vs)))
mf_sols = np.asarray(mf_sols, dtype=object).reshape((len(Us), len(Vs)))

plt.imshow(gaps.T, extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin="lower", aspect="auto")
plt.colorbar()
plt.xlabel("V")
plt.ylabel("U")
plt.title("Gap")
plt.show()


# %% SDW and CDW phase map
s_list = [sx, sy, sz]
cdw_list = []
sdw_list = []
for mf_sol in mf_sols.flatten():
    rho = meanfi.density_matrix(
        meanfi.add_tb(h_0, mf_sol),
        filling=filling,
        keys=[(0, 0)],
        integration=integration,
        filling_tol=charge_tol,
    ).density_matrix

    cdw_list.append(np.abs(meanfi.expectation_value(rho, cdw_operator)) ** 2)

    sdw_value = 0
    for s_i in s_list:
        sdw_operator_i = {(0, 0): np.kron(sz, s_i)}
        sdw_value += np.abs(meanfi.expectation_value(rho, sdw_operator_i)) ** 2
    sdw_list.append(sdw_value)

cdw_list = np.asarray(cdw_list).reshape(mf_sols.shape)
sdw_list = np.asarray(sdw_list).reshape(mf_sols.shape)

normalized_gap = gaps / np.max(gaps)
plt.imshow(
    (cdw_list - sdw_list).T,
    extent=(Us[0], Us[-1], Vs[0], Vs[-1]),
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    alpha=normalized_gap.T,
    vmin=-2.6,
    vmax=2.6,
)
plt.colorbar()
plt.xlabel("V")
plt.ylabel("U")
plt.title("CDW vs SDW")
plt.show()
# %%
