# %%
import kwant
import numpy as np
import pandas as pd

import meanfi
from meanfi.kwant_helper import utils
from scipy.optimize import root
from meanfi.solvers import cost_density

s0 = np.identity(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1, -1])

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


def onsite_int(site, U):
    return U * sx


def nn_int(site1, site2, V):
    return V * np.ones((2, 2))


builder_int = utils.build_interacting_syst(
    builder=bulk_graphene,
    lattice=graphene,
    func_onsite=onsite_int,
    func_hop=nn_int,
    max_neighbor=1,
)
params = dict(U=2.32, V=0)
h_int_temp = utils.builder_to_tb(builder_int, params)
h_int = {(0, 0): h_int_temp[(0, 0)]}  # only keep onsite for efficiency
h_0 = utils.builder_to_tb(bulk_graphene)

filling = 2
model = meanfi.Model(h_0, h_int, filling=filling, atol=1e-4, kT=1e-2)
int_keys = frozenset(h_int)
ndof = len(list(h_0.values())[0])


# %%
max_iter = 100
common_options = {
    "tol": 1e-5,
}

method_options = {
    "df-sane": {"maxfev": max_iter},
    "anderson": {"jac_options": {"M": 1}, "maxiter": max_iter},
    "broyden2": {"maxiter": max_iter},
    "hybr": {"maxfev": max_iter},
}

# Define a range of random seeds for the experiments.
seeds = [1, 11, 21]

# A list to collect benchmarking results.
benchmark_results = []


# (Optional) A sample callback function to record convergence history.
def record_history_callback(residual, iteration, history):
    history.append((iteration, residual))
    return history


# Loop over each method and each seed.
for method, unique_opts in method_options.items():

    # Merge common options with the unique options and the method itself.
    optimizer_kwargs = {**common_options, "method": method, "options": unique_opts}

    for seed in seeds:
        np.random.seed(seed)

        # Initialize the guess (example; adjust to your code)
        mf_guess = meanfi.guess_tb(int_keys, ndof)
        mu_guess = 0

        # Prepare a list to capture the convergence history (if supported by your solver).
        history = []

        #
        if method == "hybr":
            callback_fun = None
        else:
            callback_fun = lambda res, it: record_history_callback(res, it, history)

        # Run the solver.
        # Here, we assume that `meanfi.solver` supports a callback mechanism to log residuals.
        # If not, you might need to modify the solver or wrap it to collect this info.
        _, result = meanfi.solver(
            model,
            mf_guess,
            mu_guess,
            optimizer=root,
            optimizer_kwargs=optimizer_kwargs,
            debug=True,
            optimizer_return=True,
            callback=callback_fun,
            save_history=True,
        )

        if method == "hybr":
            history = cost_density.history

        # Collect information from the solver result.
        # Adjust attribute names (like `n_iter`) based on what your result object provides.
        benchmark_results.append(
            {
                "method": method,
                "seed": seed,
                "success": result.success,
                "iterations": getattr(
                    result, "n_iter", None
                ),  # Replace 'n_iter' with actual attribute name if different
                "message": result.message,
                "history": history,  # Contains tuples of (iteration, residual) if available
            }
        )

        # Optionally, print immediate feedback.
        if result.success:
            print(
                f"[{method} | seed={seed}] Solver converged in {getattr(result, 'nfev', 'N/A')} iterations."
            )
        else:
            print(f"[{method} | seed={seed}] Solver did not converge: {result.message}")

# Convert collected results into a DataFrame for easier analysis.
df_results = pd.DataFrame(benchmark_results)

# %%
import matplotlib.pyplot as plt

# log plot of residuals
# in the same plot, for the same seed, compare the residuals of different methods in one plot, and the filling residuals in the other plot
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for method in df_results["method"].unique():
    for seed in df_results["seed"].unique():
        history = next(
            row["history"]
            for _, row in df_results.iterrows()
            if row["method"] == method and row["seed"] == seed
        )
        residuals = [np.linalg.norm(residual[:-1]) for residual, _ in history]
        filling_residual = [residual[-1] for residual, _ in history]
        ax[0].plot(
            np.abs(filling_residual), marker="o", label=f"{method} (seed={seed})"
        )
        ax[1].plot(residuals, marker="o", label=f"{method} (seed={seed})")

ax[0].set_yscale("log")
ax[0].set_title("Filling Residuals Comparison")
ax[0].set_xlabel("Methods")
ax[0].set_ylabel("Filling Residuals")
ax[0].legend()
ax[0].grid(True)

# make this log scale
ax[1].set_yscale("log")
ax[1].set_title("Residuals Comparison")
ax[1].set_xlabel("Methods")
ax[1].set_ylabel("Residuals")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# %%
