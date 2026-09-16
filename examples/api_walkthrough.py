"""Executable API tour; add --sparse and/or --kwant for the optional backends."""

import argparse
from dataclasses import replace

import meanfi as mf
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--kwant", action="store_true")
args = parser.parse_args()

# Tight-binding keys are cell displacements; matrices describe cell orbitals.
h0 = {
    (0,): np.array([[0.15, 0.08j], [-0.08j, -0.1]]),
    (1,): np.diag([-0.7, -0.5]),
    (-1,): np.diag([-0.7, -0.5]),
}
interaction = {(0,): np.array([[0.0, 0.25], [0.25, 0.0]])}
model = mf.Model(h0, interaction, filling=0.8, kT=0.2)
guess = model.random_meanfield(rng=12, scale=0.03)
solution = mf.solver(model, guess)  # Default EDIIS and tolerance policy.
print("SCF:", solution.mu, solution.filling, solution.errors)
print("Internal energy per cell per orbital:", solution.internal_energy)
assert solution.entropy is solution.free_energy is None

# SCF computes selected density entries, described by their coordinates.
selected = solution.density
assert selected.covers(model.required_coordinates)
assert not selected.is_complete
print("Selected entries:", selected.coordinates.entries, selected.values)

# Request complete blocks for observables/export; reuse the solved mu.
# Entropy/free energy are opt-in and are computed only for this final state.
full = mf.density_matrix_at_mu(
    model,
    solution.mu,
    mean_field=solution.mean_field,
    keys=list(h0),
    compute_free_energy=True,
)
rho = full.to_tb()  # Selected entries cannot be converted to complete blocks.
rho_sparse = full.to_tb(sparse=True)
subset = full.select(selected.coordinates)
np.testing.assert_allclose(subset.values, selected.values, atol=1e-3)
np.testing.assert_allclose(rho_sparse[(0,)].toarray(), rho[(0,)])
assert subset.entropy == full.entropy  # Selection preserves state metadata.
print("Polarization:", mf.expectation_value(full, {(0,): np.diag([1, -1])}))
print(
    "Internal / free energy:",
    full.internal_energy,
    full.free_energy,
)
np.testing.assert_allclose(
    full.free_energy, full.internal_energy - model.kT * full.entropy
)

# Energy helpers independently evaluate entries against the supplied model.
np.testing.assert_allclose(
    mf.evaluate_internal_energy(model, full), full.internal_energy
)
np.testing.assert_allclose(mf.evaluate_free_energy(model, full), full.free_energy)

# Raw dictionaries and explicit entry selections also work.
h = model.hamiltonian_from_meanfield(solution.mean_field)
coordinates = mf.DensityCoordinates.from_entries(
    size=2, keys=[(0,)], entries=(((0,), 0, 1),)
)
coherence = mf.density_matrix_at_mu(
    h, solution.mu, kT=model.kT, coordinates=coordinates
)
print("Onsite coherence:", coherence.values[0])
# Filling may be None if the requested density does not determine the trace.
# Fixed-mu calculations do not independently refine charge to populate a diagnostic.
assert coherence.errors.charge_integration is None
assert coherence.errors.filling_residual is None

# References subtract the Hartree/Fock correction of rho_ref. Observables still
# use the physical density; reference subtraction also enters interaction energy.
reference = mf.density_matrix(model)
referenced = replace(model, reference=reference)
np.testing.assert_allclose(
    referenced.hamiltonian_from_density(reference)[(0,)], h0[(0,)]
)
reference_solution = mf.solver(
    referenced, referenced.random_meanfield(rng=1, scale=0.01)
)
print("Reference-subtracted internal energy:", reference_solution.internal_energy)
# A user-supplied density dictionary is accepted too.
manual_reference = replace(model, reference={(0,): np.diag([0.4, 0.4])})

# nk is a TOTAL point request. An explicit nk fixes the integration grid;
# omitting it enables refinement and its density-integration error estimate.
fixed = mf.density_matrix(model, integration=mf.UniformGrid(nk=64))
adaptive = mf.density_matrix(model, integration=mf.UniformGrid(initial_nk=16))
assert fixed.entry_errors is None
assert adaptive.entry_errors is not None
print("Fixed grid:", fixed.statistics.grid_shape)
# Normal zero-temperature calculations default to FermiSimplex.
cold = mf.density_matrix(replace(model, kT=0.0))
print("Zero-temperature chemical potential:", cold.mu)

# Explicit targets are optional; normal calls use the policy without overrides.
targets = replace(mf.default_solver_tolerances(1e-4), charge_integration=1e-3)
custom = mf.density_matrix(model, tol=targets)
assert custom.errors.filling_residual <= targets.filling_residual

# Other SCF methods are explicit choices. Numerical failures retain a usable
# partial result when at least one density evaluation has succeeded.
try:
    mf.solver(model, guess, scf=mf.LinearMixing(alpha=0.1, max_iterations=1))
except mf.NoConvergence as failure:
    assert failure.result is not None
    restarted = mf.solver(model, model.mean_field(failure.result.density))
    assert restarted.converged
# SolverFailure.result can be None when the initial evaluation itself fails.
# ConvergenceError catches density/root failures and SCF convergence failures.

# Spatial symmetries reduce the active SCF variables before iteration.
swap = mf.SpatialSymmetry(np.eye(1, dtype=int), {(0,): np.array([[0, 1], [1, 0]])})
symmetric = mf.Model(
    {(0,): np.zeros((2, 2)), (1,): -np.eye(2), (-1,): -np.eye(2)},
    interaction,
    filling=0.8,
    kT=0.2,
    spatial_symmetries=(swap,),
)
assert mf.solver(symmetric, symmetric.random_meanfield(rng=2, scale=0.01)).converged

# Finite systems have the empty displacement and no momentum grid.
finite = mf.Model({(): np.diag([-1.0, 1.0])}, {(): np.zeros((2, 2))}, filling=1)
np.testing.assert_allclose(
    mf.density_matrix(finite, keys=[()]).to_tb()[()], np.diag([1, 0])
)
np.testing.assert_allclose(mf.tb_to_kgrid(finite.h_0, ()), finite.h_0[()])

# BdG uses electron-first Nambu blocks and the same solver/density methods.
pwave = mf.Model(
    {(0,): np.zeros((1, 1)), (1,): -np.ones((1, 1)), (-1,): -np.ones((1, 1))},
    {(1,): np.full((1, 1), 1.8), (-1,): np.full((1, 1), 1.8)},
    filling=0.35,
    kT=0.08,
    superconducting=True,
)
pair_guess = {
    (1,): np.array([[0.0, 0.25], [-0.25, 0.0]]),
    (-1,): np.array([[0.0, -0.25], [0.25, 0.0]]),
}
bdg = mf.solver(pwave, pair_guess)
bdg_h = pwave.hamiltonian_from_meanfield(bdg.mean_field)
# Chemical potential shifts the electron and hole blocks with opposite signs.
quasiparticles = np.linalg.eigvalsh(
    mf.tb_to_kfunc(bdg_h)(np.array([0.3])) - bdg.mu * np.diag([1.0, -1.0])
)
print("BdG:", bdg.filling, quasiparticles, bdg.internal_energy)

# A normal reference has zero pairing; a BdG reference subtracts both sectors.
normal_reference = mf.density_matrix(replace(pwave, superconducting=False))
normal_referenced_pwave = replace(pwave, reference=normal_reference)
paired_reference = replace(pwave, reference=bdg.density)
bare = pwave.hamiltonian_from_meanfield()
for key, block in paired_reference.hamiltonian_from_density(bdg.density).items():
    np.testing.assert_allclose(block, bare.get(key, np.zeros_like(block)), atol=1e-14)

# Fourier helpers use explicit points PER AXIS and FFT ordering.
grid = mf.tb_to_kgrid(h0, (16,))
recovered = mf.kgrid_to_tb(grid)
np.testing.assert_allclose(mf.tb_to_kgrid(recovered, (16,)), grid, atol=1e-14)
print("Sampled Fermi level:", mf.fermi_energy(h0, filling=0.8, shape=(128,)))
print("Occupations:", mf.fermi_dirac([-1, 0, 1], kT=0.2, mu=0))

if args.sparse:
    from scipy.sparse import csr_matrix

    sparse_model = replace(
        model,
        h_0={key: csr_matrix(value) for key, value in h0.items()},
        h_int={key: csr_matrix(value) for key, value in interaction.items()},
    )
    sparse_grid = mf.UniformGrid(nk=64, matrix_function=mf.RationalFOE())

    # A custom policy receives only tol. Backends use its targets as given.
    def accuracy(tol):
        defaults = mf.default_solver_tolerances(tol)
        return replace(defaults, charge_integration=tol)

    sparse_density = mf.density_matrix(
        sparse_model, integration=sparse_grid, tolerance_policy=accuracy
    )
    print("Sparse matrix-function error:", sparse_density.errors.matrix_function_error)
    assert sparse_density.errors.density_matrix_integration is None
    sparse_solution = mf.solver(
        sparse_model, sparse_model.random_meanfield(rng=12), integration=sparse_grid
    )
    print("Sparse internal energy:", sparse_solution.internal_energy)

if args.kwant:
    import kwant
    from meanfi.interop.kwant import builder_to_tb, tb_to_builder

    lattice = kwant.lattice.chain(norbs=2)
    builder = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    builder[lattice(0)] = h0[(0,)]
    builder[lattice(0), lattice(1)] = h0[(1,)]
    tb, geometry = builder_to_tb(builder, sparse=True, return_data=True)
    rebuilt = tb_to_builder(tb, geometry["sites"], geometry["periods"])
    assert len(list(rebuilt.sites())) == 1

print("API walkthrough passed.")
