"""Public API tour. Run with --sparse and/or --kwant to exercise installed extras."""

import argparse
from dataclasses import replace

import numpy as np
import meanfi as mf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--kwant", action="store_true")
args = parser.parse_args()

# Tight-binding keys are lattice displacements; matrices describe cell orbitals.
h0 = {
    (0,): np.array([[0.15, 0.08j], [-0.08j, -0.1]]),
    (1,): np.diag([-0.7, -0.5]),
    (-1,): np.diag([-0.7, -0.5]),
}
interaction = {(0,): np.array([[0.0, 0.25], [0.25, 0.0]])}
model = mf.Model(h0, interaction, filling=0.8, kT=0.2)
guess = model.random_meanfield(rng=12, scale=0.03)
solution = mf.solver(model, guess, tol=1e-5)
print("SCF:", solution.mu, solution.filling, solution.errors, len(solution.history))
print(
    "Internal energy / entropy / free energy (per cell per orbital):",
    solution.internal_energy,
    solution.entropy,
    solution.free_energy,
)
np.testing.assert_allclose(
    solution.free_energy, solution.internal_energy - model.kT * solution.entropy
)

# A Model supplies filling, temperature, normal/BdG structure and needed entries.
selected = mf.density_matrix(model, mean_field=solution.mean_field, tol=1e-6)
assert selected.covers(model.scf_space.required_coordinates)
assert not selected.is_complete
print("Selected entries:", selected.coordinates.entries, selected.values)

# Request full blocks for plotting, export, or observables with additional support.
full = mf.density_matrix(model, mean_field=solution.mean_field, keys=list(h0), tol=1e-6)
rho = full.to_tb()  # dict[displacement, ndarray]; selected.to_tb() would raise.
rho_sparse = full.to_tb(sparse=True)
subset = full.select(selected.coordinates)
np.testing.assert_allclose(subset.values, selected.values, atol=1e-6)
np.testing.assert_allclose(rho_sparse[(0,)].toarray(), rho[(0,)])

# The same functions also accept raw Hamiltonian dictionaries and explicit layouts.
h = model.hamiltonian_from_meanfield(solution.mean_field)
at_mu = mf.density_matrix_at_mu(h, full.mu, kT=model.kT, keys=list(h0), tol=1e-6)
np.testing.assert_allclose(at_mu.values, full.values, atol=1e-6)
correction = mf.meanfield(selected, interaction)
h_from_density = model.hamiltonian_from_density(selected)
np.testing.assert_allclose(h_from_density[(0,)], mf.add_tb(h0, correction)[(0,)])
print("Orbital polarization:", mf.expectation_value(full, {(0,): np.diag([1, -1])}))
print("Internal energy:", mf.internal_energy(model, full))
print("Free energy:", mf.free_energy(model, full))
assert subset.entropy == full.entropy  # Selecting entries keeps full-state metadata.
np.testing.assert_allclose(
    mf.free_energy(model, full),
    mf.internal_energy(model, full) - model.kT * full.entropy,
)

# Reference subtraction uses the complete Hartree/Fock correction of rho-rho_ref.
reference = mf.density_matrix(model, tol=1e-6)
referenced = replace(model, reference=reference)
np.testing.assert_allclose(
    referenced.hamiltonian_from_density(reference)[(0,)], h0[(0,)]
)
reference_solution = mf.solver(
    referenced, referenced.random_meanfield(rng=1, scale=0.01), tol=1e-5
)
assert reference_solution.converged

# EDIIS is the default for normal and BdG solves, including finite temperature.
# It minimizes internal energy and never switches methods automatically.
cold = replace(model, kT=0.0)
cold_solution = mf.solver(cold, cold.random_meanfield(rng=12, scale=0.03), tol=1e-4)
np.testing.assert_allclose(cold_solution.free_energy, cold_solution.internal_energy)
print("Zero-temperature energy:", cold_solution.internal_energy)

# Integration nk is a TOTAL point request. Explicit nk fixes the mesh;
# omitting it refines to the requested accuracy at positive temperature.
fixed = mf.density_matrix(model, integration=mf.PeriodicGrid(nk=64))
controlled = mf.density_matrix(
    model, integration=mf.PeriodicGrid(dtype="complex128"), tol=1e-5
)
assert fixed.entry_errors is None
assert controlled.entry_errors is not None
print("Fixed grid:", fixed.statistics.grid_shape)

# SCF settings are keyword-only: EnergyDIIS(history_size=6, max_iterations=100).
# Alternatives include AndersonMixing(alpha=.5) and LinearMixing(alpha=.5).
try:
    mf.solver(
        model,
        guess,
        integration=mf.PeriodicGrid(nk=64),
        scf=mf.LinearMixing(alpha=0.1, max_iterations=1),
        scf_tol=1e-14,
    )
except mf.NoConvergence as failure:
    assert failure.result is not None
    # This choice belongs to the caller; neither method switches automatically.
    restarted = mf.solver(
        model, failure.result.mean_field, tol=1e-5, scf=mf.AndersonMixing()
    )
    assert restarted.converged
# SolverFailure.result may be None if the first density evaluation fails.
# All numerical convergence failures are catchable as mf.ConvergenceError.

# A symmetry constrains the reduced density variables before SCF iteration.
swap = mf.SpatialSymmetry(np.eye(1, dtype=int), {(0,): np.array([[0, 1], [1, 0]])})
symmetric = mf.Model(
    {(0,): np.zeros((2, 2)), (1,): -np.eye(2), (-1,): -np.eye(2)},
    interaction,
    filling=0.8,
    kT=0.2,
    spatial_symmetries=(swap,),
)
assert mf.solver(
    symmetric, symmetric.random_meanfield(rng=2, scale=0.01), tol=1e-5
).converged

# Finite systems use the empty displacement and an empty Fourier grid shape.
finite = mf.Model({(): np.diag([-1.0, 1.0])}, {(): np.zeros((2, 2))}, filling=1.0)
np.testing.assert_allclose(
    mf.density_matrix(finite, keys=[()]).to_tb()[()], np.diag([1, 0])
)
np.testing.assert_allclose(mf.tb_to_kgrid(finite.h_0, ()), finite.h_0[()])

# BdG uses exactly the same density and Hamiltonian methods, with electron-first
# Nambu blocks. Chemical potential shifts diag(+I, -I), not the full identity.
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
# Internal-energy EDIIS needs a longer budget for this finite-temperature case.
bdg = mf.solver(
    pwave,
    pair_guess,
    integration=mf.PeriodicGrid(nk=256),
    tol=1e-5,
    scf=mf.EnergyDIIS(max_iterations=200),
)
bdg_density = mf.density_matrix(
    pwave,
    mean_field=bdg.mean_field,
    keys=[(0,), (1,), (-1,)],
    integration=mf.PeriodicGrid(nk=256),
    tol=1e-6,
)
bdg_h = pwave.hamiltonian_from_meanfield(bdg.mean_field)
quasiparticles = np.linalg.eigvalsh(
    mf.tb_to_kfunc(bdg_h)(np.array([0.3])) - bdg.mu * np.diag([1.0, -1.0])
)
print("BdG:", bdg_density.filling, quasiparticles)
print("BdG internal / free energy:", bdg.internal_energy, bdg.free_energy)
np.testing.assert_allclose(
    mf.free_energy(pwave, bdg_density),
    mf.internal_energy(pwave, bdg_density) - pwave.kT * bdg_density.entropy,
)

# Fourier helpers use explicit points PER AXIS and FFT ordering.
grid = mf.tb_to_kgrid(h0, (16,))
recovered = mf.kgrid_to_tb(grid)
np.testing.assert_allclose(mf.tb_to_kgrid(recovered, (16,)), grid, atol=1e-14)
print("Sampled Fermi level:", mf.fermi_energy(h0, filling=0.8, shape=(128,)))
print("Occupations:", mf.fermi_dirac([-1.0, 0.0, 1.0], kT=0.2, mu=0.0))

if args.sparse:
    from scipy.sparse import csr_matrix

    sparse_model = mf.Model(
        {key: csr_matrix(value) for key, value in h0.items()},
        {key: csr_matrix(value) for key, value in interaction.items()},
        filling=0.8,
        kT=0.2,
    )
    sparse_grid = mf.PeriodicGrid(nk=64, matrix_function=mf.RationalFOE())
    sparse_density = mf.density_matrix(sparse_model, integration=sparse_grid, tol=1e-5)
    print(
        "Sparse AAA:", sparse_density.mu, sparse_density.filling, sparse_density.entropy
    )
    sparse_solution = mf.solver(
        sparse_model,
        sparse_model.random_meanfield(rng=12, scale=0.03),
        integration=sparse_grid,
        tol=1e-5,
    )
    print(
        "Sparse internal / free energy:",
        sparse_solution.internal_energy,
        sparse_solution.free_energy,
    )

if args.kwant:
    import kwant
    from meanfi.interop.kwant import builder_to_tb, tb_to_builder

    lattice = kwant.lattice.chain(norbs=2)
    builder = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    builder[lattice(0)] = h0[(0,)]
    builder[lattice(0), lattice(1)] = h0[(1,)]
    tb, data = builder_to_tb(builder, sparse=True, return_data=True)
    rebuilt = tb_to_builder(tb, data["sites"], data["periods"])
    assert len(list(rebuilt.sites())) == 1

print("API walkthrough passed.")
