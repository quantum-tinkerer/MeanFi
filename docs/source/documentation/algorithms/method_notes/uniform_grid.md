# `UniformGrid`

`UniformGrid` has one fixed-grid evaluator and a global refinement loop.
It supports full density blocks and selected real-space entries.

## Prescribed mesh

`UniformGrid(nk=N)` chooses the smallest integer `n` with `n**dimension >= N`,
and samples the periodic tensor grid without duplicating the boundary. It
performs no hidden refinement or error estimate. This mode supports finite
and zero temperature, including the explicit zero-temperature BdG workflow.
The requested and actual sizes are reported separately.

## Accuracy control

At positive temperature, omit `nk` to refine from four points per axis by
default. Set `initial_nk=N` to choose a different starting total; it uses the
same rounding as `nk`, but allows refinement. For example, in 2D,
`UniformGrid(initial_nk=256)` starts at 16 × 16 and next evaluates 32 × 32.
`nk` and `initial_nk` are mutually exclusive.

1. Find the chemical potential for the requested filling on the current grid,
   or use the supplied fixed chemical potential.
2. Integrate charge, requested density entries, band energy, and entropy.
3. Compare with the previous grid at the **same chemical potential**.
4. Accept when the density and charge changes meet their targets; otherwise
   double every axis and repeat.

There are no shifted validation grids. Coarse/fine differences are empirical:
both grids can miss the same oscillation and agree through aliasing. Choose an
initial mesh that resolves known rapid momentum variation; tighter tolerances
alone cannot detect agreement caused by aliasing.
Band energy and entropy are computed on the accepted density mesh. Their error
estimates are diagnostics and do not trigger refinement. They are available as `errors.band_energy_integration` and `errors.entropy_integration`.
Both the quantities and their errors are per cell per physical orbital.
The estimates remain empirical. Exhausting `max_points` or `max_refinements`
raises an actionable convergence error rather than returning an unconverged
result. Zero-temperature periodic sampling requires an explicit `nk`.

## Memory and work

Hamiltonians and eigenvectors are evaluated in batches bounded by `batch_size`.
Density contributions are accumulated immediately and eigenvectors discarded.
Normal eigenvalues are retained across chemical-potential evaluations and nested
refinement, avoiding repeated diagonalization throughout the filling search.
`max_spectrum_bytes` bounds the peak old-plus-new spectrum storage during
refinement and causes a clear
failure if it is insufficient. BdG matrices depend on chemical potential, so
their spectra are recomputed when it changes.

`max_points` bounds total points in a grid, not points per axis. Retained spectra,
transient batch arrays, final mesh nodes and cumulative diagonalizations measure
different quantities. Root searches and density recomputation add work beyond the
final mesh size. None of these settings is a total-process RSS limit.

Direct diagonalization is the default. Explicit `RationalFOE` is supported only
for sparse matrices on prescribed positive-temperature grids. See
[matrix functions](../matrix_functions.md) for this capability boundary.

`statistics.n_diagonalizations` counts direct spectral decompositions, including
root preparation and density integration. RationalFOE uses Gershgorin spectral bounds and
shifted linear solves, so it performs zero Hamiltonian diagonalizations. Its
`n_kernel_evals` counts point evaluations; zero diagonalizations does not imply
zero work. Scalar coefficient construction is outside this Hamiltonian count.

`dtype="complex128"` is the default matrix workspace type. Use
`dtype="complex64"` to reduce storage when its lower precision is sufficient.
This controls complex array precision, not a different integration algorithm.
