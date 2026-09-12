# `PeriodicGrid`

`PeriodicGrid` has one fixed-grid evaluator and a global refinement loop.
It supports full density blocks and selected real-space entries.

## Prescribed mesh

`PeriodicGrid(nk=N)` chooses the smallest integer `n` with `n**dimension >= N`,
and samples the periodic tensor grid without duplicating the boundary. It
performs no hidden refinement or error estimate. This mode supports finite
and zero temperature, including the explicit zero-temperature BdG workflow.
The requested and actual sizes are reported separately.

## Accuracy control

At positive temperature, omit `nk` to start from a small deterministic grid:

1. Find the chemical potential for the requested filling on the current grid,
   or use the supplied fixed chemical potential.
2. Integrate charge and requested density entries.
3. Compare with the previous grid at the **same chemical potential**.
4. When both changes meet their targets, validate with a deterministic shifted
   grid at the same resolution and chemical potential.
5. Accept only if validation passes; otherwise double every axis and repeat.

Validation offsets are deterministic and distinct by axis: `sqrt(p) % 1`
for successive primes `p = 2, 3, 5, ...`. Shifted validation is mandatory:
nested meshes alone can agree through aliasing.
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
different quantities. Validation and density recomputation add work beyond the
final mesh size. None of these settings is a total-process RSS limit.

Direct diagonalization is the default. Explicit `RationalFOE` is supported only
for sparse matrices on prescribed positive-temperature grids. See
[matrix functions](../matrix_functions.md) for this capability boundary.

`statistics.n_diagonalizations` counts direct spectral decompositions, including
root preparation and validation. It is `None` for RationalFOE, whose internal
spectral-bound work is not instrumented; `n_kernel_evals` counts its point
evaluations. A missing count must not be interpreted as zero work.
