# `AdaptiveSimplex`

`AdaptiveSimplex` is MeanFi's main integration method for normal systems at zero
temperature. [FermiSimplex](https://gitlab.kwant-project.org/qt/lineartetrahedron)
provides the native physics integration; its generic mesh engine is
[adaptivesimplex](https://gitlab.kwant-project.org/qt/adaptivesimplex).

## Accuracy control

With no `nk`, the backend refines a simplicial partition, comparing each simplex
contribution with a preview on refined descendants. It focuses refinement where
the estimated error is largest. The density stage reuses the charge-refined
mesh and computes only the required real-space entries, using preview depth one.
Density and charge tolerances are separate integration targets.

## Prescribed mesh

`AdaptiveSimplex(nk=N)` uses the native `SpectralMesh(root_level=L)` constructor.
It chooses the smallest nonnegative `L` whose node count
`(2**L + 1)**dimension` reaches `N`. Both reduced-coordinate boundary nodes `0`
and `1` are counted, even though they are periodically equivalent. Native mesh
granularity can therefore overshoot the request substantially.

For example, a request for 4096 total nodes produces 4225 nodes (65²) in 2D and
4913 nodes (17³) in 3D. Requested and actual counts are reported separately.
The backend still integrates over simplices; it does not replace them with point
sampling. Charge is estimated on the current mesh and density uses preview depth
zero, without subsequent adaptive refinement. Integration errors are unavailable.

`max_points` is a separate hard limit on retained spectral nodes, including
density-preview vertices. An initial mesh that would exceed it fails clearly.
Adaptive refinement reserves a conservative native work bound before each call;
shared vertices can make it stop below the limit. `max_refinements` independently
limits accuracy-controlled refinement.
`num_threads` limits native OpenMP threads and defaults to one; pass `None` to
leave the backend thread count unrestricted.

FermiSimplex band-energy results and normal zero-temperature SCF remain supported.
This family does not support finite temperature or superconducting BdG systems.
