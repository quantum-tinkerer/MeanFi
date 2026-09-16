# MeanFi design

MeanFi solves tight-binding models with density-density interactions by repeating

`density -> interaction correction -> Hamiltonian -> density at fixed filling -> EDIIS`.

The goal is to compute only the density entries and matrix work needed by that
calculation, with explicit accuracy targets and no hidden convergence stages.

## Physical objects

`Model` owns validated, immutable Hamiltonian/interaction dictionaries, temperature,
filling, an optional reference density and symmetry constraints. `meanfield.py`
contains the canonical interaction correction and energy contraction. For a
reference, write `delta = rho - reference`: the correction is `W[delta]` and the
normal interaction energy is `Tr(W[delta] delta)/(2N)`. The one-body energy always
uses the actual density. BdG applies the corresponding normal/pairing contractions
with Nambu counting once. Reference subtraction defines an interaction model, not
an energy difference from the reference.

`DensityCoordinates` lists requested real-space entries. `DensityResult` stores
those values and available observables/diagnostics. Missing entries are unknown;
missing metadata is `None`. No calculation is performed solely to fill a result
field. `SCFResult` adds the input correction, convergence status and iteration
history. Its correction reproduces its density; applying `model.mean_field` to
that density gives the next correction, equal only at self-consistency.

`space/` compresses Hermiticity and pairing antisymmetry into real variables.
Ordinary models use entry mappings with linear storage. Spatial constraints use
a nullspace basis of the same representation. With the Fourier convention
`H(k) = sum_R H_R exp(-ik.R)`, shifted symmetries map displacements to
`A R + s_left - s_right`. The Hamiltonian and interaction must respect the symmetry.

Filling counts electrons per cell. Energy and entropy are per cell per physical
orbital: N for a normal Hamiltonian and N for a 2N-dimensional BdG Hamiltonian.
`free_energy = internal_energy - kT * entropy` with entropy in units of k_B.

## Density and integration

`density/problem.py` resolves backend compatibility, coordinate selection and
sparse patterns once. `density/filling.py` solves `N(mu) = filling` using the
requested filling residual and chemical-potential step tolerance. It checks the
previous mu before constructing a bracket; numerical samples are cached within
the root solve. Integration and matrix-function errors do not add root tests.

- `FermiSimplex` integrates normal zero-temperature density on an adaptive mesh.
  At fixed filling, charge/mu refinement finishes before one density stage at
  that mu. Density never restarts charge refinement. At fixed mu, only density is
  integrated; filling comes from a complete requested local diagonal or is None.
  The default seed has five vertices per axis: three vertices and their midpoint
  preview alias the first cosine harmonic and can falsely report zero error.
- `UniformGrid` streams points with dense diagonalization or positive-temperature
  sparse AAA. Normal dense charge is the sum of occupations. Adaptive grids use
  coarse/fine differences only, starting with four points per axis. `nk` specifies
  a prescribed total point count; `initial_nk` specifies the adaptive starting
  count. Prescribed grids have no integration-error estimate. Sparse AAA currently
  requires a prescribed grid for periodic systems.

Fixed-mu calls have no root search or independent charge-integration estimate.
Empty requests skip the numerical backend unless entropy is requested.
They never refine to obtain such a diagnostic. Filling is derived from density
information already available, or left None. Band energy is returned when it is
available without extra matrix work, needed by a filling calculation, or requested
through free-energy evaluation. Finite systems share these contracts and warn if
irrelevant grid sizes are supplied.

## Sparse matrix functions

AAA approximates the Fermi function on a spectral interval. For Hermitian A,
`max_ij |[r(A)-f(A)]_ij| <= max_spectrum |r-f|`. The implementation checks sampled
scalar errors against `matrix_function_tol`; it does not tighten that target.
A small fitting grid and QR followed by a small SVD find the poles. One scalar fit
can be reused across nearby intervals; matrix factors belong to one H(k) and mu.

A sparse node factors each pole once, then obtains only requested inverse entries.
Charge probes request the physical charge diagonal, excluding BdG hole entries.
Density has no prerequisite charge evaluation. Energy/entropy can request missing
diagonals from the same factors. Buffers grow only with computed entries. The
periodic driver discards each node after use; retaining final-mu factors across
passes is deliberately outside this change.

## Accuracy and SCF

All tolerance dependencies live in `errors.py`. A numeric tol resolves once via
`policy(tol)`. The default assigns tol to SCF, tol/5 to density and charge
integration, tol/10 to filling residual, and tol/40 to matrix-function approximation.
The policy is independent of matrix size and whether filling or mu is fixed.
Backends use these fields directly. Explicit `ErrorTolerances` bypass the policy
unchanged.

Filling residual, charge-integration error and the final density trace are distinct.
Mesh/scalar estimates are empirical indicators, not rigorous accuracy guarantees.
Energy and entropy have no stopping targets. Unavailable estimates remain None.

EDIIS minimizes internal energy over a convex density history. Its small quadratic
objective is prepared once per update. Convergence uses the largest reconstructed
complex density-entry residual. EDIIS never switches algorithms; users explicitly
compose solver calls. Failures retain the last accepted result when available.

Entropy and free energy are omitted by default. `compute_free_energy=True` requests
entropy with a density call or once at SCF termination, including a valid failed
result. It may repeat matrix work in that final pass. AAA fits entropy on the
accepted density poles without changing them or their acceptance criterion.

## Verification and repository boundaries

Tests compare against analytic finite/chain models and converged dense references,
including selected entries, reference subtraction, pairing, units and work counts.
Coverage is used to find unexercised paths, not as a reason to remove failure checks.
The small `performance/` runner reports time and reference errors for density and
SCF workloads. Generated reports, plots and distributions belong under ignored
build directories or CI artifacts. Tutorials execute fresh calculations and use
the default policy/EDIIS; the graphene scan explicitly chooses tol=1e-2 for speed.
