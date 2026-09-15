# MeanFi design

MeanFi solves self-consistent tight-binding models with density-density
interactions. The implementation should follow the physical calculation directly:
trial density -> mean-field Hamiltonian -> density at the requested filling ->
SCF update. This document describes the contracts shared by those steps; the
algorithm reference in `docs/source/documentation/algorithms/` gives the details.

## Concepts and ownership

- `Model` owns immutable Hamiltonian and interaction blocks, physical temperature
  and filling, and an `ActiveSCFSpace` that encodes Hermiticity, particle-hole
  constraints, and optional spatial symmetry.
- `DensityCoordinates` describes computed entries. `DensityEntries` owns their
  immutable values and optional integration errors, shared when result metadata
  changes. `DensityResult` adds physical quantities and evaluation diagnostics.
  Missing entries are unknown, never implicitly zero. The SCF space reconstructs only entries fixed by its
  constraints.
- `DensityProblem` holds validated, resolved integration settings and coordinates.
  Public density calls and SCF use the same preparation boundary and evaluator.
  Integration backends consume this resolved problem directly.
- One SCF problem implements the shared evaluation flow. Small normal/BdG
  helpers supply the distinct correction and interaction-energy formulas.
- Density results own physical quantities and numerical error estimates.
  Statistics describe numerical work and retained storage, not physical values.

The runtime hierarchy follows those responsibilities:

| Location | Responsibility |
| --- | --- |
| `model.py`, `meanfield.py`, `observables.py` | Physical inputs, corrections, observables |
| `space/`, `tb/` | Coordinates, constraints, tight-binding operations |
| `scf/problem.py`, `scf/engine.py` | Shared physical SCF map and iteration |
| `density/problem.py`, `density/density.py` | Resolved inputs and backend dispatch |
| `density/integrate/periodic.py`, `periodic_grid.py` | Refinement and streamed grid evaluation |
| `density/integrate/simplex/` | Shared solve loop and native mesh operations |
| `density/kpoint/matrix_functions/` | Dense and sparse matrix-function algorithms |
| `results.py`, `errors.py` | Physical results, work statistics, numerical targets |

## Numerical methods and assumptions

For each momentum, occupations are the Fermi function of `H(k) - mu Q`, with
`Q = I` normally and the electron/hole charge diagonal for BdG models. Filling
searches use a bracket and verify the charge residual; a small change in chemical
potential alone does not establish convergence.

Normal zero-temperature calculations use `FermiSimplex`. `UniformGrid`
integration uses direct diagonalization or, for prescribed sparse calculations
at positive temperature, AAA and MUMPS selected inversion. Adaptive periodic calculations
compare nested and shifted grids. Prescribed meshes do not estimate integration
error. Sampled rational and integration checks are empirical accuracy checks,
not rigorous bounds between sample points.

AAA jointly approximates occupation and entropy, sharing poles and shifted
factorizations. A small fitting grid is refined as needed; every accepted fit
passes a dense validation grid. QR reduces the denominator solve to a small SVD.
One validated scalar fit may be reused within a calculation. Sparse coordinate
patterns belong to the calculation; numeric factors belong to an individual
Hamiltonian and chemical potential. Retained storage must remain bounded.

EDIIS is the default SCF update and minimizes internal energy over its density
history. The interaction is quadratic, so history energies and gradient
differences define a small exact quadratic objective. It is prepared once per
update; coefficient optimization does not reconstruct model states. Each SCF method runs only its own update until convergence or its
iteration limit. Methods never switch automatically. Users compose separate
solver calls, using the last valid result attached to a convergence failure
to restart with a method of their choice. SCF history and progress output keep
internal energy only. Final results also report entropy and free energy from
the existing density evaluation, without another matrix pass.

## Quantities and accuracy

Filling counts electrons per cell. Band, internal, and free energies and entropy
are per cell per physical orbital; a 2N-dimensional BdG Hamiltonian has N
physical orbitals. Entropy is in units of Boltzmann's constant. Free energy is
`internal_energy - kT * entropy`. Generic observable contractions remain raw
traces. Interaction double counting and BdG normal ordering are applied once.

Normal reference subtraction uses `delta = rho - reference` in both the
Hartree/Fock correction and quadratic interaction energy. The energy per orbital
is `(Tr(h_0 rho) + Tr(W[delta] delta)/2)/N`; differentiating the total
energy `N * U` gives the effective Hamiltonian. The reference's one-body energy
is not removed, and entropy belongs to the actual state. This defines a modified model, not an energy difference
from the reference. BdG references subtract both normal and anomalous density
components in the same quadratic functional. An N-orbital normal reference
means zero reference pairing; a 2N-dimensional BdG reference must supply the
required normal and pairing entries. Map normal references directly into the
selected entries, without constructing Nambu matrices or inserting a hole
identity into a density difference.

Density accuracy controls the calculation. Filling and SCF have their own
residual checks. Energy and entropy are computed on the accepted density mesh;
their estimated errors are diagnostics and never trigger refinement. The sparse
entropy fit shares the occupation fit's scalar accuracy and poles; band energy
uses that occupation approximation without an extra accuracy target.

The default policy assigns `tol/5` to density and charge integration, `tol/10`
to the filling residual, and `tol` to the SCF residual. An explicit density
target also supplies an omitted charge target; an explicit charge target may
be tighter or looser. Without mesh overrides, custom tolerance policies retain
both of their targets. An unavailable estimate is `None`, including
thermodynamic integration errors that a backend cannot estimate.

## Verification and release

Numerical changes are checked against exact finite systems or independently
converged dense references. Tests preserve complex BdG phases, reference
subtraction, units, coordinate selection, and resource limits. Rational
benchmarks retain their original trace tolerances, including the 1e-12
regression, and compare complete solves as well as coefficient work.

Run the core suites on Python 3.11–3.13, the optional sparse suite, slow numerical
regressions when algorithms change, executed tutorials, installed-wheel checks,
and clean dependency installation before release. Benchmark records document
reference errors, tolerances, timing conditions, and source versions. Historical
benchmark evidence is retained outside the distributions.
