# MeanFi design

MeanFi solves self-consistent tight-binding models with density-density
interactions. The calculation is:

trial density -> interaction correction -> Hamiltonian -> density at the requested
filling -> SCF update.

## Concepts and ownership

- `Model` owns immutable physical inputs, the reference density, the bare
  normal/BdG Hamiltonian, and a private reduced density space. Its public
  `required_coordinates` describes the entries needed by the interaction.
- `meanfield.py` implements the linear interaction correction and its quadratic
  energy contraction. Observables and SCF use these same operations.
- `DensityCoordinates` describes entry addresses; slices are derived from them.
  `DensityResult` exposes computed values, optional entry errors, and physical
  metadata, including evaluation temperature. Model-based results retain known
  internal energy, preserved by selected views. Its immutable
  payload is private. Missing entries are unknown.
  Only internal constrained reconstruction assembles incomplete blocks with zeros.
- `space/` builds one compact representation of Hermiticity and pairing
  antisymmetry. General spatial constraints materialize that same representation
  as a basis and reduce it further. The common path has linear storage.
- A Model owns temperature and filling; calls with a Model cannot override
  them. Use a replaced Model to change physical inputs. All method settings
  are keyword-only and all integration methods return `IntegrationInfo`.
- `density/problem.py` resolves integration defaults and compatibility once.
  Its prepared sparse coordinate pattern survives Hamiltonian updates in SCF.
- `scf/problem.py` evaluates the physical self-consistency map. One evaluation
  record carries the density, input/output states, residual and internal energy.
  The driver records accepted evaluations; the Anderson adapter owns trial
  callbacks. Mixing methods retain their own update and iteration limit.
- `tb/` owns tight-binding operations and contractions. `observables.py` combines
  those with interaction energy to expose physical observables.

## Numerical methods

For each momentum, the density is the Fermi function of `A = H(k) - mu Q`. `Q = I`
normally; BdG uses the electron/hole charge diagonal. Filling searches use a bracket
and verify the charge residual. A small chemical-potential step alone does not
establish convergence.

Normal zero-temperature calculations use `FermiSimplex`. `UniformGrid` supports
normal and BdG models, with direct diagonalization or positive-temperature sparse
AAA/MUMPS evaluation on a prescribed mesh. Adaptive grids compare coarse and fine
integrals on nested meshes. `initial_nk` selects the starting mesh; otherwise
UniformGrid starts with four points per axis and FermiSimplex with three vertices
per axis. Prescribed meshes provide no integration error estimate.

For Hermitian `A`, a scalar Fermi-function error bounds every density entry:

`max_ij |[r(A)-f(A)]_ij| <= ||r(A)-f(A)||_2 = max_spectrum |r-f|`.

AAA controls the worst density-matrix element through the scalar Fermi function,
using the policy's `matrix_function_tol` independently of the momentum-integration
target. `matrix_function_error` reports the sampled achieved scalar error for the
returned density; direct diagonalization and FermiSimplex report None for this
approximation estimate. A weighted charge trace can accumulate errors, so filling
searches also constrain the fit to one quarter of the filling-residual budget
divided by the absolute trace-weight sum. Fixed-mu fits use only the matrix-function
target. Mesh-integration and SCF residual checks remain separate.

AAA uses a small adaptive fitting grid and a dense validation grid; QR reduces its
denominator solve to a small SVD. Its acceptance depends only on the Fermi function.
Scalar checks and mesh estimates are empirical, not rigorous bounds between sampled
points. Independent dense references also check matrix errors.

Entropy is optional postprocessing. Public calculations default to
`compute_free_energy=True`; `False` leaves entropy, its error and free energy
unknown (`None`). Internal SCF evaluations never compute entropy. At termination,
including a failure with a valid state, one fixed-mu evaluation of the exact
input Hamiltonian supplies final entropy using the existing density accuracy
policy. There is no new entropy target. This may repeat matrix work once, but
retains no history of factors or eigenvectors. AAA regenerates density-accepted
poles and fits entropy on those poles; entropy never changes pole selection.

Every backend exposes `errors.entropy`, an estimate of the total entropy error
per orbital when available. It combines integration and matrix-function estimates;
prescribed periodic meshes and simplex cases lacking an entropy integration
estimate report None. Finite systems have zero integration error. An unavailable
estimate never becomes a fabricated zero. Numerical estimates remain empirical.
Computed energies belong to results, with no hidden Model reference. Observable
helpers evaluate trial densities using the supplied model and require the relevant
entries. SCFResult delegates its scalar quantities and errors to DensityResult.
Finite systems use the same integration methods; supplied nk/initial_nk are warned
about and ignored. Sparse-to-dense evaluation requires an explicit method.

One bounded scalar fit can be shared across k-points and chemical potentials.
Numeric factors belong to one Hamiltonian and chemical potential. Density, energy
and entropy reuse those factors; retained storage remains bounded.

## Interaction and SCF

Write `delta = rho - reference`. The correction is `W[delta]` and the interaction
energy is `Tr(W[delta] delta)/(2N)` normally. BdG uses the normal and conjugate
pairing contraction with Nambu counting applied once. The one-body term always uses
the actual density. References define a modified interaction model, not an energy
difference from the reference. Both complete density dictionaries and selected
results are accepted; every required entry is checked. A normal N-orbital reference
in a BdG model specifies zero reference pairing.

EDIIS is the default and minimizes internal energy over convex density history. The
interaction energy of pairwise density differences supplies its exact quadratic
curvature; reference offsets cancel. A small history objective is prepared once per
update. Entropy and free energy do not enter the optimization. SCF stops on the
largest reconstructed complex density-entry residual, or raises at its iteration
limit. Users explicitly compose solver calls to change methods. Failures retain the
last accepted result when available.

## Units, accuracy, and verification

Filling counts electrons per cell. Energy and entropy are per cell per physical
orbital: a 2N-dimensional BdG Hamiltonian has N physical orbitals. Entropy is in
units of Boltzmann's constant; free energy is `internal_energy - kT * entropy`.
Generic observable contractions remain raw traces per cell.

The existing tolerance policy assigns `tol/5` to density integration, charge
integration and matrix-function approximation, `tol/10` to filling residual, and
`tol` to SCF residual. An explicit density target also supplies an omitted charge
target; users can override charge independently. Energy and entropy have no accuracy
targets. Unavailable error estimates are `None`. The entropy estimate is diagnostic and never controls convergence.

Numerical changes are checked against exact finite systems or independently
converged dense references, including complex pairing phases, reference subtraction,
units, coordinate selection and resource limits. See `RELEASE_CHECKS.md` for
repeatable checks. Historical benchmarks remain under `performance/`, outside
distributions.
