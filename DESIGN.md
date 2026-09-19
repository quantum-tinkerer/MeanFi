# MeanFi design

MeanFi solves tight-binding or callable Bloch Hamiltonians with density-density
or finite-range bilinear interactions, in normal or superconducting mean-field
states. Its core loop is

`density -> interaction correction -> Hamiltonian -> density at fixed filling -> mixing`.

Compute only the requested density entries and the work needed for convergence.
Numerical targets are explicit; unavailable results and error estimates are `None`.

## Physical model

`Model` owns the Hamiltonian, interaction, filling, temperature, optional reference
and spatial symmetries. It validates stored matrices once and retains dense or
sparse storage. Public containers share read-only arrays without exposing structural
mutation of the model. Use a new model or `dataclasses.replace` for changes.

`h_0` accepts a tight-binding dictionary or `BlochHamiltonian(function)`. The
callable's required positional arguments determine dimension; its matrix at the
origin determines orbital count. Bind other parameters in a closure or partial.
Captured parameters must remain fixed during a calculation. Evaluations check
shapes without scanning entries; finite Hermitian values are the caller's responsibility.

Integration uses normalized measure `d^d k / (2 pi)^d` on `[0, 2 pi]^d`.
There are no domain objects or implicit Jacobians. Compose a continuum Hamiltonian
with an equal-measure map, such as the tutorial's concentric square-to-disk map.
A varying Jacobian requires weighted integration, which is not supported; multiplying
the Hamiltonian by it changes the problem. Filling and couplings must share the
chosen normalization. Displacement keys denote Fourier moments in computational
coordinates, not automatically physical correlations on the transformed domain.

`h_int` accepts the existing density-density dictionary or
`BilinearInteraction(terms)`. A `BilinearTerm(g, A, B, displacement=R)` represents
`g sum_x : (c_x† A c_x)(c_{x+R}† B c_{x+R}) :`, with real g and Hermitian A, B.
An omitted displacement means onsite in the Hamiltonian's dimension. This covers
two onsite bilinears separated by R, not arbitrary products of bond bilinears.

Terms are additive. `(g,A,B,R)` and `(g,B,A,-R)` represent the same operator, so
listing both doubles its strength. Reversing R alone generally gives a different
term. There is no implicit one-half factor in g. In contrast, a density-density
dictionary stores both `V_R` and `V_-R = V_R.T`, with a one-half factor in its
ordered-pair sum. Hamiltonian dictionaries store both Fourier partner blocks.

For `rho_R,ij = <c_{x+R,j}† c_{x,i}>` and
`kappa_R,ij = <c_{x+R,j} c_{x,i}>`, a bilinear term's Wick energy per cell is

`g [Tr(A rho_0) Tr(B rho_0) - Tr(A rho_R B rho_-R)
    + Tr(kappa_R† A kappa_R B^T)]`.

The correction adds Hartree terms at zero, `-g A rho_R B` at R and
`-g B rho_-R A` at -R. Pairing adds `g A kappa_R B^T` at R and
`g B kappa_-R A^T` at -R. Contributions add when R=0. The map preserves
`Sigma_R = Sigma_-R†` and `Delta_R = -Delta_-R^T`. Required entries follow the
supports of A and B and existing Hermiticity, pairing and spatial constraints.
Energy and EDIIS curvature reuse one half of the correction contraction.

The normal density is `rho_ij = <c_j† c_i>`; observables contract as `Tr(O rho)`.
In electron-first BdG form, the upper-right block is `kappa_ij = <c_j c_i>`.
Density-density pairing uses `Delta_ij = V_ij kappa_ij`: positive V is repulsive
and negative V is attractive. Callable BdG assembly uses
`diag(h(k), -h(-k mod 2 pi)^T)`. Domain wrappers must preserve this momentum
reversal; it is not inferred from a callable.

A reference replaces rho and kappa by their differences inside the interaction
functional; a normal reference has zero pairing. The one-body term always uses
the actual density. Reference subtraction changes the model, not the energy zero.
For finite density-density models the interaction energy per orbital is

`sum_ij V_ij (dn_i dn_j - |drho_ij|² + |dkappa_ij|²) / (2N)`.

Periodic models sum over displacement blocks. Filling counts electrons per cell.
Energies and entropy are per cell per physical orbital: N for a normal model and
N for a 2N-dimensional BdG matrix. `free_energy = internal_energy - kT * entropy`,
with entropy in units of k_B. Fourier convention: `H(k) = sum_R H_R exp(-ik.R)`.

## Inputs, results and iteration

`DensityCoordinates` selects real-space entries. `DensityResult` stores computed
values, observables and diagnostics. Unrequested entries are unknown, never zero.
Standalone energy contractions require all their entries and reject callable
Hamiltonians, whose energy cannot be reconstructed from finitely many moments.
Result energies use `E0 = E_band - <Sigma_in, rho_out>` and
`E = E0 + E_int[rho_out - reference]`, valid also before self-consistency.

`SCFResult` adds the evaluated mean field, history and convergence status. Its mean
field reproduces its density. Applying `model.mean_field(result.density)` gives
the next correction, which agrees only at self-consistency. Failures expose the
last valid state through `exception.result` when available.

EDIIS is the default. It minimizes an energy surrogate over convex combinations
of retained densities, using exact quadratic interaction curvature. It compares
internal energy at zero temperature and free energy at finite temperature. Thermal
entropy is linearly interpolated between samples; this is not the exact free
energy of the mixture. Linear and Anderson mixing are explicit alternatives.
Convergence uses the largest active-density residual. No update is prepared after
the final allowed evaluation, and methods never switch automatically.

Thermal EDIIS computes entropy alongside density and reuses it in the result.
Other solvers and standalone density calls omit it by default.
`compute_free_energy=True` requests missing entropy on the final state, including
a valid failed result; that pass may repeat matrix work.

## Numerical stages

Fixed filling first solves `N(mu) = filling`, then evaluates density. The root
search caches charge samples and accepts any sample meeting the filling target;
`mu_tol` is the chemical-potential step tolerance. Charge-integration error and
density trace are separate quantities and never add root acceptance tests.

At fixed mu there is no root search or independent charge-error calculation.
Filling comes from available density information or remains `None`. Empty requests
skip unnecessary work. `density_filling` records charge on the density partition;
`filling` can instead report the preceding charge-root result.

- `FermiSimplex` integrates normal zero-temperature densities adaptively, refining
  charge before density with no return to charge refinement. Its default seed has
  five vertices per axis to avoid aliasing the first cosine harmonic.
- `UniformGrid` uses dense diagonalization or sparse finite-temperature AAA.
  Adaptive integration compares coarse and fine grids, starting at four points
  per axis. `nk` prescribes the total point count; `initial_nk` sets the starting
  count. Prescribed grids have no integration-error estimate.
- Sparse AAA requires prescribed grids for periodic systems. It approximates the
  Fermi function, factors each pole once per matrix and mu, and obtains requested
  inverse entries. Scalar fits can be reused; matrix factors are discarded after
  each node. Entropy reuses the density-selected poles.
- BdG requires UniformGrid. At zero temperature only fixed-mu evaluation is
  supported: occupation jumps can prevent a fixed-filling root from converging.
  Finite systems ignore grid settings with a warning. Callable root brackets use
  sampled spectra or an initial norm scale, expanding by charge evaluations.

All tolerance dependencies live in `errors.py`: `tol` for SCF, `tol/5` for density
and charge integration, `tol/10` for filling residual, and `tol/40` for matrix
functions. Explicit `ErrorTolerances` bypass the policy. Backends do not tighten
targets. Error estimates are empirical; energy and entropy have no stopping targets.

Known deferred issue: FermiSimplex `nk` currently selects a prescribed mesh. The
intended contract is an adaptive point budget. This change is outside this branch;
its tutorial uses tolerance-driven integration without `nk`.

## Energy comparisons and final observables

`scf/energy.py` constructs comparisons from density parameters, generating fields,
interaction energies, chemical potentials and density-partition fillings. History
never retains meshes. Differences are anchored to the oldest retained state,
without accumulating integration errors across iterations.

When integrated energies are cheap (UniformGrid and finite systems), use their
differences, subtracting kT times entropy at positive temperature. Extrapolate each
sample to target filling with `-mu*(N-N_target)/N_orbitals`. This is first-order,
and approximate away from self-consistency. The optimizer applies no further
filling correction. Entropy includes the BdG factor removing Nambu doubling.

Normal zero-temperature FermiSimplex instead uses endpoint density response:

`Delta E_int - <(Sigma_i+Sigma_j)/2, D_j-D_i>
 + (N_target-(N_i+N_j)/2)*(mu_j-mu_i)/N_orbitals`.

Canonical contractions preserve finite-range, reference and orbital conventions.
This approximation has cubic local error along a smooth path; it is neither an
absolute energy nor a certified error bound. Independent adaptive meshes are allowed.
Before optimization, shift comparison energies and scale both differences and
curvature by their largest magnitude so SLSQP resolves small differences. A constant
objective chooses the latest point.

Simplex absolute energy is deferred until success or a valid partial failure.
Only the current evaluation retains a mesh. On the finest complete evaluated
partition, including previews, integrate the spectral hinge
`g(k) = mean_n min(epsilon_n(k)-mu, 0)`. The degree-two simplex rule assigns each
vertex weight `1/((d+1)(d+2))` and the centroid weight `(d+1)/(d+2)`.
Exact coordinate keys reuse cached spectra and deduplicate centroids; missing
centroids require batched eigenvalues only. Add mu times the density trace per
orbital to recover band energy, then assemble the physical energy above.

This step does not refine or normalize density. Release the mesh afterward.
Deferred intermediate `SCFIteration.internal_energy` values are `None`, never
relative offsets. Density convergence does not certify absolute-energy accuracy.

## Code map and verification

- `hamiltonian.py`, `interaction.py`: callable Hamiltonians and bilinear terms.
- `model.py`, `meanfield.py`, `observables.py`: physical inputs and energy functionals.
- `density/`: filling search, integration and matrix functions.
- `scf/`: iteration, comparison energies and coefficient optimization.
- `space/`: selected entries and symmetry constraints.
- `tb/`, `interop/`: tight-binding algebra and optional Kwant conversion.

Repeatable tests check bilinear/intercell normal and BdG energies against exact
Fock-space expectations within `2e-12`; density derivatives check signs and double
counting. The spinless intercell gap is checked within `2e-7` against independent
512/1024-point gap equations agreeing within `2e-13`. Mapped Dirac disk density
has a `2e-4` acceptance threshold at `tol=5e-4` (observed error `6.64e-5`).

Smooth gapped response tests include finite range and reference subtraction:
observed error at most `1.66e-8` at the smaller field step, with approximately
cubic convergence; dense references agree within `2e-13`. Thermal normal/BdG
free-energy differences agree within `3e-13`, with 128/256-point references
agreeing within `2e-12`. Lifecycle tests cover final energy, failure results and
unchanged density; simplex quadrature tests include dimensions one through four.

The executable graphene tutorial demonstrates general interactions and the
concentric disk map at the four distinct parameter points of Fig. 2(b-f).
Every point uses the same random-guess seed and scale, without named phase seeds,
and the same tolerances: SCF `5e-3`, density `1e-3`, charge integration `2e-2`,
filling `5e-3`. It does not claim to select the paper's five ordered branches.
The four columns are labeled by Hamiltonian parameters, without repeating the
shared (b)/(c) point. The full Hamiltonian allows valley mixing. MeanFi construction
and solver calls, disk coordinates and bilinear interaction construction are
executable definitions in the notebook. Only the Hamiltonian physics and plotting
remain imported helpers. Numba compilation is shown inline
and required only by this tutorial and the documentation environment, not MeanFi.
There are no tutorial threadpool controls. Generated reports and exploratory
performance experiments stay outside the MR.
