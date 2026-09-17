# MeanFi design

MeanFi solves normal-state tight-binding or callable Bloch Hamiltonians with
density-density or local bilinear interactions. Tight-binding density-density
models also support superconducting pairing. Its core loop is

`density -> interaction correction -> Hamiltonian -> density at fixed filling -> mixing`.

Compute only the requested density entries and the work needed for convergence.
Numerical targets are explicit; unavailable results and error estimates are `None`.

## Physical model

`Model` owns the Hamiltonian, interaction, filling, temperature, optional reference
and spatial symmetries. It validates stored matrices once and checks callable
outputs as they are evaluated. Tight-binding inputs retain dense or sparse storage. Public matrix containers share read-only arrays while keeping structural
mutations outside the model. Use a new model or `dataclasses.replace` for changes.

`h_0` is a tight-binding dictionary or `BlochHamiltonian(function)`.
The callable takes separate momentum coordinates in radians, e.g. `h(kx, ky)`.
Its required positional arguments determine dimension; one evaluation at the
origin determines the orbital count. Both are inferred read-only attributes,
not constructor inputs. Bind physical parameters in a closure or partial; an
ambiguous signature with variable or optional arguments is rejected. The callable
returns a finite Hermitian dense matrix whose size remains fixed. Its captured parameters must remain fixed
throughout a calculation. Integration uses the normalized BZ measure
`d^d k / (2 pi)^d` on `[0, 2 pi]^d`, including endpoints for simplex integration.
There are no domain objects or implicit Jacobians. Users can compose a continuum
Hamiltonian with an equal-measure coordinate map; for example,
`r = cutoff * sqrt(k1 / (2 pi)), theta = k2` gives a normalized disk average.
A varying Jacobian needs weighted integration, which is not supported by this
contract. It cannot be encoded by multiplying the Hamiltonian by the Jacobian.
Filling, density and interaction couplings must use the same normalization.
Nonzero displacement keys then denote Fourier moments in computational coordinates,
not automatically physical real-space correlations.

`h_int` retains the existing density-density dictionary format. Alternatively,
`BilinearInteraction(terms)` holds `BilinearTerm(g, A, B)` terms for
`g : (c† A c)(c† B c) :`, with real g and Hermitian operators in the full orbital
space. Embedding operators selects any orbital subset; no rank-four interaction
tensor or separate continuum SCF engine is needed. These local, normal-state terms
use the onsite density rho and the Wick functional

`E_int = sum g [Tr(A rho) Tr(B rho) - Tr(A rho B rho)]`,

`Sigma = sum g [Tr(B rho) A + Tr(A rho) B - A rho B - B rho A]`.

There is no implicit factor of one half in g. Both energies and EDIIS curvature
reuse `E_int = Tr(Sigma rho) / 2`, divided by the orbital count for reported
energies. Reference subtraction applies before this same map. Required density
entries are the orbital blocks touched by each nonzero term, reduced by existing
Hermiticity and spatial constraints. Callable Hamiltonians and bilinear terms
currently support normal states only; dense diagonalization works with both
FermiSimplex at zero temperature and UniformGrid at positive temperature.

The normal density is `rho_ij = <c_j† c_i>`; observables contract as `Tr(O rho)`.
In electron-first BdG form, the upper-right density block is
`kappa_ij = <c_j c_i>`. The interaction map gives Hartree/Fock terms and
`Delta_ij = V_ij kappa_ij`. Positive V is repulsive; negative V is attractive.

A reference replaces rho and kappa by their differences from that reference
inside the interaction functional. A normal reference has zero pairing.
For a finite density-density model, the interaction energy per orbital is

`sum_ij V_ij (dn_i dn_j - |drho_ij|² + |dkappa_ij|²) / (2N)`.

Periodic systems use the same contractions over displacement blocks. The one-body
energy always uses the actual density. Reference subtraction changes the
interaction model; it does not return an energy difference from the reference.

Filling counts electrons per cell. Energies and entropy are per cell per physical
orbital: N for a normal model and N for a 2N-dimensional BdG matrix.
`free_energy = internal_energy - kT * entropy`, with entropy in units of k_B.
The Fourier convention is `H(k) = sum_R H_R exp(-ik.R)`.

## Inputs, results and iteration

`DensityCoordinates` selects real-space entries. `DensityResult` stores the computed
values, observables and diagnostics. Unrequested entries are unknown, never zero.
Standalone energy functions require every entry used by their contraction.
For callable Hamiltonians, finitely many integrated density entries do not
reconstruct the bare one-body energy. Use the density/SCF result's retained
energy, computed as `E_band - <Sigma_in, rho_out> + E_int[rho_out - reference]`.
This expression is valid away from self-consistency and supplies EDIIS with the
same quadratic functional. Standalone energy contractions reject callable
Hamiltonians. No new energy refinement or energy-accuracy guarantee is added.

`SCFResult` adds the evaluated mean field, history and convergence status. Its mean
field reproduces its density. Applying `model.mean_field(result.density)` gives
the next correction, which agrees only at self-consistency. Failures expose the
last valid state through `exception.result` when available.

EDIIS is the default. It minimizes internal energy over a convex density history
and never switches methods automatically. Convergence measures the largest complex
density-entry residual. Linear and Anderson mixing are explicit alternatives.
No update is prepared after the last allowed evaluation.

Entropy and free energy are omitted by default. `compute_free_energy=True` adds
entropy to a density calculation or evaluates it once when SCF terminates,
including a valid failed result. This final pass may repeat matrix work.

## Numerical stages

Fixed filling first solves `N(mu) = filling`, then evaluates density. The root
search caches charge samples and accepts any sample meeting the filling target;
`mu_tol` is the chemical-potential step tolerance. Charge-integration error and
density trace are separate quantities and never add root acceptance tests.

At fixed mu, there is no root search or independent charge-error calculation.
Filling comes from already available density information, or remains `None`.
Energy is computed only when available without extra matrix work or needed by
the requested calculation. Empty requests skip unnecessary numerical work.

- `FermiSimplex` integrates normal zero-temperature densities adaptively. It refines
  charge before density, with no return to charge refinement. Its default seed has
  five vertices per axis to avoid aliasing the first cosine harmonic.
- `UniformGrid` uses dense diagonalization or sparse finite-temperature AAA.
  Adaptive integration compares coarse and fine grids, starting at four points
  per axis. `nk` prescribes the total point count; `initial_nk` sets the adaptive
  starting count. Prescribed grids have no integration-error estimate.
- Sparse AAA requires a prescribed grid for periodic systems. It approximates the
  Fermi function on a spectral interval, factors each pole once per matrix and mu,
  and obtains only requested inverse entries. Scalar fits can be reused; matrix
  factors are discarded after each node. Optional entropy reuses the density poles.
- Zero-temperature UniformGrid supports fixed mu only; fixed-filling calls raise
  `NotImplementedError` because the root search cannot generally resolve occupation
  jumps. Finite systems need no grid and warn about irrelevant grid settings.

All tolerance dependencies live in `errors.py`. The default policy assigns `tol`
to SCF, `tol/5` to density and charge integration, `tol/10` to filling residual,
and `tol/40` to matrix-function approximation. Explicit `ErrorTolerances` bypass
the policy. Backends do not tighten targets. Integration and scalar-fit estimates
are empirical indicators; energy and entropy have no stopping targets.

## Code map and verification

- `hamiltonian.py`, `interaction.py`: callable Hamiltonians and local bilinear terms.
- `model.py`, `meanfield.py`, `observables.py`: physical inputs, interaction map and energy.
- `density/`: filling search, momentum integration and matrix functions.
- `scf/`: iteration methods and their shared evaluation loop.
- `space/`: selected entries, Hermiticity, pairing antisymmetry and spatial constraints.
- `tb/`, `interop/`: tight-binding algebra, Fourier transforms and optional Kwant conversion.

Tests compare with exact Fock-space states, analytic models and converged dense
references; work-count tests guard against unnecessary calculations. Benchmarks
measure density and complete fixed-filling/SCF calls alongside reference errors.
Tutorials execute fresh calculations using default EDIIS and tolerance policy.
The bilinear tests compare exact Fock-space energies within `2e-12` and check the
energy derivative and EDIIS quadratic identity. The continuum example compares a
mapped Dirac disk density with its analytic normalized average: maximum entry
error `6.64e-5` at `tol=5e-4`, below the test's `2e-4` acceptance threshold.
Callable/TB equivalence tests include complex Fourier moments, fixed filling,
reference subtraction and finite-temperature SCF.
Generated reports, builds and plots stay outside version control.
