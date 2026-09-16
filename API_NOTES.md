# API review status

Updated 2026-09-16. The accuracy and result changes below implement the agreed
review; the remaining items are a backlog, not new API promises.

## Implemented

- One tolerance policy, now including `matrix_function_tol=tol/5` by default,
  and one `ErrorValues` record for all methods. AAA reports its measured
  `matrix_function_error`; unavailable estimates remain `None`.
- Independent charge targets remain available. Fixed-mu AAA calls no longer
  inherit unused filling-root constraints.
- All method settings are keyword-only. Both integrators distinguish prescribed
  `nk` from adaptive `initial_nk`, with documented total-node rounding rules.
- UniformGrid uses coarse/fine comparisons only. Prescribed meshes require no
  additional grids and report no integration estimate.
- All integrators return `IntegrationInfo`, with optional method-specific details.
  Documentation distinguishes local simplex refinements, global grid doublings,
  final points, cached nodes and cumulative evaluations.
- SCF convergence uses reconstructed complex density-entry errors for every method.
- Models own temperature and filling. Density results retain evaluation temperature
  and known internal/free energies. Selected results expose stored energies as properties; observable helpers evaluate
  trial densities and require all contraction entries.
- Entry errors consistently describe integrated real-space density. FermiSimplex
  applies its global worst-entry estimate to each entry; UniformGrid has separate
  coarse/fine entry estimates. These remain empirical estimates, not certificates.

- Entropy defaults to one final SCF evaluation, even on failure with a valid state;
  `compute_free_energy=False` skips it. All methods expose `errors.entropy`.
- Finite systems share integrators, warn and ignore mesh sizes, and have zero
  integration error. Sparse-to-dense evaluation requires an explicit method.
- Result records retain no hidden Model; SCF scalars delegate to the density.
  Invalid filling and noninteger coordinates are rejected before numerical work.

- Accuracy controls live in `ErrorTolerances`, passed through `tol`, with numeric
  shorthand and the existing policy callback. Integration objects have no accuracy
  overrides. Charge accuracy remains independently adjustable.
- Physical boundaries reject complex interaction coefficients and malformed normal
  or BdG corrections. Unsupported methods fail before numerical work.
- `Model.mean_field(density)` is the public correction operation for all models.
  `SCFResult.mean_field` is the input that produced the returned density, including
  partial results; the next correction is explicit.
- Normal and BdG corrections both allow omitted zero partner blocks. Energy
  contractions use the same canonical correction representation.
- Spatial symmetries validate the lattice map and combined Fourier unitarity;
  roundoff in a global phase no longer removes valid density directions.
- Accepted chemical-potential guesses skip bracket construction and endpoint solves.
  Tall symmetry systems use economical SVD without losing wide-system null directions.

## Still open

- Adaptive AAA integration: currently requires a prescribed mesh. Its matrix-function
  estimate does not establish Brillouin-zone integration accuracy.

EDIIS remains an internal-energy optimizer with no automatic switching. Entropy
introduces no accuracy target or stopping criterion.
