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
  and known internal/free energies. Default selected model results support energy
  helpers without extra matrix entries; arbitrary observable support remains explicit.
- Entry errors consistently describe integrated real-space density. FermiSimplex
  applies its global worst-entry estimate to each entry; UniformGrid has separate
  coarse/fine entry estimates. These remain empirical estimates, not certificates.

## Still open

- Adaptive AAA integration: currently requires a prescribed mesh. Its matrix-function
  estimate does not establish Brillouin-zone integration accuracy.
- Finite-system selection: momentum-method compatibility is still resolved before
  finite-system evaluation; prescribed finite calculations retain prescribed error
  semantics. A dedicated finite-system contract could simplify this later.
- Sparse backend defaults: automatic finite-temperature sparse selection requires
  an explicit supported method. Choosing dense evaluation must remain explicit.
- Further consolidation of physical-input validation and unsupported-configuration
  errors, and of public-call versus integration-setting tolerance overrides.
- Entropy approximation remains a diagnostic available for AAA only. A universal
  thermodynamic-error bound would need further analysis; no such target is implied.

EDIIS remains an internal-energy optimizer with no automatic switching. Entropy
introduces no accuracy target or stopping criterion.
