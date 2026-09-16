# Defaults and capabilities

When `integration=None`, MeanFi uses:

| System | Default |
| --- | --- |
| Dense normal, `kT=0` | `FermiSimplex()` |
| Dense normal or BdG, `kT>0` | `UniformGrid()` with direct diagonalization and global refinement |
| Periodic BdG, `kT=0` | Explicit `UniformGrid(nk=...)` required |
| Dense finite BdG, `kT=0` | `UniformGrid()` with direct diagonalization |
| Sparse, automatic selection | Error with migration guidance; choose an explicit supported method |

| Family and mode | Normal, `kT=0` | BdG, `kT=0` | Normal/BdG, `kT>0` |
| --- | --- | --- | --- |
| `FermiSimplex`, prescribed or accuracy-controlled | Yes | No | No |
| `UniformGrid(nk=...)` | Yes | Yes | Yes |
| `UniformGrid()` or explicit integration targets | No | No | Yes |

Direct diagonalization is the main periodic path. Fixed periodic grids also
support explicit `RationalFOE` for sparse matrices at positive temperature. No adaptive rational path
is provided. Both explicit `RationalFOE()` and implicit prescribed sparse selection use AAA. Choosing dense evaluation for a sparse input must be explicit.

`Model` defaults to `kT=0.0`. `solver` uses `EnergyDIIS()` for all supported
normal and BdG calculations. EDIIS minimizes internal energy over its density history and never
switches methods. An explicit `scf=` selects another method. All SCF settings are
keyword-only, as are integration and matrix-function settings.
The top-level `tol`
provides a convenient shared accuracy policy, while `scf_tol`, `filling_tol`,
`mu_tol`, `density_matrix_tol` and `charge_tol` separate individual budgets.
An explicit `density_matrix_tol` also supplies an omitted `charge_tol`; users
may override charge accuracy independently. The tolerance policy also sets
`matrix_function_tol=tol/5` for approximating the Fermi matrix function; change
this budget by returning a modified `ErrorTolerances` from that same policy.
Energy and entropy estimates in
`result.errors` are diagnostics, not targets.
Integration targets are populated only after the prescribed/accuracy-controlled
mode has been resolved. For periodic systems, explicit `nk` retains prescribed-size semantics. Finite
systems use the same methods, ignore `nk` and `initial_nk` with a warning, and
have zero integration error. Finite BdG calculations do not require `nk`.
At positive temperature, explicit sparse `UniformGrid()` chooses AAA for finite
systems; periodic AAA still requires `nk`.

Every density backend returns optional entropy and the expectation of the input
quadratic Hamiltonian as `band_energy`. Model-based density results also retain
known interaction-corrected `internal_energy` and `free_energy`, including the
default selected results. SCF results report interaction-corrected
`internal_energy` and `free_energy`, with entropy in units of Boltzmann's
constant. All these quantities are per cell per physical orbital. Dense periodic evaluation reuses eigenvalues; sparse AAA evaluation
shares poles and matrix factorizations between density and entropy.

`compute_free_energy=True` defaults to computing entropy in standalone density
calls and once after SCF termination. SCF iterations skip entropy. Disabling it
leaves entropy, free energy and `errors.entropy` as `None`. There is no entropy
tolerance; the common error field is also `None` if its total error cannot be
estimated, including prescribed periodic meshes.
