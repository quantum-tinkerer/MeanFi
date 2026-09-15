# Defaults and capabilities

When `integration=None`, MeanFi uses:

| System | Default |
| --- | --- |
| Dense normal, `kT=0` | `AdaptiveSimplex()` |
| Dense normal or BdG, `kT>0` | `PeriodicGrid()` with direct diagonalization and global refinement |
| BdG, `kT=0` | Explicit `PeriodicGrid(nk=...)` required |
| Sparse, automatic finite-temperature selection | Error with migration guidance; choose an explicit supported method |

| Family and mode | Normal, `kT=0` | BdG, `kT=0` | Normal/BdG, `kT>0` |
| --- | --- | --- | --- |
| `AdaptiveSimplex`, prescribed or accuracy-controlled | Yes | No | No |
| `PeriodicGrid(nk=...)` | Yes | Yes | Yes |
| `PeriodicGrid()` or explicit integration targets | No | No | Yes |

Direct diagonalization is the main periodic path. Fixed periodic grids also
support explicit `RationalFOE` for sparse matrices at positive temperature. No adaptive rational path
is provided. Both explicit `RationalFOE()` and implicit prescribed sparse selection use AAA. Choosing dense evaluation for a sparse input must be explicit.

`Model` defaults to `kT=0.0`. `solver` uses `EnergyDIIS()` for all supported
normal and BdG calculations. At finite temperature, EDIIS uses a free-energy
history bound and bounded Anderson mixing finishes convergence or handles
stagnation. An explicit `scf=` selects another method. All SCF settings are
keyword-only.
The top-level `tol`
provides a convenient shared accuracy policy, while `scf_tol`, `filling_tol`,
`mu_tol`, `density_matrix_tol`, `charge_tol`, `energy_tol` and `entropy_tol`
separate individual budgets. Energy and entropy estimates are in `result.errors`.
Integration targets are populated only after the prescribed/accuracy-controlled
mode has been resolved. An explicit `nk` always retains prescribed-size semantics.

Every density backend returns entropy and the expectation of the input
quadratic Hamiltonian as `band_energy`. SCF results report interaction-corrected
`internal_energy` and `free_energy`, with entropy in units of Boltzmann's
constant. All these quantities are per cell per physical orbital. Dense periodic evaluation reuses eigenvalues; sparse AAA evaluation
shares poles and matrix factorizations between density and entropy.
