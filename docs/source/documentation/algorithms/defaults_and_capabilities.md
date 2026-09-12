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
is provided. Choosing dense evaluation for a sparse input must be explicit.

`Model` defaults to `kT=0.0`; `solver` uses `AndersonMixing()`. The top-level `tol`
provides a convenient shared accuracy policy, while `scf_tol`, `filling_tol`,
`mu_tol`, `density_matrix_tol` and `charge_tol` separate individual budgets.
Integration targets are populated only after the prescribed/accuracy-controlled
mode has been resolved. An explicit `nk` always retains prescribed-size semantics.

FermiSimplex provides the normal zero-temperature band-energy calculation used
by energy-based SCF. Finite-temperature density integration does not provide a
thermodynamic energy or free-energy method; periodic results report unavailable
energy as `None`. Use the supported residual-based SCF methods for those workflows.
