# `MeanFi`

`MeanFi` is a Hartree-Fock solver for tight-binding models with density-density interactions.
It exposes physics through `Model`, Brillouin-zone integration through explicit `integration=` methods, and outer fixed-point updates through explicit `scf=` methods.

## Workflow

`MeanFi` follows a simple three-step workflow:

1. Define the interacting problem with `Model`.
2. Build a mean-field guess on the interaction keys.
3. Run `solver(...)` with an explicit integration method to obtain a self-consistent mean-field Hamiltonian.

```python
import meanfi

h_0 = {(0,): onsite, (1,): hopping, (-1,): hopping.T.conj()}
h_int = {(0,): onsite_interaction}

model = meanfi.Model(h_0, h_int, filling=2, kT=0.05)
integration = meanfi.AdaptiveQuadrature(density_matrix_tol=1e-4)
scf = meanfi.LinearMixing(max_iterations=80, alpha=0.5)

guess = meanfi.guess_tb(frozenset(h_int), onsite.shape[0])
result = meanfi.solver(
    model,
    guess,
    integration=integration,
    scf=scf,
    scf_tol=1e-4,
)
h_mf = meanfi.add_tb(h_0, result.mf)
density_matrix = result.density_matrix_result.density_matrix
```

## Density APIs

- `meanfi.density_matrix(...)`
  Computes the fixed-filling density matrix and returns a `DensityMatrixResult`.
- `meanfi.density_matrix_at_mu(...)`
  Computes the density matrix at an explicit chemical potential and returns a `DensityMatrixResult`.

Both APIs require an explicit integration method:

- `meanfi.AdaptiveQuadrature(density_matrix_tol=...)` for `kT > 0`
- `meanfi.AdaptiveSimplex(density_matrix_tol=..., max_refinements=...)` for `kT = 0`
- `meanfi.UniformGrid(nk=...)` for the zero-temperature fixed-grid path

The main public numerical knobs are:

- `density_matrix_tol` on adaptive integration methods
- `scf_tol` on `meanfi.solver(...)`

Advanced fixed-filling controls remain available on `density_matrix(...)` and `solver(...)`:

- `filling_tol`
- `mu_tol`
- `max_mu_iterations`

`DensityMatrixResult` exposes fully explicit public names such as `density_matrix`, `density_matrix_error`, `filling`, `target_filling`, `filling_residual`, and `info`.

## Optimizers

`meanfi.solver(...)` separates SCF method selection from integration method selection.
Built-in SCF methods are `meanfi.LinearMixing(...)` and `meanfi.AndersonMixing(...)`.
For harder self-consistent problems you can still pass an external optimizer explicitly, for example `scipy.optimize.anderson`:

```python
from scipy.optimize import anderson

result = meanfi.solver(
    model,
    guess,
    integration=integration,
    optimizer=anderson,
    optimizer_kwargs={"M": 0, "line_search": "wolfe"},
)
```

## Installation

Until the next packaged release, install from a checkout so the Git-backed quadrature dependency is available:

```bash
python -m pip install "stateful-quadrature @ git+https://github.com/Kostusas/stateful_quadrature.git"
python -m pip install -e .
```

For local development with Pixi:

```bash
pixi install
```

If you want the `kwant` conversion helpers as well:

```bash
python -m pip install ".[kwant]"
```

## Scope

Current support includes:

- density-density interactions,
- finite- and zero-temperature mean-field calculations,
- experimental finite-temperature BdG superconducting mean-field calculations derived internally from electron-block `h_int`,
- adaptive and fixed-grid Brillouin-zone integration methods,
- tight-binding dictionary workflows,
- optional `kwant` conversion helpers.

Not supported in the main package:

- zero-temperature BdG superconducting calculations.

For `kT = 0`, the Brillouin zone is treated as a torus mathematically. The simplicial backend keeps duplicated seam vertices rather than identifying opposite faces in the cache, but it starts from a seam-safe `2^d` partition of the fundamental cell so no root simplex spans opposite faces. Setting `AdaptiveSimplex(max_refinements=0)` keeps only that root mesh, which is the closest analogue to a very coarse fixed `k` grid and does not provide a density-matrix error indicator.
