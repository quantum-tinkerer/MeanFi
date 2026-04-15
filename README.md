# `MeanFi`

`MeanFi` is a finite-temperature Hartree-Fock solver for tight-binding models with density-density interactions.
Fixed-filling density updates are evaluated with adaptive quadrature and cached eigensystem reuse via `stateful_quadrature`.

## Workflow

`MeanFi` follows a simple three-step workflow:

1. Define the interacting problem with `Model`.
2. Build a mean-field guess on the interaction keys.
3. Run `solver(...)` to obtain a self-consistent mean-field Hamiltonian.

```python
import meanfi

h_0 = {(0,): onsite, (1,): hopping, (-1,): hopping.T.conj()}
h_int = {(0,): onsite_interaction}

model = meanfi.Model(
    h_0,
    h_int,
    filling=2,
    kT=0.05,
)

guess = meanfi.guess_tb(frozenset(h_int), onsite.shape[0])
mf_correction = meanfi.solver(model, guess)
h_mf = meanfi.add_tb(h_0, mf_correction)
```

## Density APIs

- `meanfi.density_matrix(...)`
  Computes the fixed-filling density matrix. The chemical potential is solved internally with safeguarded Newton steps and bisection fallback while reusing the adaptive quadrature cache.
- `meanfi.density_matrix_at_mu(...)`
  Computes the density matrix at an explicit chemical potential.

Both APIs return error estimates together with runtime statistics.

## Optimizers

`meanfi.solver(...)` now defaults to internal linear mixing, so the runtime package does not require SciPy.
For harder self-consistent problems you can pass an external optimizer explicitly, for example `scipy.optimize.anderson`:

```python
from scipy.optimize import anderson

mf_correction = meanfi.solver(
    model,
    guess,
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
- finite-temperature mean-field calculations,
- tight-binding dictionary workflows,
- optional `kwant` conversion helpers.

Not supported in the main package:

- zero-temperature production solvers,
- k-grid based self-consistent solvers,
- superconducting pairing terms.
