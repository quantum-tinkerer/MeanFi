# `MeanFi`

`MeanFi` is a finite-temperature Hartree-Fock solver for tight-binding models with density-density interactions.
The package now uses one integration backend throughout: `stateful_quadrature`.

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
    charge_tol=1e-6,
    density_atol=1e-6,
    scf_tol=1e-5,
)

guess = meanfi.guess_tb(frozenset(h_int), onsite.shape[0])
mf_correction = meanfi.solver(model, guess)
h_mf = meanfi.add_tb(h_0, mf_correction)
```

## Density APIs

- `meanfi.density_matrix(...)`
  Computes the fixed-filling density matrix. The chemical potential is solved internally with safeguarded Newton and stateful cubature reuse.
- `meanfi.density_matrix_at_mu(...)`
  Computes the density matrix at an explicit chemical potential.

Both APIs return quadrature error estimates and runtime statistics.

## Numerical model

The supported regime is finite temperature only. The package computes:

- charge integrals with a scalar stateful-cubature solve for `N(\mu)=\nu`,
- density integrals with a second pass that reuses the adaptive cache,
- outer SCF updates with Anderson mixing by default.

## Installation

```bash
pip install meanfi
```

`meanfi` depends on `stateful-quadrature` as a normal package dependency.

## Scope

Current mainline support includes:

- density-density interactions,
- finite-temperature mean-field calculations,
- tight-binding dictionary workflows,
- `kwant` conversion helpers.

Not supported in the main package:

- zero-temperature production solvers,
- k-grid based mean-field solvers,
- superconducting pairing terms.
