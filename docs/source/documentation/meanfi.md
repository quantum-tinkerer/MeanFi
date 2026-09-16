# Package reference

{download}`Download the executable API walkthrough <../../../examples/api_walkthrough.py>`

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: mean_field, hamiltonian_from_density, hamiltonian_from_meanfield, random_meanfield
```

`Model(..., reference=reference)` enables reference-state subtraction for normal
and superconducting calculations. The effective Hamiltonian is built as
`h_0 + W[rho - rho_ref]`, where `W` is the complete density-density mean-field
correction, including both Hartree and exchange-like terms. `reference` accepts
a complete density dictionary or a `DensityResult`. Selected results need only
the interaction's required entries; every required coordinate is validated when
the model is constructed. Dictionaries are copied into read-only storage.
Missing entries are never interpreted as zeros.

If the reference should be the non-interacting density at a chosen filling,
compute it explicitly and pass the result to the model:

```python
reference = meanfi.density_matrix(
    h_0,
    filling=reference_filling,
    kT=kT,
    interaction=h_int,
    tol=tol,
)
model = meanfi.Model(
    h_0,
    h_int,
    filling=filling,
    kT=kT,
    reference=reference,
)
```

`density_matrix(model)` uses the model's filling, temperature and required
coordinates. Both normal and superconducting models use this API; optional
`mean_field=correction` evaluates an interacting Hamiltonian. The same model
support is available in `density_matrix_at_mu(model, mu)`. A Model owns its
physical parameters: passing a separate `kT` or `filling` with a Model raises.
Use `dataclasses.replace(model, kT=..., filling=...)` to change them. Every
`DensityResult` records the evaluation temperature as `kT`.

`DensityCoordinates` is an address list: each entry is a displacement, row and
column. It contains no density values. `DensityResult` is the answer returned
by a calculation: values at those addresses, plus chemical potential, filling,
energies and numerical diagnostics. Most users can request full blocks with
`keys` and never construct a coordinate object:

```python
import numpy as np
import meanfi as mf

full = mf.density_matrix_at_mu(
    {(0,): np.diag([-1.0, 1.0])},
    mu=0.0, kT=0.2, keys=[(0,)],
)
rho = full.to_tb()[(0,)]  # The usual 2-by-2 density matrix.
```

Coordinates are useful when only a few entries of a large matrix are needed:

```python
occupations = mf.DensityCoordinates.from_entries(
    size=2, keys=[(0,)],
    entries=(((0,), 0, 0), ((0,), 1, 1)),
)
selected = full.select(occupations)
print(selected.values.real)  # [0.99330715, 0.00669285]
```

The selected result contains two values, not a complete matrix. Its missing
entries are unknown, so `selected.to_tb()` raises. To compute only those values
from the start, pass `coordinates=occupations` instead of `keys` to either
density function. Model-based calls select the entries needed by the interaction
by default; use `keys` explicitly for complete blocks.

For a Hamiltonian dictionary, supply exactly one of `keys`, `coordinates` or
`interaction`. `keys` requests complete blocks, while the other two options
request selected entries. `result.to_tb()` returns a dictionary of complete
blocks; `result.to_tb(sparse=True)` returns CSR blocks.
Selected results expose their read-only `coordinates`, `values`, and optional
per-entry `entry_errors`; converting
one to complete matrix blocks raises instead of filling uncomputed entries with
zeros. `result.select(coordinates)` selects both values and entry errors while
preserving temperature, chemical potential, filling, all computed energies, entropy,
and integration statistics.

`DensityResult` is returned by the density functions. Read its `coordinates`,
`values`, `entry_errors`, and physical quantities directly. The shared storage
payload is private. Manually supplied reference densities can be ordinary
matrix dictionaries:

```python
reference = {(0,): np.diag([0.5, 0.5])}
model = mf.Model(h_0, h_int, filling=1.0, reference=reference)
required = model.required_coordinates
```

## Mean-field and density matrix

`model.mean_field(density)` computes the correction from the required density
entries, including reference subtraction and superconducting pairing. It replaces
the former top-level normal-only `meanfield` function. Add it to the bare
Hamiltonian with `model.hamiltonian_from_density(density)`. The correction stored
as `solution.mean_field` is the input that produced `solution.density`; it differs
from `model.mean_field(solution.density)` before self-consistency is reached.

```{eval-rst}
.. autofunction:: meanfi.density_matrix
```

```{eval-rst}
.. autofunction:: meanfi.density_matrix_at_mu
```

```{eval-rst}
.. autofunction:: meanfi.fermi_dirac
```

```{eval-rst}
.. autoclass:: meanfi.DensityResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.ErrorTolerances
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.ErrorValues
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.IntegrationInfo
   :show-inheritance:
```

All methods return the same error and statistics records. An unavailable or
inapplicable estimate is `None`; a prescribed `nk` does not trigger extra grids
for error estimation. For density integration, inspect
`errors.density_matrix_integration`; for AAA's matrix-function approximation,
inspect `errors.matrix_function_error`. The existing tolerance policy controls
both independently. All three calculation functions accept the same `tol`: a
number for the default policy, or a complete `ErrorTolerances` record.

```python
from dataclasses import replace

requested = replace(
    meanfi.default_solver_tolerances(1e-5),
    charge_integration=1e-4,       # Independent charge-integration budget.
    filling_residual=1e-7,
    mu_tol=1e-12,                 # Energy units; not a filling-error estimate.
)
solution = meanfi.solver(model, guess, tol=requested)
density = meanfi.density_matrix(model, tol=requested)
```

An explicit record uses its fields directly and cannot be combined with a custom
policy. In a newly constructed record, omitted `charge_integration` follows
`density_matrix_integration`. When replacing an existing record, set
`charge_integration=None` to follow a changed density target again. For repeated
calculations with a custom numeric scaling rule, a policy remains available:

```python
from dataclasses import replace


def accuracy(tol):
    return replace(
        meanfi.default_solver_tolerances(tol),
        charge_integration=tol
    )


density = meanfi.density_matrix(model, tol=1e-5, tolerance_policy=accuracy)
```

The default matrix-function target is `tol/40` for every calculation. The policy
depends only on `tol`; backends use its targets unchanged. See
[accuracy and diagnostics](algorithms/accuracy.md).

`Model` validates finite filling and temperature, matching matrix sizes and
lattice dimensions, Hermiticity, and real density-density interaction coefficients. It owns read-only copies of dense or sparse
input matrices and symmetry data. Use a new model (or `dataclasses.replace`) to
change model parameters.

## Solvers

`EnergyDIIS()` is the default for normal and BdG models with every supported
integration backend. It minimizes the internal energy of convex density
combinations, using the exact quadratic mean-field interaction. Entropy and
free energy do not participate in the coefficient optimization.
EDIIS runs only its own update and raises `NoConvergence` on iteration
exhaustion. Users can explicitly restart with another method using
`model.mean_field(failure.result.density)`; see {ref}`the SCF overview <one-solve>`.
Physical free energy need not fall on every iteration.

Sparse `RationalFOE()` uses AAA at positive temperature on a prescribed
`UniformGrid(nk=...)`. Density determines the poles. Entropy uses a subsequent residue fit on those
poles and reports its approximation error; it does not affect pole selection.

SCF settings are keyword-only. To change the history or iteration budget, pass
`scf=meanfi.EnergyDIIS(history_size=6, max_iterations=100)`. Explicit
`AndersonMixing` and `LinearMixing` remain available.


```{eval-rst}
.. autofunction:: meanfi.solver
```

```{eval-rst}
.. autoclass:: meanfi.SCFResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.SCFIteration
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.ConvergenceError
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.SolverError
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.SolverFailure
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.NoConvergence
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: meanfi.observables
   :members: expectation_value, trial_internal_energy, trial_free_energy
   :show-inheritance:
```

Final SCF results report internal energy per cell per physical orbital; optional
entropy and free energy use the same normalization. Iteration history records internal energy only. Entropy is in units of Boltzmann's constant, so
`free_energy = internal_energy - model.kT * entropy`. This is Helmholtz free
energy at fixed electron filling. The chemical-potential term is not subtracted.
All thermodynamic totals are divided by N for N physical orbitals in the unit
cell. A BdG Hamiltonian has size 2N; its physical totals still divide by N after
removing Nambu doubling. Spin components count as separate orbitals.

| Quantity | Convention |
| --- | --- |
| `internal_energy`, `free_energy`, `band_energy` | Energy per cell per physical orbital |
| `entropy` | Entropy / k_B per cell per physical orbital |
| `errors.band_energy_integration`, `errors.entropy` | Same normalized units as the corresponding quantity |
| `filling` | Electrons per cell, from 0 to N |
| `mu`, `kT` | Single-particle energy units |
| Density entries | Occupations and coherences, without normalization by N |
| `expectation_value(density, observable)` | Unnormalized observable trace per cell |

Multiplying an energy or entropy result by N recovers its physical total per
cell. The generic `expectation_value` remains a trace: for example, the identity
operator gives the electron count for a normal density. Divide that trace by N
when an observable per orbital is wanted.

Compare converged solutions at the same filling and temperature using their
free energies; SCF convergence alone does not establish a global minimum.

```python
solution = meanfi.solver(
    model, model.random_meanfield(rng=0, scale=0.1), compute_free_energy=True
)
print(solution.internal_energy, solution.entropy, solution.free_energy)
```

`expectation_value` and `trial_internal_energy` accept complete tight-binding
matrix dictionaries or `DensityResult` objects. Default model-based results
retain internal energy from the evaluated band energy and interaction correction,
so energy evaluation needs no additional density entries:

```python
density = meanfi.density_matrix(model, mean_field=solution.mean_field)
print(density.internal_energy)
```

Retained energies belong to the evaluated state and model; selecting fewer
entries preserves them. A Hamiltonian-only result has no interaction model and
therefore reports `internal_energy=None` and `free_energy=None`. Explicit model
selections omitting required interaction entries may also leave these fields
unknown. An arbitrary external correction outside the interaction space needs
sufficient additional entries to recover the model's energy.

Use `trial_internal_energy(model, density)` or `trial_free_energy(model, density)`
to independently evaluate a trial state under a supplied model. These functions
require all entries used by the contraction; they do not use stored energy or
calculate missing entries. For a selected result, use its energy properties.
For an independent evaluation, request the required blocks explicitly:

```python
trial = meanfi.density_matrix(
    model, keys=sorted(set(model.h_0) | set(model.required_coordinates.keys)),
    compute_free_energy=True,
)
print(meanfi.trial_internal_energy(model, trial))
print(meanfi.trial_free_energy(model, trial))
```

Entropy and free energy are omitted by default. Set `compute_free_energy=True`
on a density function or `solver` to request them. SCF then computes entropy
once after termination, including a failure with a valid partial state.
Intermediate evaluations and EDIIS never compute entropy.

```python
solution = meanfi.solver(model, model.random_meanfield(rng=0))
assert solution.entropy is None and solution.free_energy is None
```

At fixed chemical potential, no independent charge evaluation is performed.
Filling is derived from available occupations or a complete requested physical
diagonal; otherwise it is `None`. The charge-integration diagnostic is `None`.
Band energy and internal energy can also be `None` if calculating them would
require extra work beyond the requested density. Fixed-filling calculations
retain the energy needed by SCF. Explicit `compute_free_energy=True` requests
the additional thermodynamic work.

`trial_free_energy(model, density)` requires a `DensityResult` with computed entropy:
a few real-space density blocks alone do not determine full-state entropy.
Selecting fewer entries preserves computed entropy. Every backend exposes
`errors.entropy`, a diagnostic total-error estimate where available. Finite dense
systems report zero; adaptive UniformGrid compares coarse and fine entropy;
AAA adds its sampled scalar-fit error. Prescribed periodic grids and simplex
calculations without an entropy integration estimate report `None`.

AAA fits entropy on density-selected poles without a separate tolerance. Its
entropy error can be much larger than the density tolerance. When available,
`kT * errors.entropy` estimates the entropy contribution to free-energy error;
energy error must also be considered. These are empirical estimates, not rigorous
bounds. Reference subtraction affects interaction energy, not entropy. BdG
entropy includes the factor of one half that removes Nambu doubling.

`density.band_energy` is the expectation of the **input quadratic Hamiltonian**.
It includes the BdG normal-ordering constant, excludes chemical potential, and
uses the same normalization per physical orbital as the other energies.
It is not the interacting internal energy: `internal_energy` accounts for the
interaction's double counting. The SCF result computes both energies directly,
including when its density contains only the entries required by the interaction.

## Reference-subtracted energies

A fixed normal reference defines $\delta\rho=\rho-\rho_{\rm ref}$ and the
Hamiltonian $h[\rho]=h_0+W[\delta\rho]$, with the full Hartree/Fock map $W$.
The matching internal energy per physical orbital is

$$
U[\rho]=\frac{1}{N}\left(\langle h_0,\rho\rangle
+\frac12\langle W[\delta\rho],\delta\rho\rangle\right),
$$

where $\langle A,B\rangle=\sum_R\operatorname{Tr}(A_R B_{-R})$ is the
per-cell contraction. Differentiating $N U$ with respect to density gives
$h_0+W[\delta\rho]$, so the Hamiltonian and energy use the same subtraction.
`trial_internal_energy`, result energy properties and EDIIS use this functional.
`trial_free_energy` uses it with the entropy of the actual state: $F=U-kT\,S[\rho]$.
Neither entropy nor filling is reference-subtracted.

This is an energy functional for the reference-subtracted model, not the energy
difference from the reference. At $\rho=\rho_{\rm ref}$, the interaction term
vanishes but $U=\langle h_0,\rho_{\rm ref}\rangle/N$, which need not be zero.
Changing the reference changes the effective model unless the one-body
Hamiltonian is adjusted consistently. Compare solutions with the same model
and reference; an energy difference from the reference requires explicitly
subtracting its energy evaluated with that same model.

For a superconducting model, `reference` accepts either an N-dimensional normal
`DensityResult` or a 2N-dimensional BdG `DensityResult`:

- A normal reference supplies $\rho_{\rm ref}$ and means $\kappa_{\rm ref}=0$.
- A BdG reference supplies both $\rho_{\rm ref}$ and $\kappa_{\rm ref}$.

The correction and quadratic interaction energy use the same differences
$\delta\rho=\rho-\rho_{\rm ref}$ and
$\delta\kappa=\kappa-\kappa_{\rm ref}$. The one-body energy uses the actual
normal density; entropy uses the actual full state. The factor removing Nambu
doubling and normalization by N physical orbitals are unchanged. A paired
reference sets a phase: rotating both the state and reference leaves the energy
unchanged, while rotating only the state can change its energy.

```python
from dataclasses import replace

grid = meanfi.UniformGrid(nk=256)
normal_model = meanfi.Model(h_0, h_int, filling=filling, kT=kT)
reference = meanfi.density_matrix(normal_model, integration=grid)
bdg_model = replace(normal_model, superconducting=True, reference=reference)
solution = meanfi.solver(
    bdg_model, bdg_model.random_meanfield(rng=0, scale=0.1), integration=grid
)
```

To subtract a paired reference instead, pass a density result calculated from a
BdG model. Both forms may contain selected entries, but every required entry
must be present. Missing pairing entries in a BdG reference are unknown and
raise an error; only an explicitly normal, N-dimensional reference implies zero
pairing. The conversion reads selected entries directly and preserves sparse
Hamiltonian storage.

## Tight-binding dictionary utilities

```{eval-rst}
.. automodule:: meanfi.tb.ops
   :members: add_tb, scale_tb
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.tb.transforms
   :members:
   :show-inheritance:
```

## Coordinates and symmetries

```{eval-rst}
.. automodule:: meanfi.space
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.tb.utils
   :members:
   :show-inheritance:
```

## `kwant` interface

```{eval-rst}
.. automodule:: meanfi.interop.kwant
   :members:
   :show-inheritance:
```
