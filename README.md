# `MeanFi`

## What is `MeanFi`?

`MeanFi` is a Python package for self-consistent mean-field calculations on tight-binding models with density-density interactions.
It starts from a Hamiltonian

$$
\hat{H} = \hat{H_0} + \hat{V}
$$

and computes a self-consistent mean-field correction that approximates the interaction term.

For more details, see the [theory section](https://meanfi.readthedocs.io/en/latest/documentation/theory/index.html) and the [algorithm section](https://meanfi.readthedocs.io/en/latest/documentation/algorithms/index.html).

## How to use `MeanFi`

The basic workflow has three steps:

1. **Define** the non-interacting Hamiltonian, interaction, and filling.
2. **Guess** a starting mean-field correction.
3. **Solve** for the self-consistent correction.

```python
import meanfi

# Define
h_0 = {(0,): onsite, (1,): hopping, (-1,): hopping.T.conj()}
h_int = {(0,): onsite_interaction}
model = meanfi.Model(h_0, h_int, filling=2)

# Guess
guess = model.random_meanfield(rng=0, scale=0.1)

# Solve
result = meanfi.solver(model, guess)
h_mf = meanfi.add_tb(h_0, result.mean_field)
```

For examples, see the [tutorials](https://meanfi.readthedocs.io/en/latest/tutorial/hubbard_1d.html).

## Why `MeanFi`?

- **Simple**

  The workflow is compact and close to the physics problem.

- **Extensible**

  The code is structured to be easy to read, debug, and extend.

- **Numerically focused**

  The package provides adaptive and fixed-grid Brillouin-zone integration methods for self-consistent calculations.

## Current scope

`MeanFi` currently supports:

- density-density interactions,
- zero- and finite-temperature mean-field calculations,
- superconducting BdG mean-field calculations at finite temperature,
- tight-binding dictionary workflows,
- optional `kwant` conversion helpers.

Zero-temperature BdG calculations require an explicit `PeriodicGrid(nk=...)`.

## Integration

MeanFi has two integration families. Normal zero-temperature calculations default
to `AdaptiveSimplex()` backed by FermiSimplex. Dense finite-temperature normal
and BdG calculations default to `PeriodicGrid()` with direct diagonalization,
global refinement and mandatory shifted-grid validation.

```python
# Prescribed final mesh size (total points, with documented rounding).
integration = meanfi.PeriodicGrid(nk=4096)
integration = meanfi.AdaptiveSimplex(nk=4096)

# Accuracy control; omit nk.
integration = meanfi.PeriodicGrid(density_matrix_tol=1e-5, charge_tol=1e-6)
```

Do not combine `nk` with integration targets. The top-level `tol` still controls
root finding and SCF on a prescribed mesh; integration errors then remain
unavailable. `nk=4096` means 64² periodic points in 2D or 16³ in 3D, with simplex
rounding explained in the [integration guide](https://meanfi.readthedocs.io/en/latest/documentation/algorithms/integration_families.html).

`UniformGrid`, `PeriodicQuadrature` and `AdaptiveQuadrature` have been removed.
To preserve an old `UniformGrid(nk=n)` mesh in dimension `d`, use
`PeriodicGrid(nk=n**d)`. Prescribed positive-temperature sparse `RationalFOE` remains
available; automatic sparse integration requires explicit migration and never
silently selects dense evaluation. See [migration notes](https://meanfi.readthedocs.io/en/latest/documentation/algorithms/integration_families.html#migration).

## Installation

For a fresh development checkout, use [Pixi](https://pixi.sh/). Pixi creates
the Python environment and installs the native build tools that MeanFi needs:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://gitlab.kwant-project.org/qt/meanfi.git
cd meanfi
pixi install
pixi run python -c "import meanfi; print(meanfi.__version__)"
```

The zero-temperature `AdaptiveSimplex` backend now lives in the separate
`FermiSimplex` package, which uses `adaptivesimplex` as its generic C++
mesh engine. MeanFi keeps the public `meanfi.AdaptiveSimplex` API, but the
native extension is no longer built from this repository. The current
`FermiSimplex` package vendors `adaptivesimplex`, so a normal Pixi install
does not require a separate AdaptiveSimplex checkout or CMake install.

Common development tasks can then be run through Pixi:

```bash
pixi run tests-mid
pixi run -e docs docs-build
```

If you also want the `kwant` helpers:

```bash
pixi install -e mid
```

## Citing `MeanFi`

If you use `MeanFi` in scientific work, please cite:

```bibtex
@misc{meanfi,
  author = {Vilkelis, Kostas and Zijderveld, R. Johanna and Akhmerov, Anton R. and Manesco, Antonio L.R.},
  doi = {10.5281/zenodo.11149850},
  month = {5},
  title = {MeanFi},
  year = {2024}
}
```
