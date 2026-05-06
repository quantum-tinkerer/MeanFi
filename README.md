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
guess = meanfi.guess_tb(frozenset(h_int), onsite.shape[0])

# Solve
result = meanfi.solver(model, guess)
h_mf = meanfi.add_tb(h_0, result.mf)
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

Zero-temperature BdG calculations are not part of the main package workflow.

## Installation

Until the next packaged release, install from a checkout:

```bash
python -m pip install "stateful-quadrature @ git+https://github.com/Kostusas/stateful_quadrature.git"
python -m pip install "adaptivesimplex @ git+https://gitlab.kwant-project.org/qt/adaptivesimplex.git@main"
python -m pip install -e .
```

The zero-temperature `AdaptiveSimplex` backend now lives in the separate
`adaptivesimplex` package. MeanFi keeps the public `meanfi.AdaptiveSimplex`
API, but the native extension is no longer built from this repository.

For local development with Pixi:

```bash
pixi install
```

If you also want the `kwant` helpers:

```bash
python -m pip install ".[kwant]"
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
