---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# pymf

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tutorials

tutorial/hubbard_1d.md
tutorial/graphene_example.md
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Documentation

documentation/mf_notes.md
documentation/algorithm.md
documentation/pymf.md
```

## What is pymf?

Pymf is a Python package for finding mean-field corrections to the non-interacting part of a Hamiltonian. It is designed to be simple to use and flexible enough to handle a wide range of systems. Pymf works by solving the mean-field equations self-consistently.

Finding a mean-field solution is a 4-step process:

- Define the non-interacting and interacting part of the Hamiltonian separately as hopping dictionaries.
- Combine the non-interacting and interacting parts togher with your filling into a `Model` object.
- Provide a starting guess and the number of k-points to use the `solver` function and find the mean-field correction.
- Add the mean-field correction to the non-interacting part to calculate the total Hamiltonian.

```python
import pymf

model = pymf.Model(h_0, h_int, filling=filling)
mf_sol = pymf.solver(model, guess)
h_full = pymf.add_tb(h_0, mf_sol)
```

## Why pymf?

Here is why you should use pymf:

* Minimal
  It contains the minimum of what you need to solve mean-field equations.

* Simple
  The workflow is simple and straightforward.

* Time-effective
  As pymf uses tight-binding dictionaries as input and returns, you can calculate the mean-field corrections on a coarse grid, but use the full Hamiltonian on a fine grid for observables afterward.


## How does pymf work?

## What does pymf not do yet?

* Superconductivity

## Installation

```bash
pip install .
```
## Citing

## Contributing
