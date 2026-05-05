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
# Theory

This section introduces the physical problem solved by `MeanFi` and the approximations used to turn it into a tractable self-consistent calculation.

```{toctree}
:hidden:
:maxdepth: 1

interacting_problem.md
mean_field.md
bdg.md
finite_temperature.md
```

The pages are arranged from the physical problem to the approximations used in the code:

- [Interacting problem](./interacting_problem.md): the density-density Hamiltonian and the translationally invariant tight-binding setting
- [Mean-field approximation](./mean_field.md): Hartree-Fock decoupling and the self-consistent normal-state correction
- [BdG extension](./bdg.md): anomalous pairing, electron-first BdG structure, and why superconducting fixed-filling problems are heavier
- [Finite temperature](./finite_temperature.md): the role of `kT` in both the physics and the numerics
