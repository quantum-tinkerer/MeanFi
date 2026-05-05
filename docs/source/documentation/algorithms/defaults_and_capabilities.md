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
# Defaults and capabilities

This page summarizes how `MeanFi` chooses defaults today and which method combinations are supported.

## Default integration selection

When `integration=None`, the code currently chooses:

| Condition | Default |
| --- | --- |
| `kT = 0`, normal | `AdaptiveSimplex()` |
| `kT > 0`, dense, normal | `AdaptiveQuadrature(matrix_function=DirectDiagonalization())` |
| `kT > 0`, sparse, normal | `AdaptiveQuadrature(matrix_function=RationalFOE(rational_scheme="aaa"))` |
| `kT > 0`, dense, superconducting | `AdaptiveQuadrature(matrix_function=DirectDiagonalization())` |
| `kT > 0`, sparse, superconducting | `AdaptiveQuadrature(matrix_function=RationalFOE(rational_scheme="aaa"))` |
| `kT = 0`, superconducting | no default; explicit `UniformGrid(...)` required |

## Capability table

The main supported combinations are:

| Family | `kT = 0` normal | `kT = 0` BdG | `kT > 0` normal | `kT > 0` BdG |
| --- | --- | --- | --- | --- |
| `AdaptiveSimplex` | Yes | No | No | No |
| `UniformGrid` | Yes | Yes | Yes | Yes |
| `AdaptiveQuadrature` | No | No | Yes | Yes |

## Matrix-function defaults

At finite temperature:

- dense problems default to `DirectDiagonalization()`,
- sparse problems default to `RationalFOE(rational_scheme="aaa")`.

For `UniformGrid`, the same dense/sparse distinction is still relevant when the matrix-function backend is omitted.

## Other current defaults

Some other public defaults that affect algorithm behavior are:

- `Model(..., kT=0.0)`
- `solver(..., scf=AndersonMixing(), scf_tol=1e-3)`
- adaptive integration methods default to `density_matrix_tol=1e-2`
- implicit `filling_tol` is derived from the density tolerance and orbital count

These defaults live in the runtime code, so this page should be updated whenever those policies change.
