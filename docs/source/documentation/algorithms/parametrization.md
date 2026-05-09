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
# Parametrization and symmetry reduction

The SCF solver does not iterate over every matrix entry independently.
Instead, it uses a reduced real parameter vector that removes obvious redundancy from Hermitian or BdG symmetry and from the structural support of `h_int`.

## Why reduce the parameter vector?

The reduced parameter vector has two goals:

- make the outer SCF problem smaller,
- avoid iterating over entries that can never affect the self-consistent mean field because the interaction is structurally zero there.

This reduction is implemented in the `meanfi.space.*` layer.
It provides the map $P$ used in the SCF fixed-point equation $\theta = P(\rho)$ from [SCF loop](./scf_loop.md).

## Normal-state reduction

For the normal density matrix $\rho(R)$, Hermiticity implies

:::{math}
\rho(R) = \rho^\dagger(-R).
:::

So the parametrization keeps only one representative of each Hermitian pair.
Onsite blocks keep the real diagonal and one complex triangle, offsite Hermitian partners are not parametrized twice, and support-aware parametrization retains only entries supported by `h_int`.

This is the logic behind `tb_to_rparams(...)` and `rparams_to_tb(...)`.

## BdG reduction

For BdG, the same idea is applied to the normal block, while the anomalous sector is reduced using particle-hole symmetry together with the interaction support.

At the BdG level, the quadratic Hamiltonian satisfies a particle-hole constraint of the form

:::{math}
\mathcal{H}_{\mathrm{BdG}}(k)
=
-\tau_x \mathcal{H}_{\mathrm{BdG}}(-k)^\ast \tau_x,
:::

so the lower particle-hole block is not independent of the upper one.

So the parametrization:

- keeps the normal Hermitian reduction,
- does not parametrize the lower BdG half independently,
- restricts the anomalous block by the interaction support,
- converts the retained complex data into a real parameter vector.

The current implementation is substantially reduced, although not mathematically minimal in every anomalous case.

## Support induced by `h_int`

The support logic uses the structural nonzero pattern of `h_int`:

- dense inputs use exact structural zeros,
- sparse inputs use the sparse pattern directly.

For density-density mean field, this works because the self-consistent correction is built only from matrix elements that are multiplied by the corresponding interaction entries.
Schematically, in the normal channel,

:::{math}
V_{\mathrm{MF},mn}(R) \propto v_{mn}(R)\,\rho_{mn}(R)
:::

away from the purely onsite Hartree accumulation.
So if the interaction structure enforces $v_{mn}(R)=0$, then the corresponding offsite or exchange contribution to the mean-field map vanishes identically for every iterate.
Those entries can therefore be removed from the SCF parametrization without changing the fixed-point problem.
