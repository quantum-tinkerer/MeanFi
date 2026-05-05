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

This reduction is implemented in the `meanfi.state.*` layer.

## Normal-state reduction

For the normal state:

- onsite blocks keep only the real diagonal and the complex upper triangle,
- offsite Hermitian partners are not parametrized twice,
- only entries supported by `h_int` are retained when support-aware parametrization is used.

This is the logic behind `tb_to_rparams(...)` and `rparams_to_tb(...)`.

## BdG reduction

For BdG:

- the normal block uses the same Hermitian reduction as the normal state,
- the lower half of the BdG matrix is not independently parametrized,
- the anomalous block is restricted by the interaction support,
- the top-half BdG support is converted into a real parameter vector.

The current implementation reduces the BdG state substantially, but it is not the mathematically smallest possible anomalous basis in every case.

## Support induced by `h_int`

The support logic uses the structural nonzero pattern of `h_int`:

- dense inputs use exact structural zeros,
- sparse inputs use the sparse pattern directly.

This is how `MeanFi` avoids iterating over entries that cannot contribute to the mean-field update.

## Parameter reduction versus density evaluation

The reduced parameter vector and the actual density evaluation are related but not identical concerns.

- The SCF vector is reduced aggressively.
- Some numerical density-evaluation paths still compute denser intermediate objects internally, especially on dense backends.

So it is useful to distinguish:

| Aspect | Normal | BdG |
| --- | --- | --- |
| Hermitian/BdG symmetry reduction | Yes | Yes |
| Support restriction from `h_int` | Yes | Yes |
| Lower-half redundancy removed | Not applicable | Yes |
| Fully minimal anomalous basis | Not applicable | Not completely |
| Numerical density work always reduced to the same support | Not on every backend | Not on every backend |

The reduction therefore applies most strongly to the SCF state representation, and only partially to every numerical backend.
