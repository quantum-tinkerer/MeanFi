# Matrix-function backends

Once a k-space node is chosen, `MeanFi` still has to compute the density matrix contribution at that node.
This is the role of the matrix-function backend.

```{toctree}
:hidden:
:maxdepth: 1

method_notes/direct_diagonalization.md
method_notes/rational_foe.md
```

## Single-$k$ viewpoint

At a fixed $k$, the backend evaluates the density matrix of the shifted Hamiltonian

:::{math}
H(k) - \mu Q
:::

or the corresponding normal-state version, and then extracts the relevant density block.

The same evaluation supplies charge, band energy, and entropy alongside the requested density entries.

## Exact diagonalization

The direct path diagonalizes the sampled matrix explicitly and evaluates the occupation function from the eigenvalues.
It is straightforward and robust, but dense diagonalization becomes expensive as the matrix size grows.

## Rational FOE

The rational FOE path uses AAA with shared poles for occupation and entropy, without full diagonalization.
It is especially useful in sparse finite-temperature calculations where exact diagonalization would be much heavier.

## Dense versus sparse behavior

Dense and sparse backends are not just storage choices.
They also affect which matrix-function strategies are practical:

- dense problems default to direct diagonalization,
- sparse finite-temperature problems require an explicit supported configuration,
- rational evaluation supports sparse inputs only.

Explicit `RationalFOE` is supported for sparse matrices with `UniformGrid(nk=...)`
at positive temperature. Adaptive rational integration is not supported. Selecting an
automatic sparse calculation raises migration guidance instead of silently
allocating dense matrices.

## Cost versus error scaling

At fixed $k$, the generic pattern is

:::{math}
\text{cost} \sim C_k,
\qquad
\rho(k,\mu) \approx \rho_\varepsilon(k,\mu),
:::

with the approximation error controlled either by exact dense linear algebra tolerance or by the chosen rational approximation.
For approximate backends, the family-specific page below makes the relation more explicit as a cost-versus-error law.

- [Direct diagonalization](./method_notes/direct_diagonalization.md): exact at fixed matrix size up to numerical eigensolver error, with cubic dense work
- [Rational FOE](./method_notes/rational_foe.md): approximate matrix-function evaluation with work proportional to the number of poles times the cost of one shifted solve
