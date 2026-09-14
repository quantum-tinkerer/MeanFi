# `DirectDiagonalization`

`DirectDiagonalization` is the exact matrix-function backend.

At each sampled $k$ point, it diagonalizes the shifted Hamiltonian explicitly:

:::{math}
H(k) - \mu Q = U \Lambda U^\dagger.
:::

The density matrix is then reconstructed as

:::{math}
\rho(k) = U\, f(\Lambda)\, U^\dagger,
:::

with $f$ the Fermi-Dirac occupation function.

## What “exact” means here

Here “exact” means exact for the finite sampled matrix at that k-point, up to numerical diagonalization error.
It does not mean the full Brillouin-zone integral is exact, because that still depends on the chosen integration family.

## Thermodynamic quantities

The eigenvalues and occupations also give band energy and entropy without another
eigendecomposition. Entropy is the sum of
`-f * log(f) - (1-f) * log(1-f)`, evaluated stably at empty and occupied states.
BdG sums include a factor of one half to remove Nambu doubling; band energy also
includes the normal-ordering constant.

For normal finite-temperature filling solves, the retained eigenvalues give the
charge derivative `sum(f * (1-f)) / kT`. BdG charge depends on the electron weights
of the eigenvectors, and its filling solve uses bracketing.

## Cost versus error scaling

For an $n \times n$ dense matrix, the dominant cost is dense diagonalization, which scales roughly like

:::{math}
\mathcal{O}(n^3).
:::

That is why direct diagonalization is simple and robust, but becomes expensive for large sparse problems.
At fixed matrix size, this path is not an adjustable approximation scheme in the same sense as rational FOE:
the main error is the numerical eigensolver error rather than a tunable approximation tolerance.

It is the default finite-temperature backend for dense calculations.
