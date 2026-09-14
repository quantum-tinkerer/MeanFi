# `RationalFOE`

`RationalFOE()` evaluates sparse Hamiltonians at positive temperature on a
prescribed `PeriodicGrid(nk=...)`. It uses AAA rational approximation and MUMPS
selected inversion. The density and entropy calculation shares shifted sparse
factorizations; it does not diagonalize the Hamiltonian or form its full inverse.

## One pole set for density and entropy

At each momentum, write the shifted Hamiltonian as $A=H(k)-\mu Q$, where $Q=I$
for a normal system. The scalar functions are

:::{math}
f(x)=\frac{1}{1+e^{x/kT}},
\qquad
s(x)=-f(x)\log f(x)-(1-f(x))\log(1-f(x)).
:::

AAA chooses a common denominator for these functions on a spectral interval
bounded by Gershgorin estimates. Occupation and entropy have different residues
but share the same poles:

:::{math}
g(A)\approx c_g I+2\operatorname{Re}\sum_\ell
w_{g,\ell}(A-z_\ell I)^{-1},
\qquad g\in\{f,s\}.
:::

Here the matrix real part denotes the Hermitian combination with the conjugate
pole. Off-diagonal entries use the transposed conjugate resolvent entry.

The fit combines occupation and entropy residuals, refits residues on the
resulting poles, and checks both approximations on a separate, denser scalar
grid. Samples resolve the band edges, Fermi transition, and thermal tails.
A nearly converged intermediate fit may proceed to residue refitting; only the
final partial-fraction errors determine acceptance. Constant approximations
must pass the same checks. These are sampled scalar
checks, not rigorous uniform-error certificates between sample points.

A scalar fit may be reused for a contained spectral interval after checking its
requested accuracy. Charge-only evaluations can fit just the occupation
function; the final density evaluation includes entropy.

## Reusing sparse work

For each pole, MUMPS factors $A-z_\ell I$ once. Selected inversion supplies the
requested density entries and diagonal entries. The full diagonal is needed for
entropy, including both electron and hole blocks in BdG calculations.

Band energy uses the same resolvent traces through

:::{math}
A(A-zI)^{-1}=I+z(A-zI)^{-1}.
:::

Thus entropy and band energy require no extra matrix factorizations. The sparse
node restores the chemical-potential shift. The integrator includes the BdG
normal-ordering constant, removes Nambu doubling where applicable, and divides
physical energy and entropy by the number of physical orbitals per cell.
The SCF layer subtracts interaction double counting to obtain internal energy,
then returns `free_energy = internal_energy - kT * entropy`.

Scalar tolerances account for charge trace weights, matrix size, and the spectral
energy scale. Occupation accuracy alone is insufficient for entropy near empty
or occupied states. A fit that cannot meet its requested accuracy within
`max_poles` raises `ConvergenceError`; a prescribed grid still has no
Brillouin-zone integration error estimate.

## Cost and configuration

The leading matrix cost is the number of poles times the sparse factorization
and selected-inversion cost. Sparsity pattern and fill-in determine that cost;
small dense problems can be faster with direct diagonalization. `initial_poles`
and `max_poles` control the approximation budget. There is no scheme selector.

An explicit positive-temperature sparse `PeriodicGrid(nk=...)` selects
`RationalFOE()` when its matrix function is omitted. Accuracy-controlled rational
integration is unsupported. Use `DirectDiagonalization()` explicitly when dense
evaluation of sparse inputs is wanted.

See [the AAA paper](https://epubs.siam.org/doi/10.1137/16M1106122) for scalar
rational approximation and [PEXSI](https://pexsi.readthedocs.io/en/stable/introduction.html)
for shared-pole density, energy, and free-energy evaluation.
