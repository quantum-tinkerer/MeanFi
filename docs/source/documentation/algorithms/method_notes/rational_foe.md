# `RationalFOE`

`RationalFOE()` evaluates sparse Hamiltonians at positive temperature on a
prescribed `UniformGrid(nk=...)`. It uses AAA rational approximation and MUMPS
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

AAA chooses poles using only the Fermi function on a spectral interval
bounded by Gershgorin estimates. After acceptance, entropy residues are fitted
on those fixed poles. Occupation and entropy have different residues
but share the same poles:

:::{math}
g(A)\approx c_g I+2\operatorname{Re}\sum_\ell
w_{g,\ell}(A-z_\ell I)^{-1},
\qquad g\in\{f,s\}.
:::

Here the matrix real part denotes the Hermitian combination with the conjugate
pole. Off-diagonal entries use the transposed conjugate resolvent entry.

Only the Fermi-function approximation determines pole selection and acceptance.
For Hermitian $A$, the spectral theorem gives

:::{math}
\max_{ij}|[r(A)-f(A)]_{ij}|\leq \|r(A)-f(A)\|_2
=\max_{\lambda\in\mathrm{spec}(A)} |r(\lambda)-f(\lambda)|.
:::

This controls off-diagonal as well as diagonal density entries. At fixed
chemical potential, the scalar target is `matrix_function_tol`, defaulting to
`tol/5` through the public tolerance policy. During filling searches it is the
smaller of that target and one quarter of the filling-residual budget divided
by the sum of absolute charge weights. Mesh-integration targets do not control
the scalar fit. Positive uniform-grid weights
preserve the matrix-entry bound. `errors.matrix_function_error` reports the
largest achieved sampled Fermi-fit error over the density calculation's momenta,
including revalidation of reused fits. Mesh error is a separate quantity and
remains `None` on prescribed meshes.

Fitting starts on a small grid and refines only when needed. Accepted Fermi fits
pass a dense scalar validation grid resolving edges, the transition and tails.
Entropy is fitted afterward and reports a sampled approximation error in
`errors.entropy_approximation`. It has no acceptance target and may be much less
accurate than density. Charge-only evaluations skip the entropy fit.

AAA finds denominator weights by minimizing $\|Lw\|$ with $\|w\|=1$,
where $L$ is the Loewner matrix for the Fermi function. We first compute $L=QR$, then take the
smallest right singular vector of $R$. Since $Q$ has orthonormal columns,
$\|Lw\|=\|Rw\|$: the SVD operates on a small square matrix with the same
least-squares objective. This avoids forming $L^T L$, which would square the
condition number.

A nearly converged intermediate fit may proceed to residue refitting; only the
final partial-fraction errors determine acceptance. Constant approximations
must pass the same checks. These are sampled scalar
checks, not rigorous uniform-error certificates between sample points.

One scalar fit is shared across chemical potentials and k-points within a
calculation. The first fit uses the actual spectral bounds. When an overlapping
interval extends beyond those bounds, the next fit adds 20% of the new interval's
width at each end, leaving room for subsequent shifts. If the expanded interval
cannot be fitted within the pole budget, fitting retries the actual bounds.
Every reuse checks temperature, pole budget, and sampled Fermi-function
accuracy on the current interval. Entropy coefficients are attached when needed
without changing the accepted density poles or residues.

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

The density fit accounts for charge trace weights and matrix size. Band-energy
error scales with Hamiltonian energy units. Sparse free-energy comparisons must
also account for `kT * errors.entropy_approximation`; this diagnostic does not
control convergence. A Fermi fit that cannot meet its target within `max_poles`
raises `ConvergenceError`. A prescribed grid has no Brillouin-zone integration
error estimate.

## Cost and configuration

The leading matrix cost is the number of poles times the sparse factorization
and selected-inversion cost. Sparsity pattern and fill-in determine that cost;
small dense problems can be faster with direct diagonalization. `initial_poles`
and `max_poles` control the approximation budget. There is no scheme selector.

An explicit positive-temperature sparse `UniformGrid(nk=...)` selects
`RationalFOE()` when its matrix function is omitted. Accuracy-controlled rational
integration is unsupported. Use `DirectDiagonalization()` explicitly when dense
evaluation of sparse inputs is wanted.

See [the AAA paper](https://epubs.siam.org/doi/10.1137/16M1106122) for scalar
rational approximation and [PEXSI](https://pexsi.readthedocs.io/en/stable/introduction.html)
for shared-pole density, energy, and free-energy evaluation.
