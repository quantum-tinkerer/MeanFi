# Density at one momentum

For a Hermitian $A=H(k)-\mu Q$, the density is $f(A)$. At positive temperature,
$f(E)=(1+e^{E/kT})^{-1}$. At zero temperature it is the occupied-state projector,
with half occupation for an exactly zero eigenvalue.

## Dense diagonalization

With $A=V\operatorname{diag}(E)V^\dagger$,

$$
\rho_{ij}=\sum_a V_{ia} f(E_a)V_{ja}^*.
$$

`DirectDiagonalization()` computes the spectrum and contracts only requested
density entries. Normal-state charge is simply $\sum_a f(E_a)$; BdG charge uses
the electron block. Diagonalization error is numerical roundoff, separate from
momentum integration.

UniformGrid processes matrices in batches. During a normal fixed-filling solve,
eigenvalues can be retained so trial chemical potentials need only new
occupations. Sparse inputs can explicitly request dense diagonalization;
FermiSimplex also uses dense eigensystems.

## Sparse AAA approximation

`RationalFOE()` fits the Fermi function on a spectral interval:

$$
r(E)=c+\sum_p\left[\frac{w_p}{E-z_p}
+\frac{w_p^*}{E-z_p^*}\right],\qquad \rho\approx r(A).
$$

AAA chooses support points and poles from scalar samples. A separate sampled
check accepts a fit when its error meets `matrix_function_tol`. For Hermitian
$A$, scalar approximation error bounds matrix-entry error on the actual
spectrum:

$$
\max_{ij}|[r(A)-f(A)]_{ij}|
\leq \max_{E\in\operatorname{spec}(A)}|r(E)-f(E)|.
$$

The sampled check estimates this scalar error; it is not a rigorous bound over
the entire interval. An accepted fit can be reused on a nearby spectral
interval after checking it there.

MUMPS factors $A-z_pI$ once per pole and extracts requested inverse entries.
A charge probe needs only physical diagonal entries: electron entries for BdG.
Density evaluation does not require a preceding charge probe. Energy or entropy,
when requested, obtains missing inverse diagonals using the same factors.
Selected entries already available at that chemical potential are reused.

AAA requires sparse matrices and positive temperature. Periodic calculations
use a prescribed `UniformGrid(nk=...)`; automatic sparse grid refinement is not
supported. `initial_poles` and `max_poles` control fitting work, not alternative
accuracy targets. A fit that cannot meet its target raises `ConvergenceError`.

## Optional entropy

`compute_free_energy=True` adds entropy of the full state,

$$
S=-\operatorname{Tr}\bigl[\rho\log\rho+(I-\rho)\log(I-\rho)\bigr].
$$

Dense evaluation reuses occupations. AAA fits entropy residues on the accepted
density poles and reuses their factors; entropy does not alter pole selection.
Its approximation error may differ substantially from density accuracy and is
reported only as a diagnostic. BdG totals include the half factor removing
Nambu doubling, followed by normalization per physical orbital.
