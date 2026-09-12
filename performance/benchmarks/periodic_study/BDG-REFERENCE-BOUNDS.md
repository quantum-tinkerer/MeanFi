# Reusing a BdG reference at a nearby chemical potential

These bounds hold for a fixed trial Hamiltonian and positive temperature. They do
not describe how a self-consistent Hamiltonian changes when its solution changes.
Let `tau = kT > 0`, `A_mu(k) = H0(k) - mu Q`, `||Q||_2 = 1`, and
`F_tau(A) = (I + exp(A/tau))**(-1)`. For normal systems take `Q = I`.

## Operator Lipschitz bound and proof

For any finite Hermitian matrices A and B,

\[
\|F_\tau(A)-F_\tau(B)\|_2 \le \frac{\|A-B\|_2}{4\tau}.
\]

The Matsubara representation is

\[
F_\tau(A)=\tfrac12 I+\tau\sum_{n\in\mathbb Z}(i\omega_n I-A)^{-1},
\qquad \omega_n=(2n+1)\pi\tau,
\]

with symmetric summation. The standard expansion follows from the poles of tanh;
see [Lin, Lu, Car and E, equation 2 and Appendix A](https://web.math.princeton.edu/~weinan/pdf%20files/pole.pdf).
Their density includes an additional factor two, which is omitted here. The
following norm estimate is a direct deduction from that expansion.

Subtract the representations. The difference series converges absolutely because
the resolvent identity gives

\[
R_A(z)-R_B(z)=R_A(z)(A-B)R_B(z).
\]

For Hermitian A and B, `||R_A(i omega)||_2 <= 1/|omega|`, hence

\[
\|F_\tau(A)-F_\tau(B)\|_2
\le \tau \|A-B\|_2\sum_{n\in\mathbb Z}|\omega_n|^{-2}
=\frac{\|A-B\|_2}{4\tau},
\]

using `sum_{m>=0}(2m+1)^(-2) = pi^2/8`. No simultaneous diagonalizability is
assumed, so this works for noncommuting BdG shifts.

## Density and charge consequences

Every individual matrix element has magnitude at most the operator norm. A
Fourier phase has modulus one, and the Brillouin-zone integration is a normalized
average. Therefore every selected real-space normal or anomalous density entry
satisfies

\[
|\rho_{R,ij}(\mu)-\rho_{R,ij}(\nu)|\le\frac{|\mu-\nu|}{4\tau}.
\]

For charge `N(mu) = average Tr[W F_tau(A_mu(k))]`, where W is the orthogonal
projector onto `ndof` electron coordinates, trace/operator norm duality gives

\[
|N(\mu)-N(\nu)|\le\frac{\mathrm{ndof}}{4\tau}|\mu-\nu|.
\]

For general fixed weights replace `ndof` by the nuclear norm `||W||_1`; if Q is
not a charge involution, also multiply both bounds by `||Q||_2`.

Suppose an independent reference at `mu_ref` has a charge estimate `N_ref` and
an error allowance `e_ref`. A candidate returned chemical potential is safely
qualified for filling tolerance eps if

\[
|N_\mathrm{ref}-N_\mathrm{target}|+e_\mathrm{ref}
+\frac{\mathrm{ndof}}{4\tau}|\mu_\mathrm{candidate}-\mu_\mathrm{ref}|\le\epsilon.
\]

If the inequality fails, the bound is inconclusive: evaluate the reference charge
at the candidate mu. It does not establish that the candidate is inaccurate.
If the reference error allowance is itself empirical, the bound rigorously
controls mu transfer but does not turn empirical reference convergence into a
certified integral bound. Label the combined qualification accordingly.

At `ndof=48`, `tau=0.1`, charge sensitivity is at most 120 per energy unit. To
qualify eps=1e-4 with negligible reference error using this bound alone requires
`|delta_mu| <= 8.33e-7`. At `tau=0.02` that threshold is `1.67e-7`. Thus the bound
is useful for tightly solved roots, but can be conservative for methods that
accept a physical root residual of eps/4.

When comparing a candidate density to the physical fixed-filling reference
rho(mu_ref), simply report that measured density difference with reference error.
Do not add a mu-transfer term unnecessarily: the candidate's root-induced density
error is precisely part of the quantity being tested. Add the Lipschitz term only
when using rho(mu_ref) to bound the integration error specifically at a different
candidate mu.

## Optional sharper transfer using one reference derivative

If `N'_ref` has independently controlled numerical error, the same resolvent
argument gives a global second-derivative norm bound

\[
\left\|\frac{d^2F_\tau(A_\mu)}{d\mu^2}\right\|_2
\le 2\tau\sum_n |\omega_n|^{-3}
=\frac{7\zeta(3)}{2\pi^3\tau^2}.
\]

Consequently,

\[
|N(\mu)-N(\nu)-N'(\nu)(\mu-\nu)|
\le \frac{7\zeta(3)\,\mathrm{ndof}}{4\pi^3\tau^2}|\mu-\nu|^2.
\]

This could avoid most repeated reference diagonalizations if the derivative is
computed once from the reference eigensystems. Account for derivative quadrature
error multiplied by `|delta_mu|` and reference charge error. A finite-difference
slope without its own error control does not make this a certified transfer.
No implementation or benchmark qualification currently relies on this optional
Taylor refinement.
