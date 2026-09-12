# Primary literature and the limits of the benchmark conclusions

## Periodic accuracy is a convergence theorem, not a universal ranking

[Trefethen and Weideman, *The Exponentially Convergent Trapezoidal Rule*, SIAM Review 56 (2014), Theorem 3.2](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2014_149.pdf)
proves exponential convergence for bounded periodic functions analytic in a
complex strip. Fourier alias cancellation explains the result. The strip width
and the analytic bound determine the constants. This theorem neither ranks every
quadrature method nor certifies a difference between two numerical grids. An
analyticity assumption without a quantitative bound cannot exclude unresolved
high Fourier frequencies at a particular finite resolution.

## Finite-temperature electronic integrals have relevant rigorous results

[Cancès, Ehrlacher, Gontier, Levitt and Lombardi, *Numerical quadrature in the Brillouin zone for periodic Schrödinger operators*, Section 5.3, Lemmas 5.9–5.10 and Appendix A](https://arxiv.org/pdf/1805.07144)
proves error estimates for periodic electronic integration with smearing. At fixed
Fermi–Dirac smearing width sigma, their estimates contain an exponential factor
`exp(-eta*sigma*L)` and a temperature-dependent prefactor; L is points per axis.
Their Remark 5.2 identifies sigma with physical `kT` for Fermi–Dirac occupations.
Gaussian-type energy smearing has different analyticity and faster convergence,
but changes the occupation function. It is not interchangeable with physical
finite-temperature Fermi–Dirac statistics. Also distinguish Gaussian *energy
smearing* from the Gauss–Legendre *quadrature rule* used in our benchmark.

The paper concerns periodic Schrödinger operators and explicit assumptions on
band structure. It is supporting context, not a direct theorem proving our entire
finite-matrix BdG solver or adaptive stopping criterion. For finite analytic
matrix Hamiltonians at fixed positive temperature, analyticity of the matrix
Fermi function follows directly from holomorphic matrix functional calculus and
absence of real-axis Fermi poles. This avoids relying on individual eigenvectors
being smooth at band crossings.

## Adaptive alternatives remain relevant

[Kaye, Beck, Barnett, Van Muñoz and Parcollet, *Automatic, high-order, and adaptive algorithms for Brillouin zone integration*, Sections II–IV and Appendix A](https://arxiv.org/pdf/2211.12959)
compares periodic, tree-adaptive and iterated adaptive integration for broadened
spectral/Green-function integrals. Periodic quadrature is favorable at large
broadening; iterated adaptivity becomes attractive when sharp features localize
near energy surfaces. Their automatic periodic algorithm checks different grid
sizes and warns that too-small size increments may underestimate error.

Their claimed small-broadening complexity concerns a parameter eta in resolvent
integrals. It cannot simply be relabeled as a theorem in physical temperature T
for MeanFi charge and BdG density integration. Their Hamiltonians are small and
cheap to evaluate using Wannier interpolation; the balance of assembly, spectral
work and quadrature control can differ from this 48/96-dimensional study.

Our anisotropic periodic prototype is also not their iterated adaptive method.
It chooses separate global tensor-grid orders along each axis. Iterated adaptive
integration chooses inner 1D samples separately at each outer integration point.
That distinction matters for curved Fermi surfaces and localized pockets.

## Why directional periodic refinement can matter

The following is a direct tensor-product deduction from the one-dimensional
periodic theorem, not a quoted performance claim from a paper. Suppose normalized
integration in direction j is `I_j`, the positive uniform periodic rule is `Q_j`,
and the integrand extends analytically in that direction to a strip of width
`a_j`, uniformly bounded by M for all other coordinates real. Telescoping gives

\[
I_1\cdots I_d-Q_1\cdots Q_d
=\sum_{j=1}^d Q_1\cdots Q_{j-1}(I_j-Q_j)I_{j+1}\cdots I_d.
\]

The integral and positive quadrature operators in the other real coordinates
have supremum norm one. Therefore

\[
|I-Q_{n_1,\ldots,n_d}|\le
\sum_{j=1}^d\frac{2M}{e^{a_j n_j}-1}.
\]

This gives a clear reason to use different n_j: roughly balance `a_j*n_j` across
directions. Isotropic grids resolve every direction as though it had the smallest
strip width. Strong directional anisotropy can therefore waste orders of
magnitude in a tensor point count. Our prototype estimates directional needs
empirically using axis-halved subgrids; it does not determine a_j or M and does
not turn this theorem into a certified error bound.

Global direction selection cannot exploit all forms of localized structure. A
small pocket or a narrow curved surface can require fine resolution along every
coordinate somewhere while occupying little overall volume. Spatially adaptive
or iterated methods may then use far fewer samples. Coordinate choice can also
change how strongly an integrand appears anisotropic.

## What the report should and should not conclude

The tested GM/GK implementations use the installed stateful controller, which
splits every coordinate of a selected rectangle, producing `2**d` children. It
does not choose one local bisection direction or implement iterated 1D quadrature.
Bounded batches, derivative budgeting, charge-only roots and fresh density meshes
are meaningful optimizations, but they do not exhaust adaptive algorithm design.
Thus report a best method **among the measured implementations and tested
problems**, with independently checked errors. A universal claim that periodic
sampling is mathematically fastest would be unsupported.

Timing failures caused by retaining a whole grid of eigenvectors are storage
failures, not lower bounds on the competing quadrature rule. The bounded Gaussian
and streamed fixed-mu controls specifically remove that confounding effect.
Likewise, a larger number of observed eigenvalues-only calls after the cache
fills indicates a cache-policy cost; it is not additional quadrature refinement.

The [BdG reference-transfer note](BDG-REFERENCE-BOUNDS.md) supplies a separate
rigorous finite-matrix Lipschitz bound for nearby chemical potentials. That bound
controls transferring a reference in mu; it does not certify the reference grid
itself.
