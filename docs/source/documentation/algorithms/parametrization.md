# Coordinates and symmetry

`DensityCoordinates` lists addresses $(R,a,b)$; `DensityResult` stores the
computed values at those addresses. Model-based calculations select the entries
needed by the interaction. Request `keys=[...]` when complete real-space blocks
are needed. Missing entries are unknown, not zero.

The SCF representation is private. It retains only active interaction entries,
then removes redundant real variables using Hermiticity, pairing antisymmetry,
and any supplied spatial symmetries. Conceptually,

$$
x=\begin{pmatrix}\operatorname{Re}\rho_{\mathrm{active}}\\
\operatorname{Im}\rho_{\mathrm{active}}\end{pmatrix}=By.
$$

SCF mixes the independent real vector $y$. `model.required_coordinates` selects
enough density entries to recover it. Without spatial symmetries, small index
orbits implement this map; a dense basis is only constructed when spatial
constraints need one. The model constructs the maps once and reuses them.
This reduction enforces the specified linear constraints, not positivity of an
arbitrary trial density.

## Constraints

Hermiticity and fermionic pairing antisymmetry require

$$
\rho_{ab}(R)=\rho_{ba}(-R)^*,\qquad
F_{ab}(R)=-F_{ba}(-R).
$$

A `SpatialSymmetry` acts on lattice basis states as

$$
g|R,a\rangle=\sum_{s,c} U_s[c,a]|AR+s,c\rangle.
$$

$A$ is an integer lattice map with determinant $\pm1$. Shift blocks jointly
define a unitary transformation; individual blocks need not be unitary, which
allows glide symmetries. The corresponding normal-density constraint is

$$
\rho_{ab}(R)=\sum_{s,t,c,d} U_s[c,a]^*U_t[d,b]\,
\rho_{cd}(AR+s-t).
$$

The displacement uses the **left shift minus the right shift**. Pairing uses
$U_s[c,a]U_t[d,b]$ instead of $U_s[c,a]^*U_t[d,b]$. Antiunitary symmetries also
conjugate the transformed values.

The interaction fixes the active support. If a symmetry maps an active entry
outside it, MeanFi warns and treats that outside active variable as zero;
choose an interaction support compatible with the intended symmetry. The user
is responsible for choosing a symmetry of the physical model.

`model.random_meanfield(rng=0)` uses this same constrained space to produce a
mean-field correction. Supplied guesses are projected into it, with a warning
if components are removed. The [glide tutorial](../../tutorial/glide_symmetry.md)
shows how these constraints affect a self-consistent solution.
