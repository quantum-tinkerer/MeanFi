# Theory

MeanFi uses self-consistent Hartree–Fock theory for tight-binding models with
density-density interactions. Orbital indices include spin when present.

(theory-interaction)=
## Hamiltonian and interaction

For a finite system, the model specifies

:::{math}
\hat H_0=\sum_{ij}h_{ij}c_i^\dagger c_j,
\qquad
\hat V=\frac12\sum_{ij}v_{ij}c_i^\dagger c_j^\dagger c_j c_i.
:::

The real coefficients satisfy $v_{ij}=v_{ji}$. For distinct orbitals the interaction
operator is $n_i n_j$; its diagonal terms vanish by fermionic antisymmetry.
`h_0` contains $h$ and `h_int` contains $v$.

(theory-density-matrix)=
## Density convention

MeanFi stores the creation index **second**:

:::{math}
\rho_{ij}=\langle c_j^\dagger c_i\rangle,
\qquad
\langle\hat O\rangle=\operatorname{Tr}(O\rho).
:::

If $u_a$ is an eigenvector of the effective Hamiltonian with energy $\epsilon_a$,

:::{math}
\rho=\sum_a f(\epsilon_a-\mu)u_a u_a^\dagger,
\qquad
f(x)=\frac{1}{e^{x/kT}+1}.
:::

At zero temperature, levels below $\mu$ are occupied, levels above it are empty,
and a level exactly at $\mu$ has occupation $1/2$.

(theory-tight-binding)=
## Periodic systems

A dictionary key $R$ is a displacement in lattice coordinates. A block $h(R)$
multiplies $c_{m,L+R}^\dagger c_{n,L}$, summed over cells $L$. The Fourier convention is

:::{math}
H(k)=\sum_R h(R)e^{-ik\cdot R},
\qquad
\rho(R)=\int_{\mathrm{BZ}}\frac{d^dk}{(2\pi)^d}\,
             f(H(k)-\mu I)e^{ik\cdot R}.
:::

Thus $\rho_{mn}(R)=\langle c_{n,0}^\dagger c_{m,R}\rangle$ and a one-body observable
per cell is $\sum_R\operatorname{Tr}[O(R)\rho(-R)]$.
Momenta run over a $2\pi$ interval along each lattice direction. Finite systems
use the single key `()` and need no momentum integration.

(theory-mean-field)=
## Mean-field correction and energy

For a normal finite system, Hartree and Fock terms give the linear map

:::{math}
W[\rho]_{ij}=\delta_{ij}\sum_\ell v_{i\ell}\rho_{\ell\ell}
             -v_{ij}\rho_{ij},
\qquad H_{\mathrm{MF}}=h+W[\rho].
:::

Periodic systems use the same contractions over displacement blocks.
With an optional reference density, set $\delta\rho=\rho-\rho_{\mathrm{ref}}$
and use $W[\delta\rho]$. The internal energy per physical orbital is

:::{math}
u=\frac{\operatorname{Tr}(h\rho)}{N}
  +\frac{\operatorname{Tr}(W[\delta\rho]\delta\rho)}{2N}.
:::

Without a reference, $\delta\rho=\rho$. A reference changes the interaction
functional; the one-body energy still uses the actual density. The returned
energy is not an energy difference from the reference state.

With `superconducting=True`, the calculation also retains anomalous pairing
densities in an electron-first, $2N\times2N$ Bogoliubov–de Gennes representation.
Filling and normalization still count the $N$ physical orbitals. See the
[API guide](../meanfi.md) for normal and superconducting references.

(theory-filling)=
## Self-consistency

At fixed filling $\nu$, each evaluation finds $\mu$ from the charge integral
$N(\mu)=\nu$, then evaluates the requested density entries. For a normal finite
system, the SCF loop seeks

:::{math}
\rho=f\!\left(h+W[\rho-\rho_{\mathrm{ref}}]-\mu I\right),
:::

with momentum integration for periodic systems. EDIIS uses internal energy to
mix density states. Optional entropy gives $f_{\mathrm{free}}=u-kT\,s$ after the
calculation; it does not enter EDIIS. The [algorithm overview](../algorithms/index.md)
describes the numerical stages and stopping criteria.
