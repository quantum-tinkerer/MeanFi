# Self-consistent mean field algorithm

## Mean-field approximation

The full hamiltonian is:

$$
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ijkl} v_{ijkl} c_i^\dagger c_j^\dagger c_l c_k
$$

We assume the dominant part of the ground state wavefunction comes from $\hat{H}_0$. Let's assume $b_i$ operators diagonalize the unperturbed hamiltonian:

$$
c_i^\dagger = \sum_{k} U_{ik} b_k^\dagger,
$$

such that the unperturbed groundstate wavefunction is:

$$
| 0 \rangle = \Pi_{E_i \leq \mu } b_i^\dagger |\textrm{vac}\rangle.
$$

Based on this definition, we define the normal ordering operator $:ABC...:$ such that it fulfills:

$$
:ABC...: | 0 \rangle = 0
$$

which practically means it orders $b_i$ operators based on whether its above or below the Fermi level $\mu$.

Under this definition of normal ordering, we define the Wick's expansion of the interaction term:

$$
\begin{multline}
c_i^\dagger c_j^\dagger c_l c_k = :c_i^\dagger c_j^\dagger c_l c_k: \\+  \overline{c_i^\dagger c_j^\dagger} :c_l c_k: + \overline{c_i^\dagger c_k} :c_j^\dagger c_l: - \overline{c_i^\dagger c_l} :c_j^\dagger c_k: + \overline{c_l c_k} :c_i^\dagger c_j^\dagger: - \overline{c_j^\dagger c_k} :c_i^\dagger c_l: + \overline{c_j^\dagger c_l} :c_i^\dagger c_k: \\
\overline{c_i^\dagger c_j^\dagger} \overline{c_l c_k} - \overline{c_i^\dagger c_l} \overline{c_j^\dagger c_k} + \overline{c_i^\dagger c_k} \overline{c_j^\dagger c_l}
\end{multline}
$$

where the overline defines Wick's contraction:

$$
\overline{AB} = AB - :AB:.
$$

The expectation value of the interaction with respect to the $| 0 \rangle$ is:

$$
\langle 0 | c_i^\dagger c_j^\dagger c_l c_k | 0 \rangle = \langle 0 | \overline{c_i^\dagger c_j^\dagger} \overline{c_l c_k} - \overline{c_i^\dagger c_l} \overline{c_j^\dagger c_k} + \overline{c_i^\dagger c_k} \overline{c_j^\dagger c_l}  | 0 \rangle
$$

where we can forget about all the normal ordered states since those give zero acting on the unperturbed groundstate. To evaluate this further, we utilize the mean-field approximation:

$$
A B \approx \langle A \rangle B + A \langle B \rangle - \langle A \rangle \langle B \rangle
$$

onto the contractions such that we get:

$$
\langle  c_i^\dagger c_j^\dagger c_l c_k \rangle \approx \langle c_i^\dagger c_j^\dagger \rangle \langle  c_l c_k \rangle + \langle c_i^\dagger c_k \rangle  \langle c_j^\dagger c_l \rangle - \langle c_i^\dagger c_l \rangle \langle c_j^\dagger c_k \rangle
$$

note $\langle A B \rangle \approx \langle A \rangle  \langle B \rangle$ assuming mean-field.

To consider excitations from the groundstate, we make use of the mean-field approximation defined above:

$$
\begin{multline}
c_i^\dagger c_j^\dagger c_l c_k \approx \\
\langle c_i^\dagger c_j^\dagger \rangle c_l c_k + \langle c_i^\dagger c_k \rangle c_j^\dagger c_l - \langle c_i^\dagger c_l \rangle c_j^\dagger c_k + \langle c_l c_k \rangle c_i^\dagger c_j^\dagger - \langle c_j^\dagger c_k \rangle c_i^\dagger c_l + \langle c_j^\dagger c_l \rangle c_i^\dagger c_k + \\
\langle c_i^\dagger c_j^\dagger \rangle \langle  c_l c_k \rangle + \langle c_i^\dagger c_k \rangle  \langle c_j^\dagger c_l \rangle - \langle c_i^\dagger c_l \rangle \langle c_j^\dagger c_k \rangle
\end{multline}
$$

Where we made use of the following operations:

$$
:c_i^\dagger c_j^\dagger c_l c_k: \approx 0
$$

$$
\overline{c_i^\dagger c_k} \overline{c_j^\dagger c_l} \approx \langle \overline{c_i^\dagger c_k} \rangle \overline{c_j^\dagger c_l} + \overline{c_i^\dagger c_k} \langle \overline{c_j^\dagger c_l} \rangle - \langle \overline{c_i^\dagger c_k} \rangle \langle \overline{c_j^\dagger c_i} \rangle =  \langle c_i^\dagger c_k \rangle \overline{c_j^\dagger c_l} + \overline{c_i^\dagger c_k} \langle c_j^\dagger c_l \rangle - \langle c_i^\dagger c_k \rangle \langle c_j^\dagger c_l \rangle
$$

$$
\overline{c_i^\dagger c_k} :c_j^\dagger c_l: \approx \langle \overline{c_i^\dagger c_k} \rangle  :c_j^\dagger c_l: +  \overline{c_i^\dagger c_k} \langle :c_j^\dagger c_l: \rangle - \langle \overline{c_i^\dagger c_k} \rangle \langle :c_j^\dagger c_l: \rangle = \langle \overline{c_i^\dagger c_k} \rangle  :c_j^\dagger c_l:
$$


$$
\langle \overline{c_i^\dagger c_k} \rangle  = \langle c_i^\dagger c_k - :c_i^\dagger c_k: \rangle  = \langle c_i^\dagger c_k \rangle
$$


Without any superconducting terms, the form simplifies to:

$$
\begin{multline}
c_i^\dagger c_j^\dagger c_l c_k \approx
\langle c_i^\dagger c_k \rangle c_j^\dagger c_l - \langle c_i^\dagger c_l \rangle c_j^\dagger c_k - \langle c_j^\dagger c_k \rangle c_i^\dagger c_l + \langle c_j^\dagger c_l \rangle c_i^\dagger c_k + \\
\langle c_i^\dagger c_k \rangle  \langle c_j^\dagger c_l \rangle - \langle c_i^\dagger c_l \rangle \langle c_j^\dagger c_k \rangle
\end{multline}
$$

## Finite size

### Coulomb interaction

We simplify the interaction term through the MF approximation to get:

$$
V = \frac{1}{2}\sum_{ijkl} v_{ijkl} c_i^{\dagger} c_j^{\dagger} c_l c_k
\approx
\frac12 \sum_{ijkl} v_{ijkl} \left[ \langle c_i^{\dagger} c_k \rangle c_j^{\dagger} c_l - \langle c_j^{\dagger} c_k \rangle c_i^{\dagger} c_l - \langle c_i^{\dagger} c_l \rangle c_j^{\dagger} c_k + \langle c_j^{\dagger} c_l \rangle c_i^{\dagger} c_k \right]
$$
(assuming no superconductivity)

and an additional constant part:

$$
V_0 =  \frac{1}{2} \sum_{ijkl} v_{ijkl} \left(\langle c_j^{\dagger} c_l \rangle \langle c_i^{\dagger} c_k \rangle - \langle c_j^{\dagger} c_k \rangle \langle c_i^{\dagger} c_l \rangle \right).
$$

The interaction reads:

$$
v_{ijkl} = \iint w_{i}^*(r) w_{j}^*(r') V(r, r') w_{k}(r) w_l(r') dr dr' = \\
\iint  V(|r - r'|) w_{i}^*(r)w_{k}(r) w_{j}^*(r')  w_l(r') dr dr'
$$

whereas $w_i$ is a wannier function on site i (and corresponding dof). Whenever one interchanges $i \to j, k \to l$, the Coulomb term is preserved $v_{ijkl} = v_{jilk}$

To make things more understandable, we are also going to explicitly split up position and spin indices: $i \to i \times \sigma$. In this notation, the Coulomb integral reads:

$$
v_{ijkl}^{\sigma_i \sigma_j \sigma_k \sigma_l} =
\iint V(|r - r'|) w_{i\times\sigma_i}^{*} (r)w_{k \times \sigma_k}(r) w_{j \times \sigma_j}^{*}(r')  w_{l\times \sigma_l}(r') dr dr' \delta_{\sigma_i \sigma_k} \delta_{\sigma_{j} \sigma_l}
$$

On a fine tight-binding model, we have:

$$
v_{ijkl}^{\sigma_i \sigma_j \sigma_k \sigma_l} = v_{ij} \delta_{ik} \delta_{jl} \delta_{\sigma_i \sigma_k} \delta_{\sigma_j \sigma_l}
$$

where $v_{ij} = V(r_i, r_j)$.

We shall re-define $i$ index to absorb spin:

$$
\delta_{ik} \times \delta_{\sigma_{i} \sigma_{k}} \to \delta_{ik}
$$

in this notation the above reads:

$$
v_{ijkl} = v_{ij} \delta_{ik} \delta_{jl}
$$

The mean-field terms are:

$$
\langle c_i^{\dagger} c_j\rangle = \langle \Psi_F|c_i^{\dagger} c_j | \Psi_F \rangle
$$

whereas $|\Psi_F \rangle = \Pi_{i=0}^{N_F} b_i^\dagger |0\rangle$. To make sense of things, we need to transform between $c_i$ basis (position + internal dof basis) into the $b_i$ basis (eigenfunction of a given mean-field Hamiltonian):

$$
c_i^\dagger = \sum_{k} U_{ik} b_k^\dagger
$$

where

$$
U_{ik} = \langle{i|\psi_k} \rangle.
$$

That gives us:


$$
c_i^{\dagger} c_j = \sum_{k, l} U_{ik} U_{lj}^* b_k^\dagger b_{l}
$$

and its expectation value gives us the mean-field ... field $F_{ij}$:

$$
F_{ij} = \langle c_i^{\dagger} c_j\rangle =  \sum_{k, l} U_{ik} U_{lj}^* \langle \Psi_F| b_k^\dagger b_{l}| \Psi_F \rangle =  \sum_{k} U_{ik} U_{kj}^{*}
$$

whereas I assumed the indices for wavefunctions $k,l$ are ordered in terms of increasing eigenvalue. We pop that into the definition of the mean-field correction $V$:


$$
\begin{multline}
V_{nm} = \frac12 \sum_{ijkl} v_{ijkl} \langle n| \left[ \langle c_i^{\dagger} c_k \rangle c_j^{\dagger} c_l - \langle c_j^{\dagger} c_k \rangle c_i^{\dagger} c_l - \langle c_i^{\dagger} c_l \rangle c_j^{\dagger} c_k + \langle c_j^{\dagger} c_l \rangle c_i^{\dagger} c_k \right] |m\rangle = \\
 \frac12 \sum_{ijkl} v_{ijkl} \left[ +\delta_{jn}\delta_{lm} F_{ik} -\delta_{in}\delta_{lm} F_{jk} -\delta_{jn}\delta_{km} F_{il} + \delta_{in}\delta_{km} F_{jl} \right] = \\
\frac12 \left[ \sum_{ik} v_{inkm} F_{ik} - \sum_{jk} v_{njkm} F_{jk} - \sum_{il} v_{inml} F_{il} + \sum_{jl} v_{njml} F_{jl} \right] = \\
-\sum_{ij} F_{ij} \left(v_{inmj} - v_{injm} \right)
\end{multline}
$$

where I used the $v_{ijkl} = v_{jilk}$ symmetry from Coulomb.

For a tight-binding grid, the mean-field correction simplifies to:

$$
\begin{multline}
V_{nm} = - \sum_{ij} F_{ij} \left(v_{inmj} - v_{injm} \right) = \\
-\sum_{ij}F_{ij} v_{in} \delta_{im} \delta_{nj} + \sum_{ij}F_{ij} v_{in} \delta_{ij} \delta_{nm} = \\
-F_{mn} v_{mn} + \sum_{i} F_{ii} v_{in} \delta_{nm}
\end{multline}
$$

the first term is the exchange interaction whereas the second one is the direct interaction.

Similarly, the constant offset term reads:

$$
\begin{multline}
V_0 = \frac{1}{2} \sum_{ijkl} v_{ijkl} \left(F_{jl} F_{ik} - F_{jk} F_{il} \right) = \\
\frac{1}{2} \sum_{ijkl} v_{ij} \delta_{ik} \delta_{jl} \left(F_{jl} F_{ik} - F_{jk} F_{il}\right) \\
= \frac{1}{2} \sum_{ij} v_{ij} \left(F_{ii} F_{jj} - F_{ji} F_{ij}\right)
\end{multline}
$$

where we identify the first term as the exchange (mixes indices) and the right one as the direct (diagonal in indices).

## Translational Invariance

The above assumed a finite tight-binding model - all $nm$-indices contain the position of all atoms (among other dof). In this section tho we want to consider an infinite system with translational invariance.

To begin with we deconstruct a general matrix $O_{nm}$ into the cell degrees of freedom ($nm$) and the position of the the cell itself ($ij$):

$$
O_{nm} \to O^{ij}_{nm}
$$

and we will Fourier transform the upper indices into k-space:

$$
O_{mn}(k) = \sum_{ij} O_{nm}^{ij} e^{-i k (R_i-R_j)}
$$

where I assumed $O$ (and thus all operators I will consider here) is local and thus diagonal in k-space.

Now lets rewrite our main result in the previous section using our new notation:

$$
V_{nm}^{ij} =-F_{mn}^{ij} v_{mn}^{ij} + \sum_{r,p} F_{pp}^{rr} v_{pn}^{rj} \delta_{nm} \delta^{ij}
$$

Lets first consider the second (direct) term. Lets express the corresponding $F$ term in k-space:

$$
F_{pp}^{rr} = \int e^{i k (R_r-R_r)} F_{pp}(k) dk = \int F_{pp}(k) dk
$$

Notice that in the final expression, there is no $rr$ dependence and thus this term is cell-periodic. Therefore, we shall redefine it as cell electron density $\rho$:
$$
F_{pp}^0 = F_{pp}(R = 0) = \int F_{pp}(k) dk
$$

Now since $\rho$ has no $r$ dependence, we can proceed with the sum:

$$
\sum_{r} v_{pn}^{rj} = \int v_{pn}(k) e^{ik R_j} \sum_{r} e^{-i k R_r} dk = \int v_{pn}(k) e^{ik R_j} \delta_{k, 0} dk = v_{pn}(0)
$$

We are finally ready to Fourier transform the main result. Invoking convolution theorem and the results above gives us:

$$
V_{nm}(k) = \sum_{p} F_{pp}^0 v_{pn}(0) \delta_{nm} -F_{mn}(k) \circledast v_{mn}(k) = V_n^D - F_{mn}(k) \circledast v_{mn}(k)
$$

which does make sense. The first term (direct) is a potential term coming from the mean-field and the second term (exchange) is purely responsible for the hopping.

The constant offset is:
$$
V_0 = \frac{1}{2} \sum_{r,s} \rho_r v_{rs}(0) \rho_s- \\ \frac{1}{2} tr\left[\int_{BZ} \left(F \circledast v\right)(k) F(k) dk \right]
$$

## Short-Range interactions

In the case of short-range interactions, it is much more convenient to go back to real space to both store objects and perform the operations. In real space the mean-field part of the Hamiltonian reads:

$$
V_{nm}(\mathbf{R}) = V_n^D \delta(\mathbf{R}) - F_{mn}(\mathbf{R}) v_{mn}(\mathbf{R})
$$

(the first term might need some prefactor from Fourier transformation)

where $\mathbf{R}$ is a sequence of integers representing real-space unit cell indices.

### Proposed Algorithm
Given an initial Hamiltonian $H_0 (R)$ and the interaction term $v(R)$ in real-space, the mean-field algorithm is the following:

0. Start with a mean-field guess in real-space: $V(R)$.
1. Fourier transform tight-binding model and the mean-field in real space to a given k-grid: $H_0(R) + V(R) \to H_0(k) + V(k)$
2. Diagonalize and evaluate the density matrix: $H_0(k) + V(k) \to F(k)$
3. Fourier transform the density matrix back to real-space: $F(k) \to F(R)$
4. Evaluate the new mean-field Hamiltonian $V(R)$ according to the equation above.
5. Evaluate self-consistency metric (could be based either on $V(R/k)$ or $F(R/k)$). Based on that, either stop or go back to 1 with a modified $V(R)$ starting guess.
