# Details

## Ground state definition
Before we proceed, we must find a way to simply represent the ground state of the system.
Assume there exists a non-interacting system $\hat{H}_\text{mf}$ whose groundstate closely resembles the interacting groundstate $| 0 \rangle$:

$$
| 0 \rangle \approx \Pi_{E_i \leq \mu } b_i^\dagger |\textrm{vac}\rangle,
$$

where $|\textrm{vac}\rangle$ is the vacuum state,  $b_i^\dagger$ is the creation operator of eigenstate $i$ with energy $E_i$ of $\hat{H}_\text{mf}$ and $\mu$ is the Fermi level.
We relate the $b_i^\dagger$ particles to our original basis via a unitary transformation:

$$
c_i^\dagger = \sum_{k} U_{ik} b_k^\dagger.
$$

## Normal ordering and contractions

Before proceeding, we define the *normal ordering* operation, $:ABC...:$, as a sorting of operators such that all the creation operators $b_i^\dagger$ below the Fermi level are to the left of the annihilation $b_i$ operators below the Fermi levels, and vice versa for the operators above the Fermi level.
Whenever the normal ordered operators acts on the ground  state, it gives zero:

$$
:ABC...: | 0 \rangle = 0.
$$

Lastly, we define the *contraction* of two operators $A$ and $B$ as:

$$
\overline{AB} = \hat{A}\hat{B} - :AB:.
$$

## Expansion of the interaction term
We utilize Wick's theorem to expand the interaction term:

$$
c_i^\dagger c_j^\dagger c_l c_k = :c_i^\dagger c_j^\dagger c_l c_k: \\
+ \overline{c_i^\dagger c_k} :c_j^\dagger c_l: - \overline{c_i^\dagger c_l} :c_j^\dagger c_k: - \overline{c_j^\dagger c_k} :c_i^\dagger c_l: + \overline{c_j^\dagger c_l} :c_i^\dagger c_k: + \overline{c_i^\dagger c_j^\dagger} :c_l c_k: + \overline{c_l c_k} :c_i^\dagger c_j^\dagger: \\
- \overline{c_i^\dagger c_l} \overline{c_j^\dagger c_k} +\overline{c_i^\dagger c_k} \overline{c_j^\dagger c_l}
+\overline{c_i^\dagger c_j^\dagger} \overline{c_l c_k}.
$$

## Mean-field approximation to Wick terms

We are now able to apply the mean-field approximation to the Wick terms.
Lets first apply this to the first term in the expansion:

$$
:c_i^\dagger c_j^\dagger c_l c_k: = \langle :c_i^\dagger c_j^\dagger c_l c_k: \rangle + \delta:c_i^\dagger c_j^\dagger c_l c_k: \approx 0,
$$

where the first term in the second equality is zero due to the normal ordering operation and the second term is zero since we assume deviations from the mean-field ground state are small.

Next, we apply the mean-field approximation to the second term in the expansion:

$$
\overline{c_i^\dagger c_k} :c_j^\dagger c_l: \approx \\
\langle \overline{c_i^\dagger c_k} \rangle :c_j^\dagger c_l: + \overline{c_i^\dagger c_k} \langle :c_j^\dagger c_l: \rangle - \langle  \overline{c_i^\dagger c_k} \rangle \langle :c_j^\dagger c_l: \rangle = \\
\langle c_i^\dagger c_k \rangle :c_j^\dagger c_l:.
$$

Repeating this for all terms in the expansion and collecting we find:

$$
c_i^\dagger c_j^\dagger c_l c_k \approx \\
\langle c_i^\dagger c_k \rangle c_j^\dagger c_l - \langle c_i^\dagger c_l \rangle c_j^\dagger c_k - \langle c_j^\dagger c_k \rangle c_i^\dagger c_l + \langle c_j^\dagger c_l \rangle c_i^\dagger c_k + \langle c_i^\dagger c_j^\dagger \rangle c_l c_k +  \langle c_l c_k \rangle c_i^\dagger c_j^\dagger
\\
- \langle c_i^\dagger c_l \rangle \langle c_j^\dagger c_k \rangle + \langle c_i^\dagger c_k \rangle  \langle c_j^\dagger c_l \rangle  + \langle c_i^\dagger c_j^\dagger \rangle \langle  c_l c_k \rangle.
$$
