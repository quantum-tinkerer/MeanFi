# Adaptive Quadrature Reuse

## Problem

We want to evaluate a family of parameter-dependent integrals

$$
I(\lambda) = \int_\Omega f(x; \lambda)\, dx
$$

with adaptive quadrature, while reusing expensive work across different values of $\lambda$.

The key structure is

$$
f(x; \lambda) = \Psi(x, K(x), \lambda),
$$

where:

- $x \in \Omega$ is the integration variable,
- $\lambda$ is a dynamic parameter,
- $K(x)$ is an expensive quantity that does not depend on $\lambda$,
- $\Psi$ is cheap once $K(x)$ is known.

The goal is to keep the usual per-call adaptive quadrature error control, while avoiding repeated evaluations of the expensive kernel $K(x)$.

## Basic objects

A **region** $R$ is a subdomain of $\Omega$.

Examples:

- in 1D, $R$ is an interval,
- in 2D, $R$ may be a rectangle,
- in higher dimensions, $R$ is typically a hyperrectangle.

Adaptive quadrature recursively subdivides $\Omega$ into smaller regions.
The regions that are currently active and not subdivided further are the **leaves**.

The **state** is the saved adaptive mesh together with all cached point data.

## Local rule

Assume each region $R$ has an embedded rule with nodes $x_{R,j}$ and two weight sets:

$$
Q_R^{(h)}(\lambda) = \sum_j w_{R,j}^{(h)} \, \Psi(x_{R,j}, K(x_{R,j}), \lambda),
$$

$$
Q_R^{(l)}(\lambda) = \sum_j w_{R,j}^{(l)} \, \Psi(x_{R,j}, K(x_{R,j}), \lambda).
$$

The local error indicator is

$$
e_R(\lambda) = \|Q_R^{(h)}(\lambda) - Q_R^{(l)}(\lambda)\|.
$$

If $\mathcal{L}$ is the current leaf set, then

$$
\widehat I(\lambda) = \sum_{R \in \mathcal{L}} Q_R^{(h)}(\lambda),
$$

$$
\widehat E(\lambda) \approx \sum_{R \in \mathcal{L}} e_R(\lambda).
$$

## Reuse idea

The expensive part is

$$
x \mapsto K(x).
$$

When $\lambda$ changes, the cached values of $K(x)$ remain valid.
Only the cheap map $\Psi$ changes.

So reuse means:

1. keep the current adaptive partition,
2. keep all previously evaluated kernel values $K(x)$,
3. for a new $\lambda$, recompute local quadrature estimates and local errors from cached data,
4. refine only if the current $\lambda$ still fails tolerance.

This preserves exact point evaluations while letting the adaptive mesh persist across calls.

## What is reused

Allowed reuse:

- previously visited quadrature nodes,
- cached values of $K(x)$ at those nodes,
- the current adaptive tree as the starting mesh for the next call.

Not included in the base design:

- interpolation between cached points,
- nearest-neighbor substitution,
- reuse of old local error estimates after $\lambda$ changes.

Only exact cached point data is reused.

## Contract

For each requested $\lambda$, the integrator should:

1. start from the saved state,
2. recompute local estimates and errors on the current leaves using cached kernel data,
3. form a global estimate and error indicator,
4. check convergence using a standard condition such as

$$
\widehat E(\lambda) \le \mathrm{atol} + \mathrm{rtol}\,\|\widehat I(\lambda)\|,
$$

5. if needed, subdivide selected leaves and evaluate $K(x)$ only at genuinely new nodes,
6. return the estimate for this $\lambda$ together with the updated state.

The accuracy guarantee is per call, for the specific value of $\lambda$ being requested.

## Minimal abstraction

The design can be expressed through two user-facing components:

Static kernel:

$$
K : x \mapsto \text{payload}(x)
$$

Dynamic evaluator:

$$
\Psi : (x, \text{payload}, \lambda) \mapsto \text{integrand value}
$$

The integrator is then a stateful operator

$$
(\widehat I, \widehat E, S') = \mathrm{Integrate}(S, \lambda, \mathrm{atol}, \mathrm{rtol}).
$$

## Summary

The design problem is:

> Build an adaptive quadrature engine for
>
> $$
> I(\lambda)=\int_\Omega \Psi(x, K(x), \lambda)\, dx
> $$
>
> that returns an error-controlled result for each requested $\lambda$, while reusing exact cached evaluations of the expensive static kernel $K(x)$ and reusing the existing adaptive partition as a warm start.
