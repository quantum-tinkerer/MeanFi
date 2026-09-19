# AI coding guidelines

Use a concise design document to explain the goal, the algorithm, its assumptions, and how the main concepts fit together. Let it guide planning and implementation, and keep it current as the code evolves. The code should make those concepts and relationships easy to recognize. Prioritize clarity and correctness in the code, keep the central logic visible, and leave implementation details in the code.

## Design and structure

- Before adding extra code or special cases, consider whether a simpler representation, contract, or algorithm removes the need. Avoid changes that simplify one part by making the overall design harder to understand.
- Organize code as an intuitive hierarchy of cohesive modules and small, focused files. Each folder and file should have a clear purpose. Split long files along meaningful responsibilities.
- Give each core operation one canonical implementation and reuse it directly. Minimize shims, wrappers, indirection, and near-duplicate implementations.
- Prefer small, focused functions. Extract a block when its name, inputs, and outputs make the surrounding logic easier to understand. Keep readable short blocks inline; prefer local helpers for logic needed only within one function.
- Use classes for intuitive concepts in the design when they simplify interaction, maintain validity, or coordinate related behavior. Prefer functions for standalone operations.

## Readability

- Use descriptive names consistently for the same concepts in the design document and the code. Prefer shorter names when clarity is preserved. Follow the project's language and formatting conventions.
- Prefer clear code and names. Add brief comments when intent or constraints cannot be made clear through the code itself.

## Correctness and verification

- Use sound representations and algorithms to eliminate invalid states where practical. Establish guarantees at clear boundaries, then rely on them internally. Handle failures the design cannot eliminate.
- Choose mutable or immutable objects according to the design. Keep units immutable when they should not change. Validate mutations where needed to preserve guarantees, keeping checks lightweight and avoiding redundant work or unnecessary runtime and memory costs.
- Include at least one repeatable test using the implementation to demonstrate the intended result under the assumptions stated in the design document.
- For numerical algorithms in mathematics or physics, compare against an exact solution or a numerical reference whose accuracy is established, such as a converged dense computation. Report the error. When no tolerance is specified, choose and document one based on the intended use or a justified error bound.

## Working rules

- Refactor as needed to improve the design. Preserve existing interfaces and backward compatibility only when explicitly requested; always satisfy the task's intended requirements.
- Ask before adding external dependencies, and explain why each is warranted.

## Recorded user decisions (2026-09-17)

- FermiSimplex `nk` is intended as an adaptive point budget: refine adaptively
  until approximately the requested count, subject to simplex granularity. It
  must not silently select a prescribed regular mesh. The current implementation
  violates this intent. The user explicitly deferred fixing it; record the issue
  and do not change its behavior until that work is requested.
- Energy accuracy must work on independently adaptive meshes. Do not require
  compatible/shared meshes as the solution. Keep EDIIS for the paper example.
- Investigate the current failure using saved states and short, bounded checks.
  Do not launch another long SCF run without a new user request for that work.

- For the Fig. 2 tutorial, do not engineer branch-specific starting fields.
  The user permits changing the starting seed and a single guess amplitude
  (or random-guess scale), not adding a tuned screening field.

## Recorded user decisions (2026-09-19)

- Callable Hamiltonian values are the user's responsibility. Avoid repeated
  finite/Hermitian matrix scans; keep cheap structural checks and one-time
  validation that does not materially affect evaluation performance.

- EDIIS compares relative internal energies at zero temperature and free
  energies at finite temperature (latest user decision). Use density response
  for expensive zero-temperature simplex energies and cheap integrated energies
  and entropy on thermal UniformGrid/BdG. Include signed filling extrapolation
  once, within the comparison. Reuse computed entropy in the final result.
- Expensive absolute energy is evaluated once on the final retained mesh; cheap
  absolute energies can be retained on each evaluation. Keep relative offsets
  separate from reported physical energies.
- Show Numba compilation inline in the graphene tutorial. Numba is a tutorial/docs
  dependency, not a MeanFi runtime dependency.

- Keep the graphene tutorial free of threadpool controls. Show MeanFi model and
  solver construction explicitly. Use the same random-guess seed/scale and the
  same tolerance policy for every parameter point, without phase-specific seeds.
  Do not claim that generic starts reproduce all named branches of Fig. 2.
