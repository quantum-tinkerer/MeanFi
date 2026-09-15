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
