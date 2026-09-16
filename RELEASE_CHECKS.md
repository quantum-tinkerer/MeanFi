# Release checks

Run these checks from the repository root before a release:

- `pixi run tests-all`: core suites on Python 3.11, 3.12 and 3.13, plus sparse
  and Kwant integration. Numerical suites compare against exact or independently
  converged references and treat unexpected warnings as errors.
- `pixi run -e test-sparse tests-perf-slow`: extended numerical regressions,
  required after changes to approximation or integration algorithms.
- `pixi run -e docs docs-build`: strict documentation build. After API changes,
  force notebook execution with `-D nb_execution_mode=force` in Sphinx options.
- `pixi run -e test-sparse python examples/api_walkthrough.py --sparse --kwant`:
  exercise the public workflows, including normal and superconducting references.
- `pixi run -e test-py312 check-wheel` and
  `pixi run -e test-sparse check-wheel-sparse`: build from the source distribution
  and check installed wheels outside the checkout.
- `pixi run -e test-py312 check-install`: clean dependency installation.
- `pixi run -e precommit pre-commit run --all-files`: formatting and static checks.

EDIIS uses internal energy and never switches methods automatically. Density
accuracy controls integration; `matrix_function_tol` controls AAA pole selection.
Adaptive UniformGrid compares coarse and fine grids only. Tests include aliasing
limitations and independent dense checks of the matrix-function estimate. Sparse entropy is an
estimate on those poles with a separately reported approximation error; it has
no accuracy target and does not affect EDIIS. Prescribed meshes do not estimate
Brillouin-zone integration error.

Benchmark reports record the tested revision, numerical errors, hardware/thread
conditions and timings. Dated reports under `performance/benchmarks/release_results/`
are historical evidence, not a statement about the current implementation.
The protected `performance/benchmarks/periodic_study/` is retained unchanged.

Publication still requires a release version and a distribution target that
accepts the pinned FermiSimplex Git dependency. Running checks does not publish.
