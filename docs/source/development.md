# Development and release

Use `pixi install --locked` from a checkout to reproduce the development
environment. MeanFi itself is pure Python. Its required FermiSimplex dependency
builds a native extension; Pixi supplies the compiler and build tools. The
[design document](design.md) describes the implementation contracts and verification
assumptions.

For `pip install .`, install the matching FermiSimplex beta from PyPI first, or
provide its wheel with `--find-links` when testing unpublished distributions.
Building FermiSimplex from source requires Git and a C++20 compiler; its build
isolation installs the declared Python build tools. Native builds use bundled
OpenBLAS on Linux/Windows and Accelerate on macOS, and vendor the AdaptiveSimplex
mesh engine. No separate mesh-engine checkout is needed. The optional `sparse` and `kwant`
extras also need their native libraries
when built from source; use `pixi install -e test-sparse` for a managed setup.

## Repeatable checks

```bash
pixi run tests-all
pixi run -e test-sparse tests-perf-slow
pixi run -e docs docs-build
pixi run -e test-py312 build
pixi run -e precommit pre-commit run --all-files
```

The sparse CI coverage job runs the complete suite, including the heavier
`perf_slow` numerical-reference checks. Local `tests` tasks omit that group;
`tests-perf-slow` runs it separately.

The rhombohedral-graphene tutorial executes all four SCF calculations and
renders their bands during the docs build. Documentation execution has no time
limit. If a build runs longer than ten minutes, inspect the active calculation
and its progress before proceeding; do not skip cells or relax tolerances to
hide a regression. The tutorial uses one validated tolerance policy and random
start for every parameter point.

The test task writes coverage and test reports to ignored `build/test-reports/`.
CI also builds the source distribution and wheel, installs the wheel into a
separate environment, then checks its import location and an exact density
calculation. That small smoke check catches packaging mistakes; the numerical
suite exercises the algorithms and optional backends.

## Benchmarks

```bash
pixi run benchmark
pixi run -e test-sparse benchmark --sparse --sizes 100 200 --output build/benchmarks.json
```

The benchmark compares density and EDIIS workloads with known numerical or
analytic references, including full sparse fixed-filling searches. It reports
timings, largest entry errors, charge probes and iteration counts; optional JSON
includes chemical-potential errors, work counters and environment details. See
`performance/README.md` in the checkout for the workloads. Generated results
belong in ignored `build/` or CI artifacts, not in version control.

## FermiSimplex release prerequisite

MeanFi 2.0.0b1 requires exactly FermiSimplex 0.2.0b1. Its distribution metadata
uses that version requirement, without a Git URL, so pip installs the matching
published package and can use its platform wheel.

The Pixi development environment and lock file use the same published beta.
FermiSimplex supplies wheels for Python 3.11–3.13 on Linux x86-64, Windows AMD64,
and Intel/Apple Silicon macOS. No Git source override is needed.
Do not restore a Git URL in `[project.dependencies]`:
[PyPI rejects direct URL dependencies][direct-urls].

[direct-urls]: https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies

FermiSimplex 0.1.0 is not compatible: its `SpectralMesh` lacks
`integrate_density_components` and `occupied_weights`, both required by this
adapter. Publish and verify FermiSimplex 0.2.0b1 before publishing MeanFi 2.0.0b1.
The compatible source APIs and the complete presentation notebook must pass
against the built wheels before release tags are created.

For a local, unpublished package check:

```bash
python -m build
python -m twine check dist/*
python -m venv build/beta-env
build/beta-env/bin/python -m pip install --only-binary=:all: --find-links dist "meanfi==2.0.0b1"
build/beta-env/bin/python -m pip check
```

These commands build and install local artifacts; they do not upload packages.
The existing tag-triggered CI publishing jobs are reserved for an approved
release. Preparing a beta branch does not require a release tag or a PyPI upload.
