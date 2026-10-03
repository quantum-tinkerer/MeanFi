# Development and release

Use `pixi install --locked` from a checkout to reproduce the development
environment. MeanFi itself is pure Python. Its required FermiSimplex dependency
builds a native extension; Pixi supplies the compiler and build tools. The
[design document](design.md) describes the implementation contracts and verification
assumptions.

For `pip install .`, provide Git and a C++20 compiler. FermiSimplex's build
isolation installs its declared Python build tools. Native builds use bundled
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

The rhombohedral-graphene tutorial executes its setup during the docs build,
but marks the four-point SCF cell `skip-execution` because it exceeds the
routine build budget. Run that cell interactively for the full study. The
physics tests independently check its compiled Hamiltonian, interactions and
exact density limits; published docs do not present an unexecuted SCF result.

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

MeanFi pins an immutable FermiSimplex commit in `pyproject.toml` and `pixi.lock`.
It provides the shared occupation enclosure, certificate reuse, adaptive
polynomial density cubature and retained spectral snapshots used by the adapter.
Checkout and direct-wheel installations use this same source dependency.

[PyPI rejects direct URL dependencies][direct-urls]. Before publishing MeanFi
there, replace the Git requirement with a compatible FermiSimplex release,
refresh the lock and rerun the installation and numerical checks.

[direct-urls]: https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies

The published FermiSimplex 0.1.0 is not compatible: its `SpectralMesh` lacks
`integrate_density_components` and `occupied_weights`, both required by this
adapter. Keep the working Git pin until a release includes these APIs; changing
only the dependency version would break zero-temperature calculations.
