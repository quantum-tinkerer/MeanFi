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
pixi run -e test-py312 check-wheel
pixi run -e test-sparse check-wheel-sparse
pixi run -e test-py312 check-install
pixi run -e precommit pre-commit run --all-files
```

The first two wheel checks build the source distribution and wheel, install the
wheel outside the checkout, and exercise normal/BdG calculations, both
integration families, thermodynamics, and sparse availability. They borrow the
active environment's dependencies and install the wheel offline.

`check-install` runs the same checks in a virtual environment with no inherited
site packages. It resolves and installs dependencies from the wheel metadata,
including the pinned FermiSimplex source, and runs `pip check`. This requires
network access and the native toolchain; it verifies Python dependency
installation rather than providing an independent operating-system toolchain.
CI runs all three installation checks. Pass `--clean --sparse` to
`tools/check_wheel.py` to also exercise installation of the sparse extra when
MUMPS headers, libraries, and a working `pkg-config` are available. The
`test-sparse` environment supplies prebuilt bindings and native libraries;
rebuilding [python-mumps](https://pypi.org/project/python-mumps/) from PyPI also
requires `pkg-config` to find MUMPS's metadata (for a Pixi environment, under
`$CONDA_PREFIX/lib/pkgconfig`).

## FermiSimplex release prerequisite

MeanFi pins FermiSimplex commit
`7921df11518f45505bd09ff32497f50d75ba4bce` because it provides the density,
charge, and band-energy API used by the simplex adapter. That revision identifies
itself as 0.2.0; the published [PyPI distribution](https://pypi.org/project/FermiSimplex/)
is currently 0.1.0 and cannot replace it.

The immutable source dependency supports reproducible checkout and direct-wheel
installation. [PyPI rejects direct URL dependencies][direct-urls], so a PyPI release
of MeanFi first requires a compatible FermiSimplex release. Then
replace the Git requirement with that tested release version, refresh the lock,
and rerun the installation and numerical checks. Do not downgrade the dependency
or remove band-energy support to make metadata uploadable.

[direct-urls]: https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies
