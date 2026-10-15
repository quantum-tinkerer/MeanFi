# Releasing 2.0.0b2

This release accepts FermiSimplex 0.2.0b1 and 0.2.0b2, whose native numerical API
is unchanged. The upper bound deliberately limits compatibility to the tested
betas. The checked-in Pixi lockfile keeps the already-published b1 wheels so CI
can run before FermiSimplex b2 is published. Fresh pip installations select b2
once it is available. The browser runtime explicitly installs the b2 Wasm wheel.

## Merge and tag

1. Merge and release FermiSimplex `v0.2.0b2` first. Wait for its PyPI upload and
   the Pyodide wheel attached to the matching GitHub release.
2. Merge this MeanFi release preparation into main and wait for its pipeline.
3. Tag the tested merge commit:

   ```sh
   git tag -a v2.0.0b2 -m "MeanFi 2.0.0b2"
   git push origin v2.0.0b2
   ```

The existing GitLab tag pipeline builds the source and pure-Python wheel and
publishes them to PyPI after tests, wheel checks, documentation and pre-commit
checks pass. The existing `PYPI_TOKEN` CI variable is used; no manual upload is
needed. The package version is stored in `meanfi/_version.py`, not inferred from
the tag.

Verify in a clean Python 3.11, 3.12 or 3.13 environment:

```sh
python -m pip install "meanfi==2.0.0b2" "fermisimplex==0.2.0b2"
python -m pip check
```

## Browser consumers

Use the ordinary MeanFi `2.0.0b2` pure-Python wheel with FermiSimplex's
`fermisimplex-0.2.0b2-cp313-cp313-pyodide_2025_0_wasm32.whl` release asset
and Pyodide 0.28.3. Load Pyodide's NumPy/SciPy packages and install the
FermiSimplex wheel before installing MeanFi. No alternate MeanFi source tree,
numerical patch, or separate MeanFi browser wheel is needed.
