"""Build a distribution and exercise its installed wheel outside the checkout.

Run in the core test-py312 environment, or with --sparse in test-sparse.
Dependencies are supplied by the active environment; wheel installation is offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


SMOKE = r"""
import importlib.util
from importlib.metadata import metadata
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_array
from scipy.special import expit
import meanfi

assert Path(meanfi.__file__).is_relative_to(Path(sys.prefix))
assert importlib.util.find_spec("stateful_quadrature") is None
sparse_enabled = sys.argv[1] == "sparse"
assert (importlib.util.find_spec("mumps") is not None) == sparse_enabled
requirements = metadata("meanfi").get_all("Requires-Dist")
assert not any("stateful" in item for item in requirements)
mumps_requirement = next(item for item in requirements if "python-mumps" in item)
assert "extra == 'sparse'" in mumps_requirement

h = {(0,): np.zeros((1, 1)), (1,): np.array([[-1.]]), (-1,): np.array([[-1.]])}
for integration, temperature in (
    (meanfi.AdaptiveSimplex(nk=33), 0.),
    (meanfi.AdaptiveSimplex(), 0.),
    (meanfi.PeriodicGrid(nk=32), .2),
    (meanfi.PeriodicGrid(), .2),
):
    result = meanfi.density_matrix(h, filling=.5, kT=temperature,
                                  keys=[(0,)], integration=integration)
    assert abs(result.filling - .5) < 1e-5
for superconducting in (False, True):
    model = meanfi.Model(h, {(0,): np.zeros((1, 1))}, filling=.5,
                        kT=.2, superconducting=superconducting)
    result = meanfi.solver(model, model.random_meanfield(rng=0, scale=0),
                          integration=meanfi.PeriodicGrid(nk=32))
    assert result.converged

sparse_h = {key: csr_array(value) for key, value in h.items()}
try:
    result = meanfi.density_matrix(
        sparse_h, filling=.43, kT=.2, keys=[(0,)], filling_tol=1e-7,
        integration=meanfi.PeriodicGrid(nk=32, matrix_function=meanfi.RationalFOE()),
    )
except ImportError as exc:
    assert not sparse_enabled and "meanfi[sparse]" in str(exc)
else:
    assert sparse_enabled
    energies = -2 * np.cos(2 * np.pi * np.arange(32) / 32)
    assert abs(np.mean(expit((result.mu - energies) / .2)) - .43) < 1e-7
print("Installed wheel passed:", "sparse extra" if sparse_enabled else "core without MUMPS")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="meanfi-wheel-check-") as directory:
        work = Path(directory)
        artifacts = work / "dist"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(artifacts),
            ],
            cwd=root,
            check=True,
        )
        wheel = next(artifacts.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            assert not any(
                name.startswith(("performance/", "meanfi/tests/"))
                for name in archive.namelist()
            )
        environment = work / "environment"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            check=True,
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--ignore-installed",
                f"{wheel}[sparse]" if args.sparse else str(wheel),
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", SMOKE, "sparse" if args.sparse else "core"],
            cwd=work,
            check=True,
        )
        (root / "dist").mkdir(exist_ok=True)
        for artifact in artifacts.iterdir():
            shutil.copy2(artifact, root / "dist" / artifact.name)


if __name__ == "__main__":
    main()
