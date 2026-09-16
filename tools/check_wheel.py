"""Build a distribution and exercise its installed wheel outside the checkout.

Run in the core test-py312 environment, or with --sparse in test-sparse.
By default, reuse the active environment's dependencies for an offline smoke test.
Use --clean to resolve and install dependencies into an isolated virtual environment;
this requires network access and the native compiler supplied by Pixi.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


SMOKE = r"""
from dataclasses import replace
import importlib.util
from importlib.metadata import metadata
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_array
from scipy.special import entr, expit
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
    (meanfi.FermiSimplex(nk=33), 0.),
    (meanfi.FermiSimplex(), 0.),
    (meanfi.UniformGrid(nk=32), .2),
    (meanfi.UniformGrid(), .2),
):
    result = meanfi.density_matrix(h, filling=.5, kT=temperature,
                                  keys=[(0,)], integration=integration)
    assert abs(result.filling - .5) < 1e-5
    assert np.isfinite(result.band_energy)
    assert result.entropy >= 0
for superconducting in (False, True):
    model = meanfi.Model(h, {(0,): np.zeros((1, 1))}, filling=.5,
                        kT=.2, superconducting=superconducting)
    result = meanfi.solver(model, model.random_meanfield(rng=0, scale=0),
                          integration=meanfi.UniformGrid(nk=32))
    assert result.converged
    assert np.isfinite(result.internal_energy)
    assert abs(result.free_energy - (result.internal_energy - .2 * result.entropy)) < 1e-12
    density = meanfi.density_matrix(model, mean_field=result.mean_field, keys=[(0,)],
                                   integration=meanfi.UniformGrid(nk=32))
    assert density.to_tb()[(0,)].shape == ((2, 2) if superconducting else (1, 1))
cold = meanfi.Model(h, {(0,): np.zeros((1, 1))}, filling=.5)
assert meanfi.solver(cold, cold.random_meanfield(rng=0)).free_energy is not None
grid = meanfi.tb_to_kgrid(h, (8,))
np.testing.assert_allclose(meanfi.tb_to_kgrid(meanfi.kgrid_to_tb(grid), (8,)), grid, atol=1e-14)

# Distinct coupled bands exercise actual selected inverses, not a constant fit.
sparse_h = {key: csr_array(np.kron(value, np.eye(2))) for key, value in h.items()}
sparse_h[(0,)] = csr_array([[0., .2], [.2, 0.]])
try:
    result = meanfi.density_matrix(
        sparse_h, filling=.86, kT=.2, keys=[(0,)],
        tol=replace(meanfi.default_solver_tolerances(1e-3), filling_residual=1e-7),
        integration=meanfi.UniformGrid(nk=32, matrix_function=meanfi.RationalFOE()),
    )
except ImportError as exc:
    assert not sparse_enabled and "meanfi[sparse]" in str(exc)
else:
    assert sparse_enabled
    energies = -2 * np.cos(2 * np.pi * np.arange(32) / 32)[:, None] + np.array([-.2, .2])
    assert abs(np.mean(np.sum(expit((result.mu - energies) / .2), axis=1)) - .86) < 1e-7
    occupations = expit((result.mu - energies) / .2)
    assert abs(result.band_energy - np.mean(energies * occupations)) < 5e-5
    assert result.errors.entropy is None  # A prescribed periodic mesh has no total estimate.
    assert np.isfinite(result.entropy)
    finite = meanfi.density_matrix_at_mu(
        {(): sparse_h[(0,)]}, mu=.13, kT=.2, keys=[()], integration=meanfi.UniformGrid(), tol=1e-9,
    )
    p = expit((.13 - np.linalg.eigvalsh(sparse_h[(0,)].toarray())) / .2)
    assert abs(finite.entropy - np.mean(entr(p) + entr(1-p))) <= finite.errors.entropy + 1e-12
    lean = meanfi.density_matrix_at_mu(
        {(): sparse_h[(0,)]}, mu=.13, kT=.2, keys=[()], integration=meanfi.UniformGrid(),
        compute_free_energy=False,
    )
    assert lean.entropy is lean.errors.entropy is None

print("Installed wheel passed:", "sparse extra" if sparse_enabled else "core without MUMPS")
"""


def main():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="install dependencies from scratch instead of borrowing the active environment",
    )
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
            [sys.executable, "-m", "venv"]
            + ([] if args.clean else ["--system-site-packages"])
            + [str(environment)],
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
            ]
            + ([] if args.clean else ["--no-index", "--no-deps", "--ignore-installed"])
            + [f"{wheel}[sparse]" if args.sparse else str(wheel)],
            cwd=work,
            check=True,
        )
        if args.clean:
            subprocess.run([str(python), "-m", "pip", "check"], cwd=work, check=True)
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
