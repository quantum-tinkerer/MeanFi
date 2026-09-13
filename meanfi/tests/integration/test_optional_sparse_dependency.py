"""The dense and simplex package does not import the optional MUMPS runtime."""

from pathlib import Path
import subprocess
import sys


def test_core_runs_without_mumps_and_sparse_request_explains_extra():
    script = r"""
import importlib.abc
import sys

class NoMumps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mumps" or fullname.startswith("mumps."):
            raise ModuleNotFoundError("MUMPS intentionally unavailable", name="mumps")

sys.meta_path.insert(0, NoMumps())
import numpy as np
from scipy.sparse import csr_matrix
import meanfi

h = {(0,): np.diag([-0.4, 0.4]), (1,): np.eye(2) * 0.1, (-1,): np.eye(2) * 0.1}
for method, temperature in ((meanfi.PeriodicGrid(nk=16), 0.2), (meanfi.AdaptiveSimplex(nk=16), 0.0)):
    result = meanfi.density_matrix_at_mu(h, mu=0.0, kT=temperature, keys=[(0,)], integration=method)
    assert abs(result.filling - 1.0) < 1e-8
assert "mumps" not in sys.modules

try:
    meanfi.density_matrix_at_mu(
        {key: csr_matrix(value) for key, value in h.items()}, mu=0.0, kT=0.2,
        keys=[(0,)], integration=meanfi.PeriodicGrid(nk=4, matrix_function=meanfi.RationalFOE()),
    )
except ImportError as exc:
    assert "meanfi[sparse]" in str(exc), str(exc)
else:
    raise AssertionError("Sparse MUMPS path ran without its dependency")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
