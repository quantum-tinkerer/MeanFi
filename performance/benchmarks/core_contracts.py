"""Bounded accepted-guess benchmark against direct finite-matrix references.

Run sequentially with one BLAS/OpenMP thread. --checkout accepts the parent
revision as well as the current implementation; only public density APIs are used.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkout", type=Path, default=Path(__file__).resolve().parents[2]
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.checkout.resolve()))
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.special import expit  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import meanfi as mf  # noqa: E402
from meanfi.tb.bdg import assemble_bdg_tb  # noqa: E402
from meanfi.tb.ops import to_dense  # noqa: E402

records = []
with threadpool_limits(1):
    for n in (100, 200):
        for bdg in (False, True):
            h = sparse.diags([-np.ones(n - 1), -np.ones(n - 1)], [-1, 1], format="csr")
            interaction = {(0,): -0.3 * h}
            model = mf.Model(
                {(0,): h}, interaction, filling=n / 2, kT=0.2, superconducting=bdg
            )
            correction = None
            if bdg:
                pairing = sparse.diags(
                    [0.1 * np.ones(n - 1), -0.1 * np.ones(n - 1)], [-1, 1], format="csr"
                )
                correction = assemble_bdg_tb(
                    {(0,): sparse.csr_matrix((n, n))}, {(0,): pairing}, ndof=n
                )
            matrix = to_dense(model.hamiltonian_from_meanfield(correction)[(0,)])
            energies, vectors = np.linalg.eigh(matrix)
            exact = (vectors * expit(-energies / 0.2)) @ vectors.conj().T
            for backend in (mf.DirectDiagonalization(), mf.RationalFOE()):
                times = []
                for repeat in range(4):
                    start = time.perf_counter()
                    result = mf.density_matrix(
                        model,
                        mean_field=correction,
                        integration=mf.UniformGrid(nk=1, matrix_function=backend),
                        tol=1e-6,
                        compute_free_energy=False,
                    )
                    times.append(time.perf_counter() - start)
                error = np.max(
                    abs(
                        result.values
                        - result.coordinates.values_from_assembled_matrix(exact)
                    )
                )
                assert error < 1e-7, error
                record = dict(
                    physical_orbitals=n,
                    matrix_size=len(matrix),
                    superconducting=bdg,
                    backend=type(backend).__name__,
                    cold_seconds=times[0],
                    median_seconds=statistics.median(times[1:]),
                    charge_evaluations=result.statistics.charge_evaluations,
                    matrix_evaluations=result.statistics.n_kernel_evals,
                    density_error=float(error),
                    filling_error=abs(result.filling - n / 2),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
