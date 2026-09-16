"""Compare density-controlled AAA fits with the pre-cleanup implementation.

Use --checkout for an archived or checked-out revision. Each repetition starts
with a fresh node and fit cache. Timings include scalar validation, and separate
initial fitting from subsequent charge/density evaluation. Dense references are
computed outside the timed region. Run comparisons sequentially.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    import scipy
    from scipy import sparse
    from scipy.special import entr
    from threadpoolctl import threadpool_limits
    import meanfi as mf
    from meanfi.density.kpoint.matrix_functions.rational import (
        PreparedMumpsRationalNode,
    )
    from meanfi.density.kpoint.matrix_functions.rational.common import (
        SparseRationalLayout,
    )

    # This harness also runs archived revisions with the former parameter name.
    matrix_target = (
        "matrix_function_tol"
        if "matrix_function_tol"
        in inspect.signature(PreparedMumpsRationalNode).parameters
        else "density_tolerance"
    )
    cases = []
    with threadpool_limits(1):
        for size in (100, 200):
            h = sparse.diags(
                [-np.ones(size - 1), np.linspace(-0.3, 0.3, size), -np.ones(size - 1)],
                [-1, 0, 1],
                format="csr",
                dtype=complex,
            )
            coords = mf.DensityCoordinates.from_entries(
                size=size,
                keys=[()],
                entries=tuple(((), i, i) for i in range(size)) + (((), 0, size - 1),),
            )
            eigenvalues, vectors = np.linalg.eigh(h.toarray())
            occupations = mf.fermi_dirac(eigenvalues, 0.1, 0.05)
            exact = (vectors * occupations) @ vectors.conj().T
            reference_entropy = (
                float(np.sum(entr(occupations) + entr(1 - occupations))) / size
            )
            for thermal in (False, True):
                fit_times, evaluation_times = [], []
                for _ in range(args.repeat + 1):
                    node = PreparedMumpsRationalNode(
                        h,
                        kT=0.1,
                        q_diag=np.ones(size),
                        options=mf.RationalFOE(),
                        charge_tolerance=1e-6,
                        **{matrix_target: 1e-7},
                        compute_thermodynamics=thermal,
                        layout=SparseRationalLayout.build(
                            density_coordinates=coords,
                            trace_weights_diag=np.ones(size),
                            include_all_diagonal=True,
                        ),
                    )
                    start = time.perf_counter()
                    terms = node._sparse_terms(0.05)
                    fit_times.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    charge = node.charge(0.05)
                    values = node.density_values_from_charge_order(0.05)
                    if thermal:
                        energy, entropy = node.thermodynamics(0.05)
                    evaluation_times.append(time.perf_counter() - start)
                density_error = float(
                    np.max(abs(values - coords.values_from_assembled_matrix(exact)))
                )
                charge_error = float(abs(charge - occupations.sum()))
                assert density_error <= 1e-7 and charge_error <= 1e-6
                cases.append(
                    dict(
                        size=size,
                        thermodynamics=thermal,
                        fit_seconds=statistics.median(fit_times[1:]),
                        evaluation_seconds=statistics.median(evaluation_times[1:]),
                        fit_samples=fit_times[1:],
                        evaluation_samples=evaluation_times[1:],
                        poles=len(terms.shifts),
                        density_error=density_error,
                        charge_error=charge_error,
                        energy_error=None
                        if not thermal
                        else float(abs(energy - eigenvalues @ occupations)) / size,
                        entropy_error=None
                        if not thermal
                        else abs(entropy / size - reference_entropy),
                        reported_entropy_error=getattr(terms, "entropy_error", None),
                    )
                )
    digest = hashlib.sha256()
    for path in sorted((args.checkout / "meanfi").rglob("*.py")):
        if "tests" not in path.parts:
            digest.update(str(path.relative_to(args.checkout)).encode())
            digest.update(path.read_bytes())
    report = dict(
        label=args.label,
        runtime_sha256=digest.hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        logical_cpus=1,
        blas_threads=1,
        repeats=args.repeat,
        density_target=1e-7,
        charge_target=1e-6,
        cases=cases,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
