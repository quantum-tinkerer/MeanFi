"""Compare final-only entropy with an archived checkout, using single-thread SCF.

Run old and new checkouts sequentially. Each case discards one warmup, reports
median timings, and compares the final state with direct diagonalization outside
the timed region. No benchmark output is included in package distributions.
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
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    import scipy
    from scipy import sparse
    from scipy.special import entr, expit
    from threadpoolctl import threadpool_limits
    import meanfi as mf
    from meanfi.scf.problem import SCFProblem
    import meanfi.density.kpoint.matrix_functions.rational.prepared_sparse as rational

    original_fit, original_evaluate = (
        rational.fit_entropy,
        SCFProblem.evaluate_mean_field,
    )
    entropy_fits, final_seconds, last_input = 0, 0.0, None

    def fit(*args, **kwargs):
        nonlocal entropy_fits
        entropy_fits += 1
        return original_fit(*args, **kwargs)

    def evaluate(self, mean_field, *args, **kwargs):
        nonlocal final_seconds, last_input
        last_input = mean_field
        start = time.perf_counter()
        result = original_evaluate(self, mean_field, *args, **kwargs)
        if kwargs.get("compute_entropy"):
            final_seconds += time.perf_counter() - start
        return result

    rational.fit_entropy, SCFProblem.evaluate_mean_field = fit, evaluate
    optional = "compute_free_energy" in inspect.signature(mf.solver).parameters
    cases = []
    with threadpool_limits(limits=1):
        for size in (100, 200):
            h = sparse.diags(
                [-np.ones(size - 1), np.linspace(-0.4, 0.4, size), -np.ones(size - 1)],
                [-1, 0, 1],
                format="csr",
                dtype=complex,
            )
            interaction = sparse.diags(
                [np.full(size - 1, 0.12)] * 2, [-1, 1], format="csr"
            )
            for use_sparse in (False, True):
                model = mf.Model(
                    {(): h if use_sparse else h.toarray()},
                    {(): interaction if use_sparse else interaction.toarray()},
                    filling=0.43 * size,
                    kT=0.2,
                )
                for compute in (True, False) if optional else (True,):
                    times, final_times, fit_counts = [], [], []
                    for _ in range(args.repeat + 1):
                        entropy_fits, final_seconds = 0, 0.0
                        start = time.perf_counter()
                        result = mf.solver(
                            model,
                            {(): np.zeros((size, size))},
                            integration=mf.UniformGrid(
                                **({} if optional else {"nk": 1})
                            ),
                            scf=mf.LinearMixing(alpha=0.7, max_iterations=50),
                            tol=1e-6,
                            **({"compute_free_energy": compute} if optional else {}),
                        )
                        times.append(time.perf_counter() - start)
                        final_times.append(final_seconds)
                        fit_counts.append(entropy_fits)
                    matrix = model.hamiltonian_from_meanfield(last_input)[()]
                    matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
                    energies, vectors = np.linalg.eigh(matrix)
                    p = expit((result.mu - energies) / model.kT)
                    exact = (vectors * p) @ vectors.conj().T
                    error = np.max(
                        abs(
                            result.density.values
                            - result.density.coordinates.values_from_assembled_matrix(
                                exact
                            )
                        )
                    )
                    assert error < 1e-7, error
                    cases.append(
                        dict(
                            size=size,
                            sparse=use_sparse,
                            compute_free_energy=compute,
                            median_seconds=statistics.median(times[1:]),
                            samples=times[1:],
                            final_seconds=statistics.median(final_times[1:]),
                            entropy_fits=fit_counts[1:],
                            iterations=len(result.history),
                            internal_energy=result.internal_energy,
                            max_density_error=float(error),
                            entropy=result.entropy,
                            entropy_error=None
                            if result.entropy is None
                            else float(
                                abs(result.entropy - np.mean(entr(p) + entr(1 - p)))
                            ),
                            reported_entropy_error=getattr(
                                result.errors,
                                "entropy",
                                getattr(result.errors, "entropy_approximation", None),
                            ),
                        )
                    )
    source_hash = hashlib.sha256()
    for path in sorted((args.checkout / "meanfi").rglob("*.py")):
        if "tests" not in path.parts:
            source_hash.update(str(path.relative_to(args.checkout)).encode())
            source_hash.update(path.read_bytes())
    args.output.write_text(
        json.dumps(
            dict(
                label=args.label,
                source_sha256=source_hash.hexdigest(),
                checkout=str(args.checkout.resolve()),
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                platform=platform.platform(),
                threads=1,
                repeats=args.repeat,
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
