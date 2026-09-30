"""Measure the charge algorithm on the paper's frozen 36-orbital input.

No SCF iterations. The 1D reference is analytic; 2D uses the paper's saved
axis-exchanged quadrature reference and preserves its error diagnostics.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
from statistics import median
from time import perf_counter

import numpy as np
from fermisimplex import SpectralMesh


def timeout(*_):
    raise TimeoutError("30 second limit")


def load_paper_model(path, dimension):
    os.environ["BENCH_DIM"] = str(dimension)
    os.environ["BENCH_SITES"] = "36"
    spec = importlib.util.spec_from_file_location("paper_model", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def compare(paper, repeats):
    rows = []
    for dimension in (1, 2):
        model = load_paper_model(paper / "results/single_step/model.py", dimension)
        a, n = model.inputs()
        reference, checks = model.exact_density(a, n)
        mu = checks["mu"]
        h = {model.ZERO: (a + np.diag(model.U * n)).astype(complex)}
        h.update(
            {
                key: -t * np.eye(model.SITES, dtype=complex)
                for key, t in model.neighbors()
            }
        )
        for target in (1e100, 1e-2, 1e-3):
            row = dict(
                dimension=dimension,
                method="occupation",
                target=target,
                reference_checks=checks,
            )
            times = []
            try:
                for _ in range(repeats):
                    signal.alarm(30)
                    mesh = SpectralMesh(h, root_level=2)
                    start = perf_counter()
                    result = mesh.integrate_charge(
                        mu=mu,
                        target_error=target,
                        error_depth=2,
                        max_refinements=2000,
                    )
                    times.append(perf_counter() - start)
                    signal.alarm(0)
                error = float(abs(result.value - np.trace(reference)))
                row.update(
                    seconds=median(times),
                    timing_samples=times,
                    actual_error=error,
                    estimated_error=result.stopping_error,
                    covered=error <= result.stopping_error + 1e-10,
                    refinements=result.stats.refinements,
                    vertices=mesh.active_vertices,
                    hamiltonian_evaluations=result.stats.evaluations
                    + result.error_stats.hamiltonian_evaluations,
                    eigensystems=result.stats.evaluations
                    + result.error_stats.full_eigensystems
                    + result.error_stats.reduced_eigensystems
                    + result.error_stats.norm_eigensystems,
                )
            except (RuntimeError, TimeoutError) as error:
                row["failure"] = str(error)
            finally:
                signal.alarm(0)
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, timeout)
    args.output.write_text(
        json.dumps(compare(args.paper, args.repeats), indent=2) + "\n"
    )
