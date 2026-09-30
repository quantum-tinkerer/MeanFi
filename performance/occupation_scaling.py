"""One worker per historical/current build; fixed affinity and batched timings.

Set PYTHONPATH to the desired FermiSimplex build. Historical labels select the
API present at commit 98c5009; the current build has only one algorithm.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from fermisimplex import SpectralMesh


def run(args):
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    spec = importlib.util.spec_from_file_location("models", args.models)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    tb, mu, reference, _ = models.multiband(args.bands)
    multiplicity = 1
    if args.family == "replicas":
        pair, mu, reference, _ = models.multiband(2)
        multiplicity = args.bands // 2
        rng = np.random.default_rng(42)
        basis, _ = np.linalg.qr(
            rng.normal(size=(args.bands, args.bands))
            + 1j * rng.normal(size=(args.bands, args.bands))
        )
        tb = {
            k: basis @ np.kron(np.eye(multiplicity), a) @ basis.conj().T
            for k, a in pair.items()
        }
        reference *= multiplicity
    model = tb
    if args.representation == "callable":

        def model(x):
            return sum(a * np.exp(-2j * np.pi * k[0] * x) for k, a in tb.items())

    elif args.representation == "callable_dense":
        # Same physical Hamiltonian, assembled in another orbital basis and
        # transformed on every call. This measures an expensive O(N^3) oracle.
        rng = np.random.default_rng(149)
        basis, _ = np.linalg.qr(
            rng.normal(size=(args.bands, args.bands))
            + 1j * rng.normal(size=(args.bands, args.bands))
        )
        inner = {k: basis.conj().T @ a @ basis for k, a in tb.items()}

        def model(x):
            matrix = sum(a * np.exp(-2j * np.pi * k[0] * x) for k, a in inner.items())
            return basis @ matrix @ basis.conj().T

    options = (
        {}
        if args.variant == "current"
        else {"method": "legacy" if args.variant == "legacy" else "quadratic"}
    )

    def once():
        mesh = SpectralMesh(model, root_level=2)
        result = mesh.integrate_charge(
            mu=mu,
            target_error=1e-5 * multiplicity,
            error_depth=2,
            max_refinements=3000,
            **options,
        )
        return mesh, result

    mesh, result = once()  # warm libraries; also establish a bounded batch size
    start = perf_counter()
    once()
    elapsed = perf_counter() - start
    batch = max(1, min(1000, int(args.batch_seconds / max(elapsed, 1e-6))))
    samples = []
    for _ in range(args.repeats):
        start = perf_counter()
        for _ in range(batch):
            once()
        samples.append((perf_counter() - start) / batch)
    stats = result.error_stats
    return dict(
        variant=args.variant,
        bands=args.bands,
        representation=args.representation,
        family=args.family,
        target=1e-5 * multiplicity,
        cpu=cpu,
        batch=batch,
        repeats=args.repeats,
        timing_samples=samples,
        seconds=median(samples),
        actual_error=abs(result.value - reference),
        estimated_error=result.stopping_error,
        vertices=mesh.active_vertices,
        refinements=result.stats.refinements,
        simplex_visits=result.stats.simplex_visits,
        hamiltonians=result.stats.evaluations + stats.hamiltonian_evaluations,
        eigensystems=result.stats.evaluations
        + stats.full_eigensystems
        + stats.reduced_eigensystems
        + stats.norm_eigensystems,
        initial_active_dimension_sum=stats.initial_active_dimension_sum,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("legacy", "previous", "current"), required=True
    )
    parser.add_argument("--bands", type=int, required=True)
    parser.add_argument(
        "--family", choices=("spectators", "replicas"), default="spectators"
    )
    parser.add_argument(
        "--representation", choices=("tb", "callable", "callable_dense"), required=True
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-seconds", type=float, default=0.04)
    print(json.dumps(run(parser.parse_args())))
