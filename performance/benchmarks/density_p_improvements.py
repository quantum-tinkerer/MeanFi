"""Accuracy stress tests and thread scaling for the improved density p-integrator.

--backend selects an isolated directory containing a fermisimplex package,
including its native binary. This bypasses the editable-install import hook so
before/after experiments cannot accidentally load the same binary.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


def load_backend(directory):
    if directory is not None:
        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if not type(finder).__module__.lower().startswith("_fermisimplex_editable")
        ]
        sys.path.insert(0, str(directory.resolve()))
    import fermisimplex

    native = importlib.import_module("fermisimplex._native")
    return fermisimplex.SpectralMesh, str(native.__file__)


def accuracy(mesh_type):
    from performance.benchmarks.density_p_cubature import model, uniform

    rows = []
    with threadpool_limits(limits=1):
        for name in ["bulk_2d", "qwz_2d", "bulk_3d", "metal_2d"]:
            for seed in range(3):
                tb, keys, mu, reference = model(name)
                d = len(keys[0])
                rng = np.random.default_rng(seed)
                rotation, _ = np.linalg.qr(
                    rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                )
                shift = rng.uniform(0, 1, d)
                tb = {
                    key: np.exp(-2j * np.pi * np.dot(key, shift))
                    * (rotation @ value @ rotation.conj().T)
                    for key, value in tb.items()
                }
                delta = 0.0
                if reference is None:
                    nk = 128 if d < 3 else 64
                    lower = uniform(tb, keys, mu, nk)
                    reference = uniform(tb, keys, mu, 2 * nk)
                    delta = float(np.max(np.abs(lower - reference)))
                    if delta > 1e-7:
                        raise RuntimeError("Reference failed grid-doubling check")
                else:
                    reference = np.array(
                        [
                            np.exp(-2j * np.pi * np.dot(key, shift))
                            * (rotation @ value @ rotation.conj().T)
                            for key, value in zip(keys, reference, strict=True)
                        ]
                    )
                for tol in [1e-3, 1e-4, 1e-5]:
                    mesh = mesh_type(tb)
                    charge = mesh.integrate_charge(mu=mu, target_error=tol)
                    result = mesh.integrate_density_components_p(
                        mu=mu,
                        lattice_vectors=keys,
                        components=[
                            [k, i, j]
                            for k in range(len(keys))
                            for i in range(2)
                            for j in range(2)
                        ],
                        target_error=tol,
                    )
                    actual = float(
                        np.max(
                            np.abs(result.values.reshape(len(keys), 2, 2) - reference)
                        )
                    )
                    row = dict(
                        model=name,
                        seed=seed,
                        tolerance=tol,
                        error=actual,
                        estimate=result.stopping_error,
                        ref_delta=delta,
                        converged=result.stats.target_reached,
                        charge_converged=charge.stats.target_reached,
                        evaluations=result.stats.evaluations,
                    )
                    rows.append(row)
                    print(row, flush=True)
                    if result.stats.target_reached and actual > tol + delta:
                        raise RuntimeError(
                            f"Accepted result misses accuracy target: {row}"
                        )
    return rows


def threads(mesh_type, repeat):
    from performance.benchmarks.density_p_cubature import model, uniform

    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for name, copies in [
            ("bulk_2d", 1),
            ("bulk_3d", 1),
            ("metal_2d", 1),
            ("bulk_2d", 16),
            ("bulk_2d", 32),
        ]:
            tb, keys, mu, reference = model(name)
            if reference is None:
                reference = uniform(tb, keys, mu, 64)
            tb = {key: np.kron(np.eye(copies), value) for key, value in tb.items()}
            components = [
                [k, i, j]
                for k in range(len(keys))
                for b in range(copies)
                for i in range(2 * b, 2 * b + 2)
                for j in range(2 * b, 2 * b + 2)
            ]
            mesh = mesh_type(tb)
            with threadpool_limits(limits=1, user_api="openmp"):
                mesh.integrate_charge(mu=mu, target_error=1e-5)
            for count in [1, 4]:
                times = []
                with threadpool_limits(limits=count, user_api="openmp"):
                    pools = threadpool_info()
                    for trial in range(repeat + 1):
                        start = perf_counter()
                        result = mesh.integrate_density_components_p(
                            mu=mu,
                            lattice_vectors=keys,
                            components=components,
                            target_error=1e-5,
                        )
                        elapsed = perf_counter() - start
                        if trial:
                            times.append(elapsed)
                actual = float(
                    np.max(
                        np.abs(
                            result.values.reshape(len(keys), copies, 2, 2)
                            - reference[:, None]
                        )
                    )
                )
                row = dict(
                    model=name,
                    ndof=2 * copies,
                    threads=count,
                    seconds=float(np.median(times)),
                    times=times,
                    threadpools=pools,
                    actual_error=actual,
                    estimated_error=result.stopping_error,
                    evaluations=result.stats.evaluations,
                    p_refinements=result.stats.p_refinements,
                    converged=result.stats.target_reached,
                )
                rows.append(row)
                print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["accuracy", "threads"])
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    mesh_type, binary = load_backend(args.backend)
    rows = (
        accuracy(mesh_type)
        if args.mode == "accuracy"
        else threads(mesh_type, args.repeat)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(binary=binary, records=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
