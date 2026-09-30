"""Compare separate occupation builds on identical problems and tolerances.

Set PYTHONPATH to the chosen build. There is no production algorithm selector.
References are analytic occupied lengths/volumes, with 1e-12 slack for roundoff.
The timings include mesh construction and exclude one warmup per case.
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


def coupled_model(size):
    # Scaled independent pairs have the same analytic occupied fraction.
    # A dense complex basis makes the matrix arithmetic representative;
    # distinct scales remove the center's accidental degeneracies.
    scales = np.diag(np.linspace(1, 1.37, size // 2))
    rng = np.random.default_rng(149)
    basis, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    onsite = basis @ np.kron(scales, np.diag([-0.2, 0.8])) @ basis.conj().T
    slope = (
        basis @ np.kron(scales, np.array([[1.0, 0.2], [0.2, -1.0]])) @ basis.conj().T
    )

    def model(x):
        return onsite + x * slope

    reference = size / 2 * (1 - np.sqrt(1 - 4 * 1.04 * 0.16) / 1.04)
    return model, reference


def cases(models):
    for name, (model, mu, reference, dimension) in models.models():
        for target in (1e100, 1e-5 if dimension == 1 else 1e-4):
            yield name, model, mu, reference, dimension, target, 3000
    model, mu, reference, dimension = models.multiband(192)
    yield "mixed_192", model, mu, reference, dimension, 1e-5, 3000

    def sphere(x, y, z):
        return np.array([[(x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - 0.29**2]])

    def shell(x, y, z):
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
        return np.array([[(r2 - 0.16**2) * (r2 - 0.29**2)]])

    for name, model, reference in (
        ("sphere", sphere, 4 * np.pi * 0.29**3 / 3),
        ("shell", shell, 4 * np.pi * (0.29**3 - 0.16**3) / 3),
    ):
        for target in (1e100, 1e-2, 1e-3):
            yield name, model, 0, reference, 3, target, 600

    for size in (2, 12, 36, 96):
        model, reference = coupled_model(size)
        for target in (1e100, 1e-5 * size / 2):
            yield f"coupled_{size}", model, 0, reference, 1, target, 3000


def run(models, repeats, case_index=None, root_level=2):
    rows = []
    for index, (name, model, mu, reference, dimension, target, cap) in enumerate(
        cases(models)
    ):
        if case_index is not None and index != case_index:
            continue
        row = dict(
            model=name,
            dimension=dimension,
            target=target,
            reference=reference,
            refinement_cap=cap,
            root_level=root_level,
        )
        times = []
        try:
            batch = 1
            for repetition in range(repeats + 1):
                signal.alarm(30)
                start = perf_counter()
                for _ in range(batch):
                    mesh = SpectralMesh(model, root_level=root_level)
                    result = mesh.integrate_charge(
                        mu=mu, target_error=target, error_depth=2, max_refinements=cap
                    )
                elapsed = (perf_counter() - start) / batch
                times.append(elapsed)
                if repetition == 0:
                    batch = max(1, min(1000, int(0.04 / max(elapsed, 1e-6))))
                signal.alarm(0)
            actual = abs(result.value - reference)
            stats = result.error_stats
            row.update(
                seconds=median(times[1:]),
                batch=batch,
                timing_samples=times[1:],
                actual_error=actual,
                estimated_error=result.stopping_error,
                covered=bool(actual <= result.stopping_error + 1e-12),
                target_reached=result.stats.target_reached,
                vertices=mesh.active_vertices,
                refinements=result.stats.refinements,
                hamiltonians=result.stats.evaluations + stats.hamiltonian_evaluations,
                eigensystems=result.stats.evaluations
                + stats.reduced_eigensystems
                + stats.norm_eigensystems,
                center_eigensystems=stats.reduced_eigensystems,
                micro_simplices=stats.micro_simplices,
                terminal_simplices=stats.terminal_simplices,
                initial_active_dimension_sum=stats.initial_active_dimension_sum,
            )
        except (RuntimeError, TimeoutError) as error:
            signal.alarm(0)
            row["failure"] = str(error)
        rows.append(row)
    return rows


def hidden_quartic_pockets():
    rows = []
    for dimension in (3, 4):
        mesh = SpectralMesh({(0,) * dimension: np.array([[1.0]])}, root_level=0)
        vertices = mesh.points[mesh.simplices[0]]
        inverse = np.linalg.inv(np.vstack([vertices.T, np.ones(dimension + 1)]))

        def three(x, y, z):
            a, b, c, d = inverse @ [x, y, z, 1]
            return np.array([[0.001 - a * b * c * d]])

        def four(x, y, z, w):
            a, b, c, d, e = inverse @ [x, y, z, w, 1]
            return np.array([[0.001 + a * b * c * (d - e)]])

        model = {3: three, 4: four}[dimension]
        weights = (
            [0.25, 0.25, 0.25, 0.25] if dimension == 3 else [0.25, 0.25, 0.25, 0, 0.25]
        )
        witness = np.array(weights) @ vertices
        negative = float(model(*witness)[0, 0])
        assert negative < 0 < model(*vertices[0])[0, 0]
        enclosure = SpectralMesh(model, root_level=0).occupation_enclosures(mu=0)[0]
        rows.append(
            dict(
                dimension=dimension,
                false_gap=enclosure.fixed_occupation,
                negative_witness=negative,
                remainder=enclosure.interpolation_error,
            )
        )
    return rows


def alarm(*_):
    raise TimeoutError("30 second case limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--root-level", type=int, default=2)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    signal.signal(signal.SIGALRM, alarm)
    spec = importlib.util.spec_from_file_location("models", args.models)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    result = dict(
        cpu=cpu,
        charge=run(models, args.repeats, args.case_index, args.root_level),
        hidden_quartic_pockets=hidden_quartic_pockets(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
