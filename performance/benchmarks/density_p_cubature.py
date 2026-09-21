"""Compare density p-cubature and legacy h-refinement on identical charge meshes.

Run with the modified FermiSimplex on PYTHONPATH. All timings use one BLAS
thread. References are analytic where possible and checked by grid doubling
otherwise. No fitted exponent is inferred from unconverged results.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from fermisimplex import SpectralMesh
from threadpoolctl import threadpool_limits

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1, -1]).astype(complex)


def model(name):
    if name == "rotating_1d":
        a = (SZ + 1j * SX) / 2
        tb = {(1,): a, (-1,): a.conj().T}
        reference = np.stack([np.eye(2) / 2, -a / 2])
    elif name == "metal_2d":
        a = (SZ + 1j * SX) / 2
        tb = {
            (1, 0): 0.4 * np.eye(2),
            (-1, 0): 0.4 * np.eye(2),
            (0, 1): a,
            (0, -1): a.conj().T,
        }
        theta = np.arccos(-0.25)
        filling = 1 - theta / np.pi
        reference = np.stack(
            [
                filling * np.eye(2) / 2,
                -np.sin(theta) / np.pi * np.eye(2) / 2,
                -filling * a / 2,
            ]
        )
    else:
        d = 3 if name == "bulk_3d" else 2
        mass = 1.3 if name == "qwz_2d" else d + 1.0
        tb = {(0,) * d: mass * SZ}
        for axis in range(d):
            key = tuple(int(i == axis) for i in range(d))
            a = (SZ + 1j * (SX if axis % 2 == 0 else SY)) / 2
            tb[key] = a
            tb[tuple(-v for v in key)] = a.conj().T
        reference = None
    d = len(next(iter(tb)))
    keys = [(0,) * d] + [tuple(int(i == j) for i in range(d)) for j in range(d)]
    mu = -1.2 if name == "metal_2d" else 0.0
    return tb, keys, mu, reference


def uniform(tb, keys, mu, nk):
    """Batched midpoint grid, with matrix size 2 and bounded memory."""
    d = len(keys[0])
    total = np.zeros((len(keys), 2, 2), complex)
    for start in range(0, nk**d, 32768):
        indices = np.arange(start, min(start + 32768, nk**d))
        points = np.stack(np.unravel_index(indices, (nk,) * d), axis=1)
        points = (points + 0.5) / nk
        h = np.zeros((len(points), 2, 2), complex)
        for key, hopping in tb.items():
            h += np.exp(-2j * np.pi * (points @ key))[:, None, None] * hopping
        energies, vectors = np.linalg.eigh(h)
        occupied = energies < mu
        rho = (vectors * occupied[:, None, :]) @ vectors.conj().transpose(0, 2, 1)
        for k, key in enumerate(keys):
            total[k] += np.einsum("n,nij->ij", np.exp(2j * np.pi * (points @ key)), rho)
    return total / nk**d


def run(name, tolerances, repeat, *, charge_tol=None, methods=None, copies=1):
    tb, keys, mu, reference = model(name)
    base_tb = tb
    d = len(keys[0])
    components = [
        [k, i, j]
        for k in range(len(keys))
        for block in range(copies)
        for i in range(2 * block, 2 * block + 2)
        for j in range(2 * block, 2 * block + 2)
    ]
    reference_delta = 0.0
    if reference is None:
        nk = 64 if d == 3 else 128
        lower = uniform(tb, keys, mu, nk)
        reference = uniform(tb, keys, mu, 2 * nk)
        reference_delta = float(np.max(np.abs(reference - lower)))
        if reference_delta > min(tolerances) / 100:
            raise RuntimeError(f"{name}: reference is not sufficiently converged")
    tb = {key: np.kron(np.eye(copies), value) for key, value in base_tb.items()}
    reference = np.tile(reference[:, None, :, :], (1, copies, 1, 1))
    rows = []
    for tol in tolerances:
        for method in methods or ["linear", "centroid", "h", "p"]:
            timings, charges = [], []
            for trial in range(repeat + 1):
                mesh = SpectralMesh(tb)
                start = perf_counter()
                charge = mesh.integrate_charge(
                    mu=mu, target_error=tol if charge_tol is None else charge_tol
                )
                charge_seconds = perf_counter() - start
                leaves = mesh.active_simplices
                start = perf_counter()
                common = dict(
                    mu=mu, lattice_vectors=keys, components=components, target_error=tol
                )
                if method in ("p", "centroid"):
                    result = mesh.integrate_density_components_p(
                        **common, max_degree=2 if method == "centroid" else 21
                    )
                else:
                    result = mesh.integrate_density_components(
                        **common,
                        preview_depth=0 if method == "linear" else 1,
                        max_refinements=300000,
                    )
                elapsed = perf_counter() - start
                if trial:
                    timings.append(elapsed)
                    charges.append(charge_seconds)
            values = result.values.reshape(reference.shape)
            error = float(np.max(np.abs(values - reference)))
            row = dict(
                model=name,
                ndof=2 * copies,
                charge_tolerance=tol if charge_tol is None else charge_tol,
                dimension=d,
                method=method,
                tolerance=tol,
                actual_error=error,
                estimated_error=float(result.stopping_error),
                converged=bool(result.stats.target_reached),
                charge_converged=bool(charge.stats.target_reached),
                charge_seconds=float(np.median(charges)),
                density_seconds=float(np.median(timings)),
                charge_evaluations=int(
                    charge.stats.evaluations
                    + charge.error_stats.hamiltonian_evaluations
                ),
                density_evaluations=int(result.stats.evaluations),
                charge_simplices=leaves,
                final_simplices=mesh.active_simplices,
                p_refinements=result.stats.p_refinements,
                max_degree=result.stats.max_degree,
                reference_delta=reference_delta,
            )
            rows.append(row)
            print(
                f"{name:12s} {tol:.0e} {method:8s} err={error:.3g} "
                f"est={result.stopping_error:.3g} evals={result.stats.evaluations:7d} "
                f"t={row['density_seconds']:.4f}s converged={row['converged']}",
                flush=True,
            )
    # Uniform work/accuracy ladder gives a baseline independent of estimators.
    for nk in [8, 16, 32, 64, 128] if d < 3 else [8, 16, 32, 64]:
        start = perf_counter()
        value = uniform(base_tb, keys, mu, nk)[:, None, :, :]
        rows.append(
            dict(
                model=name,
                ndof=2 * copies,
                charge_tolerance=tol if charge_tol is None else charge_tol,
                dimension=d,
                method="uniform",
                nk=nk,
                actual_error=float(np.max(np.abs(value - reference))),
                density_seconds=perf_counter() - start,
                density_evaluations=nk**d,
                reference_delta=reference_delta,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rotating_1d", "bulk_2d", "qwz_2d", "bulk_3d", "metal_2d"],
    )
    parser.add_argument(
        "--tolerances", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5]
    )
    parser.add_argument("--charge-tol", type=float, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=["linear", "centroid", "h", "p"]
    )
    parser.add_argument("--copies", nargs="+", type=int, default=[1])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance/results/density_p_cubature.json"),
    )
    args = parser.parse_args()
    if min(args.copies) < 1:
        parser.error("--copies must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    rows = []
    with threadpool_limits(limits=1):
        for name in args.models:
            for copies in args.copies:
                rows.extend(
                    run(
                        name,
                        args.tolerances,
                        args.repeat,
                        charge_tol=args.charge_tol,
                        methods=args.methods,
                        copies=copies,
                    )
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        platform=platform.platform(),
                        numpy=np.__version__,
                        repeat=args.repeat,
                        records=rows,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
