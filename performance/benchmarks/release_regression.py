"""Bounded sequential release regressions with independent shifted-grid references.

Run with one BLAS/OpenMP thread. --checkout allows an older checkout to be timed
with its PeriodicQuadrature implementation; that compatibility is study-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--checkout", type=Path, default=Path(__file__).resolve().parents[2]
)
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--references", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.checkout.resolve()))

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from threadpoolctl import threadpool_limits, threadpool_info  # noqa: E402

import meanfi  # noqa: E402
from meanfi.density.density import evaluate_density_matrix_fixed_filling  # noqa: E402
from meanfi.density.integrate.bdg import solve_bdg_density_fixed_filling  # noqa: E402
from meanfi.tb.bdg import assemble_bdg_tb  # noqa: E402


def problem(name):
    """Small connected supercells; the wire has no transport band folding."""
    shape = {
        "square_metal": (4, 4),
        "gapped": (4, 4),
        "cold_bdg": (8,),
        "wire": (4, 4),
        "bounded_3d": (2, 2, 2),
    }[name]
    size = int(np.prod(shape))
    dimension = 1 if name == "wire" else len(shape)
    zero = (0,) * dimension
    rng = np.random.default_rng(90210)
    onsite = np.diag(rng.uniform(-0.2, 0.2, size)).astype(complex)
    h = {zero: onsite}
    pair = {}
    for pos in np.ndindex(shape):
        row = np.ravel_multi_index(pos, shape)
        if name == "gapped":
            onsite[row, row] = 1.5 * (-1) ** sum(pos)
        if name == "cold_bdg":
            onsite[row, row] = 0
        for axis in range(len(shape)):
            if name == "wire" and pos[axis] == shape[axis] - 1:
                continue
            neighbor = list(pos)
            neighbor[axis] = (neighbor[axis] + 1) % shape[axis]
            col = np.ravel_multi_index(tuple(neighbor), shape)
            key = (
                zero
                if name == "wire"
                else tuple(
                    int(d == axis and pos[axis] == shape[axis] - 1)
                    for d in range(dimension)
                )
            )
            opposite = tuple(-v for v in key)
            hopping = 0.3 if name == "wire" else 1 - 0.2 * axis
            for R, i, j in ((key, row, col), (opposite, col, row)):
                h.setdefault(R, np.zeros((size, size), complex))[i, j] -= hopping
            if name == "cold_bdg":
                pair.setdefault(key, np.zeros((size, size), complex))[row, col] += 0.2
                pair.setdefault(opposite, np.zeros((size, size), complex))[
                    col, row
                ] -= 0.2
    if name == "wire":
        hopping = -np.diag(rng.uniform(0.8, 1.2, size)).astype(complex)
        h[(1,)], h[(-1,)] = hopping, hopping.copy()
    temperature = {
        "square_metal": 0.2,
        "gapped": 0.05,
        "cold_bdg": 0.01,
        "wire": 0.05,
        "bounded_3d": 0.2,
    }[name]
    filling = size * (0.5 if name == "gapped" else 0.43)
    interaction = {R: np.asarray(matrix != 0, complex) for R, matrix in h.items()}
    interaction[zero] += np.eye(size)
    model = meanfi.Model(
        h,
        interaction,
        filling=filling,
        kT=temperature,
        superconducting=name == "cold_bdg",
    )
    mf = (
        assemble_bdg_tb({zero: np.zeros((size, size), complex)}, pair, ndof=size)
        if pair
        else None
    )
    return model, mf


def evaluate(model, mf):
    settings = dict(
        density_matrix_tol=1e-4,
        charge_tol=2.5e-5,
        max_points=262144,
        batch_size=128,
        matrix_function=meanfi.DirectDiagonalization(),
    )
    if hasattr(meanfi, "PeriodicGrid"):
        integration = meanfi.PeriodicGrid(**settings)
    else:
        integration = meanfi.PeriodicQuadrature(**settings)
    coordinates = model.scf_space.required_coordinates
    common = dict(
        keys=list(coordinates.keys),
        integration=integration,
        density_coordinates=coordinates,
        mu_tol=1e-12,
        max_charge_evaluations=200,
        mu_guess=0.0,
    )
    if mf is not None:
        return solve_bdg_density_fixed_filling(model, mf, filling_tol=2.5e-5, **common)
    return evaluate_density_matrix_fixed_filling(
        model.h_0,
        filling=model.filling,
        kT=model.kT,
        tolerances=meanfi.ErrorTolerances(1e-3, 1e-4, 2.5e-5, 2.5e-5),
        **common,
    )[1]


def reference(model, mf, result, order, shift):
    """Independent Fourier assembly and full covariance, at the returned mu."""
    h = model.h_0 if mf is None else model.bdg_hamiltonian_from_meanfield(mf)
    dimension = len(next(iter(h)))
    size = next(iter(h.values())).shape[0]
    keys = np.asarray(list(h))
    matrices = np.asarray(list(h.values()))
    coordinates = model.scf_space.required_coordinates
    values = np.zeros(coordinates.value_count, complex)
    charge = 0.0
    count = order**dimension
    for start in range(0, count, 128):
        indices = np.array(
            np.unravel_index(
                np.arange(start, min(start + 128, count)), (order,) * dimension
            )
        ).T
        points = 2 * np.pi * (indices + shift) / order - np.pi
        matrices_k = np.einsum("pk,kij->pij", np.exp(-1j * points @ keys.T), matrices)
        q = (
            np.ones(size)
            if mf is None
            else np.r_[np.ones(size // 2), -np.ones(size // 2)]
        )
        matrices_k[:, np.arange(size), np.arange(size)] -= result.mu * q
        eigenvalues, vectors = np.linalg.eigh(matrices_k)
        covariance = (
            vectors * expit(-eigenvalues / model.kT)[:, None, :]
        ) @ vectors.conj().transpose(0, 2, 1)
        charge += np.trace(
            covariance[:, : model._ndof, : model._ndof], axis1=1, axis2=2
        ).real.sum()
        for R, rows, cols, section in coordinates.iter_key_coordinates():
            values[section] += np.sum(
                covariance[:, rows, cols] * np.exp(1j * points @ R)[:, None], axis=0
            )
    return values / count, charge / count


records = []
with threadpool_limits(1):
    for name in ("square_metal", "gapped", "cold_bdg", "wire", "bounded_3d"):
        model, mf = problem(name)
        evaluate(model, mf)
        timings = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result = evaluate(model, mf)
            timings.append(time.perf_counter() - started)
        record = dict(
            case=name,
            temperature=model.kT,
            matrix_size=model._ndof * (2 if mf is not None else 1),
            seconds=statistics.median(timings),
            timings=timings,
            mu=result.mu,
            filling=result.filling,
            statistics=asdict(result.statistics),
            errors=asdict(result.errors),
        )
        if args.references:
            order = {
                "square_metal": 67,
                "gapped": 33,
                "cold_bdg": 513,
                "wire": 1025,
                "bounded_3d": 41,
            }[name]
            coarse, charge_coarse = reference(model, mf, result, order, 0.371)
            fine, charge_fine = reference(model, mf, result, order + 12, 0.619)
            record.update(
                density_error=float(np.max(abs(result.density.values - fine))),
                charge_error=abs(charge_fine - model.filling),
                reference_density_change=float(np.max(abs(coarse - fine))),
                reference_charge_change=abs(charge_coarse - charge_fine),
            )
        records.append(record)
        print(f"{name}: {record['seconds']:.3f} s", flush=True)
    output = dict(
        checkout=str(args.checkout.resolve()),
        python=sys.version,
        numpy=np.__version__,
        threadpools=threadpool_info(),
        results=records,
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=2) + "\n")
