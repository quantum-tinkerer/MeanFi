"""Time density and SCF calculations, reporting errors against known references."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import csr_array
from scipy.special import expit
from threadpoolctl import threadpool_info, threadpool_limits

import meanfi as mf


def measure(name, calculation, reference, repeat):
    calculation()  # Warm imports and native libraries outside the timed samples.
    seconds = []
    for _ in range(repeat):
        start = perf_counter()
        result = calculation()
        seconds.append(perf_counter() - start)
    density = result.density if isinstance(result, mf.SCFResult) else result
    record = {
        "name": name,
        "seconds": seconds,
        "median_seconds": median(seconds),
        "density_max_error": float(np.max(np.abs(density.values - reference))),
        "errors": asdict(density.errors),
        "density_work": asdict(density.statistics),
    }
    if isinstance(result, mf.SCFResult):
        record["iterations"] = len(result.history)
        record["converged"] = result.converged
    return record


def density_benchmarks(size, repeat, sparse, initial_nk):
    # H(k) = A - 2 cos(k) I: its eigenvectors are k independent, so an
    # independent reference needs only one diagonalization of A.
    onsite = np.diag(np.linspace(-0.8, 0.8, size))
    onsite += np.diag(np.full(size - 1, 0.3), 1)
    onsite += np.diag(np.full(size - 1, 0.3), -1)
    hopping = -np.eye(size)
    h = {(0,): onsite, (1,): hopping, (-1,): hopping}
    rows = np.concatenate((np.arange(size), np.arange(size - 1)))
    cols = np.concatenate((np.arange(size), np.arange(1, size)))
    coordinates = mf.DensityCoordinates(size, ((0,),), (rows,), (cols,))
    energies, vectors = np.linalg.eigh(onsite)
    mu = 0.17

    def reference(occupations):
        return ((vectors * occupations) @ vectors.T)[rows, cols]

    zero_temperature = reference(np.arccos(np.clip((energies - mu) / 2, -1, 1)) / np.pi)

    def thermal_reference(nk):
        k = 2 * np.pi * np.arange(nk) / nk
        return reference(
            np.mean(expit((mu - energies[:, None] + 2 * np.cos(k)) / 0.2), axis=1)
        )

    thermal = thermal_reference(2048)
    reference_difference = float(np.max(np.abs(thermal - thermal_reference(1024))))
    # This checks the reference itself, not a claimed bound for the estimator.
    assert reference_difference < 1e-12
    cases = [
        ("simplex", h, 0.0, mf.FermiSimplex(initial_nk=initial_nk), zero_temperature),
        ("dense-grid", h, 0.2, mf.UniformGrid(nk=64), thermal),
    ]
    if sparse:
        cases.append(
            (
                "sparse-grid",
                {key: csr_array(value) for key, value in h.items()},
                0.2,
                mf.UniformGrid(nk=64, matrix_function=mf.RationalFOE()),
                thermal,
            )
        )
    records = []
    for name, hamiltonian, temperature, integration, expected in cases:
        record = measure(
            f"density/{name}/N={size}",
            lambda: mf.density_matrix_at_mu(
                hamiltonian,
                mu,
                kT=temperature,
                coordinates=coordinates,
                integration=integration,
            ),
            expected,
            repeat,
        )
        record.update(
            size=size,
            temperature=temperature,
            mu=mu,
            reference_grid_difference=0.0 if temperature == 0 else reference_difference,
        )
        records.append(record)
    return records


def scf_benchmarks(repeat):
    hopping = -np.eye(2)
    h = {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping}
    interaction = {(0,): np.array([[0.0, 4.0], [4.0, 0.0]])}
    records = []
    for temperature in (0.0, 0.2):
        model = mf.Model(h, interaction, filling=1, kT=temperature)
        guess = model.random_meanfield(rng=0, scale=0.2)
        # At half filling U=4 is below the magnetic instability of this
        # translation-invariant chain: the onsite density is exactly I/2.
        reference = model.required_coordinates.values_from_tb({(0,): np.eye(2) / 2})
        records.append(
            measure(
                f"scf/{'simplex' if temperature == 0 else 'dense-grid'}",
                lambda: mf.solver(model, guess),
                reference,
                repeat,
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--initial-nk", type=int, help="starting simplex mesh, in total points"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="also benchmark RationalFOE (requires MUMPS)",
    )
    parser.add_argument(
        "--output", type=Path, help="optional JSON report, e.g. build/benchmarks.json"
    )
    args = parser.parse_args()
    if args.repeat < 1 or min(args.sizes) < 2:
        parser.error("repeat must be positive and sizes must be at least 2")
    with threadpool_limits(limits=1):
        records = []
        for size in args.sizes:
            records.extend(
                density_benchmarks(size, args.repeat, args.sparse, args.initial_nk)
            )
        records.extend(scf_benchmarks(args.repeat))
        report = {
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "meanfi": mf.__version__,
                "platform": platform.platform(),
                "threadpools": threadpool_info(),
            },
            "settings": {
                "sizes": args.sizes,
                "repeat": args.repeat,
                "sparse": args.sparse,
                "tol": 1e-3,
                "threads": 1,
                "initial_nk": args.initial_nk,
            },
            "benchmarks": records,
        }
    print(
        f"{'Calculation':38} {'Median (s)':>12} {'Density error':>15} {'SCF steps':>10}"
    )
    for record in records:
        print(
            f"{record['name']:38} {record['median_seconds']:12.4f} "
            f"{record['density_max_error']:15.3g} {str(record.get('iterations', '-')):>10}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
