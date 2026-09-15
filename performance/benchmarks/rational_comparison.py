"""Compare AAA and Ozaki density expansions against independent eigensystems.

Use an archived checkout to compare historical AAA and Ozaki implementations::

    mkdir -p /tmp/meanfi-rational-baseline
    git archive 3443495 | tar -x -C /tmp/meanfi-rational-baseline
    .pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
        --checkout /tmp/meanfi-rational-baseline --output results.json

Without --checkout, this uses the current runtime and only its available schemes.
--thermodynamics also checks energy, entropy, and reuse of density factorizations.
One CPU and one BLAS thread are used. Cold measurements clear scalar fit caches;
warm measurements construct a new node sharing the previous scalar fit. Every
measurement still factors its matrices. Public fixed-filling calls use their
normal per-calculation caches. Timings exclude the independent dense references.
--large-matrices measures N=100, 200, and 512 with chain, rectangular-grid, and
random-graph sparsity, including chemical-potential searches. It reports energy
and entropy errors per orbital; the original node suite reports total-trace
errors for comparison with its archived results. Accuracy requires the density
and charge targets; thermodynamic errors are separate diagnostics. Use --per-cell
when benchmarking public APIs from before thermodynamic normalization.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkout", type=Path, default=Path(__file__).resolve().parents[2]
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-revision", help="Commit of an archived checkout")
parser.add_argument(
    "--thermodynamics",
    action="store_true",
    help="Also benchmark full-matrix energy/entropy traces (nodes only)",
)
parser.add_argument("--large-matrices", action="store_true")
parser.add_argument(
    "--per-cell", action="store_true", help="Archived public energy/entropy units"
)
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--cpu", type=int, default=2)
parser.add_argument("--only", help="Comma-separated case-name substrings")
parser.add_argument("--max-poles", type=int, default=128)
parser.add_argument("--scheme", choices=("aaa", "ozaki"))
parser.add_argument(
    "--certify-ozaki",
    action="store_true",
    help="Benchmark a scalar-certified Ozaki stopping rule",
)
parser.add_argument("--timeout", type=float, default=20.0)
args = parser.parse_args()
if args.repeat < 1:
    parser.error("--repeat must be positive")
os.sched_setaffinity(0, {args.cpu})
sys.path.insert(0, str(args.checkout.resolve()))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import entr, expit  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

import meanfi  # noqa: E402
from meanfi.density.kpoint.matrix_functions.mumps_backend import (  # noqa: E402
    SelectedInverseFactorization,
)
from meanfi.density.kpoint.matrix_functions.rational import (  # noqa: E402
    PreparedMumpsRationalNode,
)
from meanfi.density.kpoint.matrix_functions.rational import scheme as scalar_scheme  # noqa: E402
from meanfi.density.kpoint.matrix_functions.rational import prepared_sparse  # noqa: E402

from meanfi.density.kpoint.matrix_functions.common import (  # noqa: E402
    spectral_interval,
    shift_by_mu,
)
from meanfi.space.coordinates import DensityCoordinates  # noqa: E402


_aaa_sample_grid = scalar_scheme._aaa_sample_grid
_evaluate_canonical_rational = scalar_scheme._evaluate_canonical_rational
_ozaki_exact_poles_and_residues = getattr(
    scalar_scheme, "_ozaki_exact_poles_and_residues", None
)
_ozaki_terms = getattr(scalar_scheme, "_ozaki_terms", None)
if (args.scheme == "ozaki" or args.certify_ozaki) and _ozaki_terms is None:
    parser.error("this checkout has no Ozaki implementation; use the archived baseline")


def rational_options(scheme):
    options = dict(initial_poles=4, max_poles=args.max_poles)
    if "rational_scheme" in meanfi.RationalFOE.__dataclass_fields__:
        options["rational_scheme"] = scheme
    return meanfi.RationalFOE(**options)


def clear_scalar_cache():
    if _ozaki_exact_poles_and_residues is not None:
        _ozaki_exact_poles_and_residues.cache_clear()


def certified_ozaki_charge(self, mu):
    """Study-only replacement: certify the scalar function before factorization."""
    if self.options.rational_scheme != "ozaki":
        return original_charge(self, mu)
    if self._last_mu == float(mu):
        return self._last_charge
    interval = spectral_interval(shift_by_mu(self.matrix, mu, self.q_diag))
    grid = _aaa_sample_grid(*interval, count=8192, kT=self.kT)
    exact = expit(-grid / self.kT)
    poles = self.options.initial_poles
    while True:
        terms = _ozaki_terms(poles, self.kT)
        approximate = _evaluate_canonical_rational(
            grid, constant=terms.constant, shifts=terms.shifts, residues=terms.residues
        )
        if np.max(np.abs(approximate - exact)) <= self._charge_scalar_tolerance():
            break
        if poles == self.options.max_poles:
            raise meanfi.ConvergenceError(
                "Ozaki scalar certification exceeded max_poles"
            )
        poles = min(2 * poles, self.options.max_poles)
    (
        self._last_charge,
        self._last_terms,
        self._last_factorizations,
        self._last_charge_entries,
    ) = self._evaluate_charge_for_pole_count(mu, poles)
    self._last_mu = float(mu)
    return self._last_charge


node_parameters = inspect.signature(PreparedMumpsRationalNode).parameters
original_charge = PreparedMumpsRationalNode.charge
if args.certify_ozaki:
    PreparedMumpsRationalNode.charge = certified_ozaki_charge


@dataclass(frozen=True)
class Case:
    name: str
    size: int
    kT: float
    tolerance: float = 1e-8
    family: str = "metal"
    mu: float = 0.13
    workflow: str = "node"
    filling_fraction: float = 0.43
    nk: int = 1


def cases():
    if args.large_matrices:
        for n in (100, 200, 512):
            for family in ("metal", "rectangle", "random"):
                for temperature in (0.2, 0.02):
                    yield Case(
                        f"large_{family}_n{n}_T{temperature:g}",
                        n,
                        temperature,
                        family=family,
                    )
                yield Case(
                    f"filling_{family}_n{n}_T0.02",
                    n,
                    0.02,
                    family=family,
                    workflow="filling",
                    nk=4,
                )
        return
    # Repeat widths/temperatures at fixed size to distinguish setup from sparse
    # factorization work. Symmetric spectra deliberately challenge trace-only
    # stopping criteria. Matrix sizes refer to electron-space dimensions.
    for n in (4, 32, 128, 512):
        for temperature in (2.0, 0.2, 0.02):
            yield Case(f"metal_n{n}_T{temperature:g}", n, temperature)
    for temperature in (0.2, 0.02, 0.002, 0.0002):
        yield Case(
            f"symmetric_T{temperature:g}", 32, temperature, family="symmetric", mu=0.0
        )
    for family in ("gapped", "degenerate", "empty", "full"):
        yield Case(family, 64, 0.02, family=family, mu=0.0)
    for tolerance in (1e-4, 1e-6, 1e-10, 1e-12):
        yield Case(f"tolerance_{tolerance:g}", 32, 0.02, tolerance=tolerance)
    for n, temperature in ((16, 0.2), (64, 0.02), (128, 0.002)):
        yield Case(f"bdg_n{n}_T{temperature:g}", n, temperature, family="bdg")
    for family, n in (("square", 64), ("square", 256), ("cube", 64), ("cube", 216)):
        for temperature in (0.2, 0.02):
            yield Case(
                f"geometry_{family}_n{n}_T{temperature:g}",
                n,
                temperature,
                family=family,
                tolerance=1e-6,
            )
    for family, n, temperature, nk in (
        ("metal", 8, 0.2, 8),
        ("metal", 32, 0.02, 4),
        ("gapped", 16, 0.02, 4),
        ("bdg", 8, 0.2, 8),
        ("bdg", 16, 0.02, 4),
    ):
        yield Case(
            f"filling_{family}_n{n}_T{temperature:g}",
            n,
            temperature,
            family=family,
            workflow="filling",
            nk=nk,
        )


def electron_matrix(case):
    n = case.size
    rng = np.random.default_rng(174729 + n)
    diagonal = rng.uniform(-0.4, 0.4, n)
    if case.family == "symmetric":
        diagonal[:] = 0.0
    elif case.family == "gapped":
        diagonal = 3.0 * (-1.0) ** np.arange(n)
    elif case.family in {"degenerate", "empty", "full"}:
        energy = {"degenerate": 0.0, "empty": 8.0, "full": -8.0}[case.family]
        return sparse.eye(n, dtype=complex, format="csr") * energy
    if case.family == "random":
        rows = np.repeat(np.arange(n), 4)
        cols = rng.integers(0, n, size=rows.size)
        values = -rng.uniform(0.5, 1.0, rows.size) * np.exp(0.17j)
        graph = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        matrix = graph + graph.conj().T + sparse.diags(diagonal)
    elif case.family in {"square", "cube", "rectangle"}:
        dimension = 3 if case.family == "cube" else 2
        side = round(n ** (1 / dimension))
        if case.family == "rectangle":
            side = int(np.sqrt(n))
            while n % side:
                side -= 1
            shape = (side, n // side)
        else:
            shape = (side,) * dimension
        matrix = sparse.diags(diagonal, format="lil", dtype=complex)
        for position in np.ndindex(shape):
            row = np.ravel_multi_index(position, shape)
            for axis in range(dimension):
                if position[axis] + 1 == shape[axis]:
                    continue
                neighbor = list(position)
                neighbor[axis] += 1
                col = np.ravel_multi_index(tuple(neighbor), shape)
                value = -(1 - 0.15 * axis) * np.exp(0.17j)
                matrix[row, col] = value
                matrix[col, row] = value.conjugate()
    else:
        hopping = -(1.0 + 0.15 * rng.random(n - 1)).astype(complex)
        # Complex hopping tests conjugate-pair reconstruction.
        if case.family != "symmetric":
            hopping *= np.exp(0.17j)
        matrix = sparse.diags(
            [hopping, diagonal, hopping.conj()], [-1, 0, 1], format="csr"
        )
    matrix = matrix.tocsr()
    if args.large_matrices:
        # Match the Gershgorin radius so geometry changes matrix cost without
        # also making the scalar approximation arbitrarily harder.
        matrix *= 2.5 / np.max(np.asarray(abs(matrix).sum(axis=1)))
    return matrix


def matrix_and_charge(case):
    h = electron_matrix(case)
    if case.family != "bdg":
        return h, np.ones(case.size), np.ones(case.size)
    upper = sparse.diags(np.full(case.size - 1, 0.19), 1, shape=h.shape)
    pairing = (upper - upper.T).tocsr()
    matrix = sparse.bmat([[h, pairing], [pairing.conj().T, -h.T]], format="csr")
    return (
        matrix,
        np.r_[np.ones(case.size), -np.ones(case.size)],
        np.r_[np.ones(case.size), np.zeros(case.size)],
    )


def selected_coordinates(matrix, keys):
    support = matrix.tocoo()
    diagonal = np.arange(matrix.shape[0])
    pairs = np.unique(
        np.c_[np.r_[support.row, diagonal], np.r_[support.col, diagonal]], axis=0
    )
    return DensityCoordinates.from_pairs(
        size=matrix.shape[0],
        keys=keys,
        pairs_by_key={key: (pairs[:, 0], pairs[:, 1]) for key in keys},
    )


def dense_reference(matrix, q_diag, trace_weights, mu, kT):
    shifted = matrix.toarray() - np.diag(mu * q_diag)
    energies, vectors = np.linalg.eigh(shifted)
    density = (vectors * expit(-energies / kT)) @ vectors.conj().T
    charge = float(np.dot(trace_weights, density.diagonal().real))
    return density, charge


@contextmanager
def instrumentation():
    stats = dict(
        factorizations=0,
        selected_inverse_calls=0,
        scalar_fit_calls=0,
        scalar_build_calls=0,
        scalar_build_seconds=0.0,
        factor_seconds=0.0,
        selected_inverse_seconds=0.0,
        scalar_fit_seconds=0.0,
        accepted_poles={},
    )
    original_factor = SelectedInverseFactorization.factor
    original_inverse = SelectedInverseFactorization.selected_inverse
    original_terms = PreparedMumpsRationalNode._sparse_terms
    original_build = prepared_sparse._aaa_terms_for_interval

    def build(*positional, **keywords):
        stats["scalar_build_calls"] += 1
        start = time.perf_counter()
        try:
            return original_build(*positional, **keywords)
        finally:
            stats["scalar_build_seconds"] += time.perf_counter() - start

    def factor(self, matrix):
        stats["factorizations"] += 1
        start = time.perf_counter()
        try:
            return original_factor(self, matrix)
        finally:
            stats["factor_seconds"] += time.perf_counter() - start

    def inverse(self, pattern):
        stats["selected_inverse_calls"] += 1
        start = time.perf_counter()
        try:
            return original_inverse(self, pattern)
        finally:
            stats["selected_inverse_seconds"] += time.perf_counter() - start

    def terms(self, *positional, **keywords):
        stats["scalar_fit_calls"] += 1
        start = time.perf_counter()
        try:
            result = original_terms(self, *positional, **keywords)
            count = str(len(result.shifts))
            stats["accepted_poles"][count] = stats["accepted_poles"].get(count, 0) + 1
            return result
        finally:
            stats["scalar_fit_seconds"] += time.perf_counter() - start

    SelectedInverseFactorization.factor = factor
    SelectedInverseFactorization.selected_inverse = inverse
    PreparedMumpsRationalNode._sparse_terms = terms
    prepared_sparse._aaa_terms_for_interval = build
    try:
        yield stats
    finally:
        SelectedInverseFactorization.factor = original_factor
        SelectedInverseFactorization.selected_inverse = original_inverse
        PreparedMumpsRationalNode._sparse_terms = original_terms
        prepared_sparse._aaa_terms_for_interval = original_build


def timeout(_signum, _frame):
    raise TimeoutError(f"measurement exceeded {args.timeout:g} seconds")


signal.signal(signal.SIGALRM, timeout)


def measure(call, *, reference_charge, reference_values, tolerance):
    with instrumentation() as stats:
        start = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        try:
            charge, values, extra = call()
            error_charge = abs(charge - reference_charge)
            error_density = float(
                np.max(np.abs(values - reference_values), initial=0.0)
            )
            # Strictly compare absolute requested tolerances; retain raw errors
            # so floating-point headroom can be assessed independently.
            result = dict(
                status="ok",
                charge_error=error_charge,
                density_error=error_density,
                accurate=(
                    error_charge <= tolerance
                    and error_density <= tolerance
                    and extra.get("thermo_factorizations", 0) == 0
                    and extra.get("thermo_inverse_queries", 0) == 0
                ),
                thermodynamics_within_density_tolerance=(
                    extra.get("energy_error", 0.0) <= tolerance
                    and extra.get("entropy_error", 0.0) <= tolerance
                )
                if "entropy_error" in extra
                else None,
                **extra,
            )
        except Exception as exc:
            result = dict(
                status="failure",
                exception=type(exc).__name__,
                message=str(exc),
                accurate=False,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        result["seconds"] = (
            time.perf_counter() - start - result.get("reference_seconds", 0.0)
        )
    return dict(result, **stats)


def node_case(case, scheme):
    matrix, q_diag, trace_weights = matrix_and_charge(case)
    coords = selected_coordinates(matrix, [()])
    rho, charge = dense_reference(matrix, q_diag, trace_weights, case.mu, case.kT)
    reference_values = coords.values_from_assembled_matrix(rho)
    energies = np.linalg.eigvalsh(matrix.toarray() - np.diag(case.mu * q_diag))
    occupation = expit(-energies / case.kT)
    reference_energy = float(np.dot(energies, occupation))
    thermal_method = getattr(PreparedMumpsRationalNode, "thermodynamics", None)
    if thermal_method is not None and "Tr[H" in (thermal_method.__doc__ or ""):
        reference_energy = float(np.trace(matrix.toarray() @ rho).real)
    reference_entropy = float(
        np.sum(entr(occupation) + entr(expit(energies / case.kT)))
    )
    normalization = matrix.shape[0] if args.large_matrices else 1
    options = rational_options(scheme)
    runs = {"cold": [], "warm": []}
    interval_cache = []

    def evaluate():
        start = time.perf_counter()
        settings = dict(
            kT=case.kT,
            q_diag=q_diag,
            options=options,
            charge_tolerance=case.tolerance,
            density_tolerance=case.tolerance,
            shared_aaa_interval_cache=interval_cache,
        )
        thermodynamics = args.thermodynamics or args.large_matrices
        if "layout" in node_parameters:
            settings["layout"] = prepared_sparse.SparseRationalLayout.build(
                density_coordinates=coords,
                trace_weights_diag=trace_weights,
                include_all_diagonal=thermodynamics,
            )
        else:
            settings.update(
                density_coordinates=coords, trace_weights_diag=trace_weights
            )
        if thermodynamics:
            tolerance = normalization * case.tolerance
            if "compute_thermodynamics" in node_parameters:
                settings["compute_thermodynamics"] = True
            elif "band_energy_tolerance" in node_parameters:
                settings.update(
                    band_energy_tolerance=tolerance, entropy_tolerance=tolerance
                )
            else:
                settings["thermodynamic_tolerance"] = tolerance
        node = PreparedMumpsRationalNode(matrix, **settings)
        setup_seconds = time.perf_counter() - start
        obtained_charge = node.charge(case.mu)
        density = node.density_values_from_charge_order(case.mu)
        poles = len(node._last_terms.shifts)
        extra = dict(setup_seconds=setup_seconds, final_poles=poles)
        if args.thermodynamics or args.large_matrices:
            with instrumentation() as thermal_work:
                start = time.perf_counter()
                energy, entropy = node.thermodynamics(case.mu)
                extra["thermodynamic_seconds"] = time.perf_counter() - start
            extra.update(
                energy=energy,
                entropy=entropy,
                energy_error=abs(energy - reference_energy) / normalization,
                entropy_error=abs(entropy - reference_entropy) / normalization,
                entropy_approximation_error=(
                    None
                    if getattr(node._last_terms, "entropy_error", None) is None
                    else node._last_terms.entropy_error
                    * matrix.shape[0]
                    / normalization
                ),
                thermo_factorizations=thermal_work["factorizations"],
                thermo_inverse_queries=thermal_work["selected_inverse_calls"],
            )
        del node
        return (
            obtained_charge,
            density,
            extra,
        )

    for _ in range(args.repeat):
        if any(run["status"] == "failure" for run in runs["cold"] + runs["warm"]):
            break
        clear_scalar_cache()
        interval_cache.clear()
        gc.collect()
        for state in ("cold", "warm"):
            runs[state].append(
                measure(
                    evaluate,
                    reference_charge=charge,
                    reference_values=reference_values,
                    tolerance=case.tolerance,
                )
            )
            if runs[state][-1]["status"] == "failure":
                break
    return dict(
        case=asdict(case),
        scheme=scheme,
        matrix_size=matrix.shape[0],
        nnz=matrix.nnz,
        selected_entries=coords.value_count,
        spectral_width_over_kT=float(np.ptp(energies) / case.kT),
        spectral_radius_over_kT=float(np.max(np.abs(energies)) / case.kT),
        reference_charge=charge,
        thermodynamic_normalization=normalization,
        runs=runs,
    )


def filling_case(case, scheme):
    h0 = electron_matrix(case)
    zero = (0,)
    hopping = -0.18 * sparse.eye(case.size, dtype=complex, format="csr")
    h = {zero: h0, (1,): hopping, (-1,): hopping}
    model = meanfi.Model(
        h,
        {zero: sparse.eye(case.size, format="csr")},
        filling=case.size * case.filling_fraction,
        kT=case.kT,
        superconducting=case.family == "bdg",
    )
    correction = None
    if model.superconducting:
        bdg, q_diag, weights = matrix_and_charge(case)
        bare = sparse.block_diag([h0, -h0.T], format="csr")
        correction = {zero: bdg - bare}
    else:
        q_diag = weights = np.ones(case.size)
    total_h = model.hamiltonian_from_meanfield(correction)
    keys = list(total_h)
    coords = selected_coordinates(total_h[zero], keys)
    points = 2 * np.pi * np.arange(case.nk) / case.nk
    matrices = [
        sum(
            value.toarray() * np.exp(-1j * key[0] * point)
            for key, value in total_h.items()
        )
        for point in points
    ]

    def reference(mu, include_density=False):
        total_charge = total_energy = total_entropy = 0.0
        values = np.zeros(coords.value_count, complex)
        for matrix, point in zip(matrices, points, strict=True):
            energies, vectors = np.linalg.eigh(matrix - np.diag(mu * q_diag))
            rho = (vectors * expit(-energies / case.kT)) @ vectors.conj().T
            total_charge += np.dot(weights, rho.diagonal().real)
            if include_density:
                total_energy += float(np.trace(matrix @ rho).real)
                occupation = expit(-energies / case.kT)
                total_entropy += float(
                    np.sum(entr(occupation) + entr(expit(energies / case.kT)))
                )
                phases = np.exp(1j * point * np.array(keys)[:, 0])
                values += coords.values_from_assembled_matrix(rho, phases=phases)
        if model.superconducting:
            total_energy = 0.5 * total_energy + 0.5 * case.nk * float(
                np.sum(total_h[zero].diagonal()[: case.size]).real
            )
            total_entropy *= 0.5
        normalization = 1 if args.per_cell else case.size
        return (
            total_charge / case.nk,
            values / case.nk,
            total_energy / (case.nk * normalization),
            total_entropy / (case.nk * normalization),
        )

    mu_reference = brentq(
        lambda mu: reference(mu)[0] - model.filling, -20.0, 20.0, xtol=2e-13
    )
    reference_charge, reference_values, _, _ = reference(mu_reference, True)
    options = rational_options(scheme)
    # This harness also runs the earlier AAA/Ozaki checkouts.
    grid = meanfi.UniformGrid if hasattr(meanfi, "UniformGrid") else meanfi.PeriodicGrid
    integration = grid(nk=case.nk, matrix_function=options)

    def evaluate():
        result = meanfi.density_matrix(
            model,
            mean_field=correction,
            coordinates=coords,
            integration=integration,
            tol=case.tolerance,
            filling_tol=case.tolerance,
            mu_tol=1e-11,
        )
        extra = dict(
            mu=result.mu,
            mu_error=abs(result.mu - mu_reference),
            charge_evaluations=result.statistics.charge_evaluations,
        )
        if (
            getattr(result, "entropy", None) is not None
            and result.band_energy is not None
        ):
            start = time.perf_counter()
            _, _, band_energy, entropy = reference(result.mu, True)
            extra.update(
                reference_seconds=time.perf_counter() - start,
                energy_error=abs(result.band_energy - band_energy),
                entropy_error=abs(result.entropy - entropy),
                entropy_approximation_error=getattr(
                    result.errors, "entropy_approximation", None
                ),
            )
        return result.filling, result.values, extra

    runs = {"cold": [], "warm": []}
    for _ in range(args.repeat):
        if any(run["status"] == "failure" for run in runs["cold"] + runs["warm"]):
            break
        clear_scalar_cache()
        gc.collect()
        for state in ("cold", "warm"):
            runs[state].append(
                measure(
                    evaluate,
                    reference_charge=reference_charge,
                    reference_values=reference_values,
                    tolerance=case.tolerance,
                )
            )
            if runs[state][-1]["status"] == "failure":
                break
    return dict(
        case=asdict(case),
        scheme=scheme,
        matrix_size=coords.size,
        nnz=sum(m.nnz for m in total_h.values()),
        selected_entries=coords.value_count,
        reference_mu=mu_reference,
        reference_charge=reference_charge,
        runs=runs,
    )


def summarize(record):
    summary = {}
    for state, runs in record["runs"].items():
        if not runs:
            summary[state] = None
            continue
        summary[state] = dict(
            median_seconds=statistics.median(run["seconds"] for run in runs),
            successes=sum(run["status"] == "ok" for run in runs),
            accurate=sum(run["accurate"] for run in runs),
            median_factorizations=statistics.median(
                run["factorizations"] for run in runs
            ),
            max_charge_error=max(
                (run["charge_error"] for run in runs if "charge_error" in run),
                default=None,
            ),
            max_density_error=max(
                (run["density_error"] for run in runs if "density_error" in run),
                default=None,
            ),
        )
    return summary


try:
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except subprocess.CalledProcessError:
    revision = None
report = dict(
    purpose="Sparse rational accuracy and cost comparison",
    thermodynamics=args.thermodynamics or args.large_matrices,
    large_matrices=args.large_matrices,
    public_thermodynamics_per_orbital=not args.per_cell,
    certified_ozaki=args.certify_ozaki,
    timing_note=(
        "Study-only Ozaki certification is included in total seconds, "
        "but scalar_fit_seconds instruments only production _sparse_terms. "
        "scalar_build_seconds is its subset spent constructing new AAA fits; "
        "the difference includes cache validation and spectral bounds."
    ),
    rational_source_sha256=hashlib.sha256(
        b"".join(
            path.name.encode() + path.read_bytes()
            for path in sorted(
                (
                    args.checkout / "meanfi/density/kpoint/matrix_functions/rational"
                ).glob("*.py")
            )
        )
    ).hexdigest(),
    source_sha256=hashlib.sha256(
        b"".join(
            str(path.relative_to(args.checkout)).encode() + path.read_bytes()
            for path in sorted((args.checkout / "meanfi").rglob("*.py"))
        )
    ).hexdigest(),
    checkout=str(args.checkout.resolve()),
    revision=args.source_revision or revision,
    imported_package=meanfi.__file__,
    python=platform.python_version(),
    numpy=np.__version__,
    scipy=scipy.__version__,
    cpu=next(
        (
            line.partition(":")[2].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    ),
    affinity=list(os.sched_getaffinity(0)),
    threadpools=threadpool_info(),
    repeat=args.repeat,
    max_poles=args.max_poles,
    measurement_timeout_seconds=args.timeout,
    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    records=[],
)


def write_report():
    # Keep metadata readable and each measurement record on one reviewable line.
    metadata = {key: value for key, value in report.items() if key != "records"}
    header = json.dumps(metadata, indent=2).removesuffix("\n}")
    records = ",\n".join("    " + json.dumps(record) for record in report["records"])
    args.output.write_text(header + ',\n  "records": [\n' + records + "\n  ]\n}\n")


args.output.parent.mkdir(parents=True, exist_ok=True)
for case in cases():
    if args.only and not any(part in case.name for part in args.only.split(",")):
        continue
    if args.thermodynamics and case.workflow != "node":
        continue
    # Alternate scheme order by case to reduce one-direction warm-up bias.
    order = ("aaa", "ozaki") if len(report["records"]) % 4 == 0 else ("ozaki", "aaa")
    for scheme in order:
        if scheme == "ozaki" and _ozaki_terms is None:
            continue
        if args.scheme and scheme != args.scheme:
            continue
        record = (
            node_case(case, scheme)
            if case.workflow == "node"
            else filling_case(case, scheme)
        )
        record["summary"] = summarize(record)
        report["records"].append(record)
        write_report()
        print(case.name, scheme, json.dumps(record["summary"]), flush=True)
report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
write_report()
