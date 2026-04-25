import argparse
import json
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize._nonlin import NoConvergence

import meanfi
from meanfi.kwant_helper import kwant_examples, utils
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "profiling" / "compare_main_vs_quadrature_output"
MAIN_FILES = [
    "meanfi/tb/__init__.py",
    "meanfi/tb/tb.py",
    "meanfi/tb/transforms.py",
    "meanfi/mf.py",
    "meanfi/params/__init__.py",
    "meanfi/params/rparams.py",
    "meanfi/observables.py",
    "meanfi/kwant_helper/__init__.py",
    "meanfi/kwant_helper/utils.py",
    "meanfi/kwant_helper/kwant_examples.py",
    "meanfi/tb/utils.py",
    "meanfi/model.py",
    "meanfi/solvers.py",
    "meanfi/__init__.py",
]
MAIN_SOLVER_STEPS = 80
CURRENT_SOLVER_STEPS = 80
CURRENT_MIXING_KWARGS = {"M": 0, "line_search": "wolfe"}
RETRY_SUBDIVISIONS = 50_000
EPSILON = 1e-12


def copy_tb(tb: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in tb.items()}


def canonical_key(key: Any) -> tuple[int, ...]:
    return tuple(int(component) for component in key)


def restrict_tb(tb: dict[Any, np.ndarray], keys: list[Any]) -> dict[Any, np.ndarray]:
    canonical_tb = {canonical_key(key): value for key, value in tb.items()}
    return {
        key: np.array(canonical_tb[canonical_key(key)], copy=True)
        for key in keys
    }


def find_local_key(tb: dict[Any, np.ndarray]) -> Any:
    for key in tb:
        if all(int(component) == 0 for component in key):
            return key
    raise ValueError("Could not find local key.")


def max_abs_tb_difference(
    lhs: dict[Any, np.ndarray], rhs: dict[Any, np.ndarray], keys: list[Any]
) -> float:
    return max(float(np.max(np.abs(lhs[key] - rhs[key]))) for key in keys)


def max_relative_tb_difference(
    lhs: dict[Any, np.ndarray], rhs: dict[Any, np.ndarray], keys: list[Any]
) -> float:
    diffs = []
    for key in keys:
        lhs_norm = float(np.linalg.norm(lhs[key]))
        rhs_norm = float(np.linalg.norm(rhs[key]))
        denom = max(lhs_norm, rhs_norm, EPSILON)
        diffs.append(float(np.linalg.norm(lhs[key] - rhs[key]) / denom))
    return max(diffs)


def format_number(value: float | None, *, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    if not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e3 or magnitude < 1e-2:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def resolve_ref(ref: str) -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--verify", ref],
            cwd=REPO_ROOT,
            text=True,
        )
        .strip()
    )


def git_show(ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        text=True,
    )


def rewrite_imports(source: str, alias: str) -> str:
    source = source.replace("from meanfi.", f"from {alias}.")
    source = source.replace("import meanfi.", f"import {alias}.")
    source = re.sub(r"(^|\n)from meanfi import", rf"\1from {alias} import", source)
    source = re.sub(
        r"(^|\n)import meanfi(\n|$)",
        rf"\1import {alias} as meanfi\2",
        source,
    )
    return source


def load_ref_package(ref: str, alias: str) -> types.ModuleType:
    package = types.ModuleType(alias)
    package.__path__ = []
    package.__package__ = alias
    sys.modules[alias] = package
    for subpackage in ("tb", "params", "kwant_helper"):
        module_name = f"{alias}.{subpackage}"
        module = types.ModuleType(module_name)
        module.__path__ = []
        module.__package__ = module_name
        sys.modules[module_name] = module

    for path in MAIN_FILES:
        source = rewrite_imports(git_show(ref, path), alias)
        if path == "meanfi/__init__.py":
            module_name = alias
            module = sys.modules[module_name]
        else:
            module_name = path[:-3].replace("/", ".").replace("meanfi", alias, 1)
            module = sys.modules.get(module_name, types.ModuleType(module_name))
            sys.modules[module_name] = module
        module.__file__ = f"<git:{ref}:{path}>"
        module.__package__ = module_name.rsplit(".", 1)[0] if "." in module_name else module_name
        exec(compile(source, module.__file__, "exec"), module.__dict__)

    return sys.modules[alias]


def is_quadrature_failure(exc: Exception) -> bool:
    message = str(exc)
    return "Stateful adaptive quadrature did not converge" in message


def compute_gap(tb: dict[Any, np.ndarray], nk_dense: int) -> float:
    kham = meanfi.tb_to_kgrid(tb, nk_dense)
    eigenvalues = np.linalg.eigvalsh(kham)
    below = eigenvalues[eigenvalues <= 0]
    above = eigenvalues[eigenvalues > 0]
    if below.size == 0 or above.size == 0:
        return float("nan")
    return float(np.min(above) - np.max(below))


def staggered_magnetization(rho: dict[Any, np.ndarray], local_key: Any) -> float:
    occupations = np.real(np.diag(rho[local_key]))
    magnetization = 0.5 * (
        (occupations[0] - occupations[1]) + (occupations[3] - occupations[2])
    )
    return float(abs(magnetization))


def graphene_cdw(rho: dict[Any, np.ndarray], local_key: Any) -> float:
    operator = {local_key: np.kron(np.diag([1, -1]), np.eye(2))}
    return float(abs(meanfi.expectation_value(rho, operator)))


@dataclass(frozen=True)
class Thresholds:
    gap: float
    order: float
    meanfield: float


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    label: str
    order_label: str
    order_function: Callable[[dict[Any, np.ndarray], Any], float]
    build_system: Callable[[], tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], float]]
    seed: int
    guess_scale: float
    nk_values: tuple[int, ...]
    kT_values: tuple[float, ...]
    tolerance_values: tuple[float, ...]
    max_mu_iterations: int
    max_subdivisions: int
    scf_tol: float
    dense_gap_nk: int
    thresholds: Thresholds
    gapless_floor: float


@dataclass
class CaseContext:
    definition: CaseDefinition
    h_0: dict[Any, np.ndarray]
    h_int: dict[Any, np.ndarray]
    filling: float
    keys: list[Any]
    local_key: Any
    ndof: int
    guess: dict[Any, np.ndarray]


@dataclass
class MainFullResult:
    nk: int
    success: bool
    runtime_s: float
    h_full: dict[Any, np.ndarray] | None = None
    rho: dict[Any, np.ndarray] | None = None
    meanfield: dict[Any, np.ndarray] | None = None
    physical_mu: float | None = None
    order_value: float | None = None
    gap: float | None = None
    failure: str | None = None


@dataclass
class MainDensityResult:
    nk: int
    success: bool
    runtime_s: float
    rho: dict[Any, np.ndarray] | None = None
    meanfield: dict[Any, np.ndarray] | None = None
    density_mu: float | None = None
    order_value: float | None = None
    gap: float | None = None
    failure: str | None = None


@dataclass
class CurrentFullResult:
    kT: float
    tolerance: float
    max_subdivisions: int
    success: bool
    runtime_s: float
    h_full: dict[Any, np.ndarray] | None = None
    rho: dict[Any, np.ndarray] | None = None
    meanfield: dict[Any, np.ndarray] | None = None
    physical_mu: float | None = None
    order_value: float | None = None
    gap: float | None = None
    retry_note: str = "none"
    failure: str | None = None
    solver_info: Any | None = None
    density_info: Any | None = None


@dataclass
class CurrentDensityResult:
    kT: float
    tolerance: float
    max_subdivisions: int
    success: bool
    runtime_s: float
    rho: dict[Any, np.ndarray] | None = None
    meanfield: dict[Any, np.ndarray] | None = None
    density_mu: float | None = None
    order_value: float | None = None
    gap: float | None = None
    retry_note: str = "none"
    failure: str | None = None
    density_info: Any | None = None


@dataclass
class ComparisonMetrics:
    gap_abs_diff: float
    order_abs_diff: float
    meanfield_max_rel_diff: float
    rho_max_abs_diff: float
    mu_abs_diff: float
    accepted: bool


@dataclass
class CaseComparisonSummary:
    case: str
    label: str
    ref_sha: str
    chosen_nk: int
    chosen_kT: float
    chosen_tolerance: float
    retry_note: str
    acceptable: bool
    suitability_note: str
    density_only: dict[str, Any]
    full_scf: dict[str, Any]
    sweeps: dict[str, Any]
    selection_notes: list[str] = field(default_factory=list)


def build_hubbard_1d() -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], float]:
    hopping = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hopping + hopping.T.conj(), (1,): hopping, (-1,): hopping.T.conj()}
    h_int = {(0,): 8.0 * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int, 2.0


def build_graphene() -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], float]:
    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, {"U": 0.2, "V": 1.2})
    return h_0, h_int, 2.0


CASE_DEFINITIONS = {
    "hubbard1d": CaseDefinition(
        name="hubbard1d",
        label="1D Bipartite Hubbard",
        order_label="abs delta m_staggered",
        order_function=staggered_magnetization,
        build_system=build_hubbard_1d,
        seed=123,
        guess_scale=0.2,
        nk_values=(100, 200, 400),
        kT_values=(0.02, 0.01, 0.005),
        tolerance_values=(1e-8,),
        max_mu_iterations=64,
        max_subdivisions=10_000,
        scf_tol=1e-8,
        dense_gap_nk=4_000,
        thresholds=Thresholds(gap=1e-5, order=1e-4, meanfield=5e-3),
        gapless_floor=1e-4,
    ),
    "graphene": CaseDefinition(
        name="graphene",
        label="Interacting Graphene",
        order_label="abs delta CDW",
        order_function=graphene_cdw,
        build_system=build_graphene,
        seed=7,
        guess_scale=0.2,
        nk_values=(18, 24, 36),
        kT_values=(0.05, 0.02, 0.01),
        tolerance_values=(1e-3, 3e-4, 1e-4),
        max_mu_iterations=80,
        max_subdivisions=10_000,
        scf_tol=5e-5,
        dense_gap_nk=160,
        thresholds=Thresholds(gap=1e-3, order=1e-3, meanfield=1e-3),
        gapless_floor=1e-2,
    ),
}


class ComparisonRunner:
    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.ref_sha = resolve_ref(ref)
        alias = f"meanfi_ref_{self.ref_sha[:12]}"
        self.ref_pkg = load_ref_package(ref, alias)
        self.current_pkg = meanfi
        self._contexts: dict[str, CaseContext] = {}
        self._main_full_cache: dict[tuple[str, int], MainFullResult] = {}
        self._current_full_cache: dict[
            tuple[str, float, float, int], CurrentFullResult
        ] = {}

    def prepare_case(self, case_name: str) -> CaseContext:
        if case_name in self._contexts:
            return self._contexts[case_name]

        definition = CASE_DEFINITIONS[case_name]
        h_0, h_int, filling = definition.build_system()
        keys = list(h_int.keys())
        local_key = find_local_key(h_0)
        ndof = next(iter(h_0.values())).shape[0]
        np.random.seed(definition.seed)
        guess = self.current_pkg.guess_tb(
            frozenset(h_int),
            ndof,
            scale=definition.guess_scale,
        )
        context = CaseContext(
            definition=definition,
            h_0=h_0,
            h_int=h_int,
            filling=filling,
            keys=keys,
            local_key=local_key,
            ndof=ndof,
            guess=guess,
        )
        self._contexts[case_name] = context
        return context

    def run_main_full_scf(self, context: CaseContext, nk: int) -> MainFullResult:
        cache_key = (context.definition.name, nk)
        if cache_key in self._main_full_cache:
            return self._main_full_cache[cache_key]

        start = time.perf_counter()
        try:
            model = self.ref_pkg.Model(context.h_0, context.h_int, filling=context.filling)
            solution = self.ref_pkg.solver(
                model,
                copy_tb(context.guess),
                nk=nk,
                optimizer_kwargs={
                    "M": 0,
                    "line_search": "wolfe",
                    "maxiter": MAIN_SOLVER_STEPS,
                    "f_tol": context.definition.scf_tol,
                },
            )
            h_full = self.current_pkg.add_tb(context.h_0, solution)
            rho_full, _ = self.ref_pkg.density_matrix(h_full, filling=context.filling, nk=nk)
            rho = restrict_tb(rho_full, context.keys)
            physical_mf = self.current_pkg.meanfield(rho, context.h_int)
            physical_h = self.current_pkg.add_tb(context.h_0, physical_mf)
            physical_mu = float(
                self.ref_pkg.fermi_energy(physical_h, filling=context.filling, nk=nk)
            )
            result = MainFullResult(
                nk=nk,
                success=True,
                runtime_s=time.perf_counter() - start,
                h_full=h_full,
                rho=rho,
                meanfield=physical_mf,
                physical_mu=physical_mu,
                order_value=context.definition.order_function(rho, context.local_key),
                gap=compute_gap(h_full, context.definition.dense_gap_nk),
            )
        except NoConvergence as exc:
            rho_params = np.asarray(exc.args[0], dtype=float)
            rho_trial = rparams_to_tb(rho_params, list(context.h_int), context.ndof)
            physical_mf = self.current_pkg.meanfield(rho_trial, context.h_int)
            physical_h = self.current_pkg.add_tb(context.h_0, physical_mf)
            physical_mu = float(
                self.ref_pkg.fermi_energy(physical_h, filling=context.filling, nk=nk)
            )
            local_shift = {
                context.local_key: -physical_mu * np.eye(context.ndof, dtype=complex)
            }
            solution = self.current_pkg.add_tb(physical_mf, local_shift)
            h_full = self.current_pkg.add_tb(context.h_0, solution)
            rho_full, _ = self.ref_pkg.density_matrix(h_full, filling=context.filling, nk=nk)
            rho = restrict_tb(rho_full, context.keys)
            residual = float(np.linalg.norm(tb_to_rparams(rho) - rho_params))
            result = MainFullResult(
                nk=nk,
                success=True,
                runtime_s=time.perf_counter() - start,
                h_full=h_full,
                rho=rho,
                meanfield=self.current_pkg.meanfield(rho, context.h_int),
                physical_mu=physical_mu,
                order_value=context.definition.order_function(rho, context.local_key),
                gap=compute_gap(h_full, context.definition.dense_gap_nk),
                failure=(
                    f"used last iterate after NoConvergence at maxiter={MAIN_SOLVER_STEPS}; "
                    f"residual={residual:.3e}"
                ),
            )
        except Exception as exc:
            result = MainFullResult(
                nk=nk,
                success=False,
                runtime_s=time.perf_counter() - start,
                failure=str(exc),
            )

        self._main_full_cache[cache_key] = result
        return result

    def run_main_density(
        self, context: CaseContext, hamiltonian: dict[Any, np.ndarray], nk: int
    ) -> MainDensityResult:
        start = time.perf_counter()
        try:
            rho_full, density_mu = self.ref_pkg.density_matrix(
                hamiltonian, filling=context.filling, nk=nk
            )
            rho = restrict_tb(rho_full, context.keys)
            result = MainDensityResult(
                nk=nk,
                success=True,
                runtime_s=time.perf_counter() - start,
                rho=rho,
                meanfield=self.current_pkg.meanfield(rho, context.h_int),
                density_mu=float(density_mu),
                order_value=context.definition.order_function(rho, context.local_key),
                gap=compute_gap(hamiltonian, context.definition.dense_gap_nk),
            )
        except Exception as exc:
            result = MainDensityResult(
                nk=nk,
                success=False,
                runtime_s=time.perf_counter() - start,
                failure=str(exc),
            )
        return result

    def _run_current_full_once(
        self,
        context: CaseContext,
        kT: float,
        tolerance: float,
        max_subdivisions: int,
    ) -> CurrentFullResult:
        start = time.perf_counter()
        model = self.current_pkg.Model(
            context.h_0,
            context.h_int,
            filling=context.filling,
            kT=kT,
            charge_tol=tolerance,
            density_atol=tolerance,
            scf_tol=context.definition.scf_tol,
        )
        note = None
        try:
            solution, solver_info = self.current_pkg.solver(
                model,
                copy_tb(context.guess),
                mixing="anderson",
                mixing_kwargs=dict(CURRENT_MIXING_KWARGS),
                max_scf_steps=CURRENT_SOLVER_STEPS,
                max_mu_iterations=context.definition.max_mu_iterations,
                max_subdivisions=max_subdivisions,
                return_info=True,
            )
            h_full = self.current_pkg.add_tb(context.h_0, solution)
            rho, _, _, density_info = self.current_pkg.density_matrix(
                h_full,
                filling=context.filling,
                kT=kT,
                keys=context.keys,
                charge_tol=tolerance,
                density_atol=tolerance,
                max_mu_iterations=context.definition.max_mu_iterations,
                max_subdivisions=max_subdivisions,
            )
        except NoConvergence as exc:
            rho_params = np.asarray(exc.args[0], dtype=float)
            rho_trial = rparams_to_tb(rho_params, context.keys, context.ndof)
            rho, _, mu_final, density_info = model.density_matrix(
                rho_trial,
                keys=context.keys,
                max_mu_iterations=context.definition.max_mu_iterations,
                max_subdivisions=max_subdivisions,
            )
            residual = float(np.linalg.norm(tb_to_rparams(rho) - rho_params))
            physical_mf = self.current_pkg.meanfield(rho, context.h_int)
            solution = dict(physical_mf)
            solution[context.local_key] = solution.get(
                context.local_key,
                np.zeros((context.ndof, context.ndof), dtype=complex),
            ) - mu_final * np.eye(context.ndof, dtype=complex)
            h_full = self.current_pkg.add_tb(context.h_0, solution)
            solver_info = types.SimpleNamespace(
                iterations=CURRENT_SOLVER_STEPS,
                residual_norm=residual,
                mu=mu_final,
                total_charge_integration_calls=None,
                total_density_integration_calls=None,
                total_kernel_evals=None,
                total_evaluator_evals=None,
            )
            note = (
                f"used last iterate after NoConvergence at maxiter={CURRENT_SOLVER_STEPS}; "
                f"residual={residual:.3e}"
            )

        return CurrentFullResult(
            kT=kT,
            tolerance=tolerance,
            max_subdivisions=max_subdivisions,
            success=True,
            runtime_s=time.perf_counter() - start,
            h_full=h_full,
            rho=rho,
            meanfield=self.current_pkg.meanfield(rho, context.h_int),
            physical_mu=float(solver_info.mu),
            order_value=context.definition.order_function(rho, context.local_key),
            gap=compute_gap(h_full, context.definition.dense_gap_nk),
            failure=note,
            solver_info=solver_info,
            density_info=density_info,
        )

    def run_current_full_scf(
        self,
        context: CaseContext,
        kT: float,
        tolerance: float,
    ) -> CurrentFullResult:
        cache_key = (
            context.definition.name,
            float(kT),
            float(tolerance),
            context.definition.max_subdivisions,
        )
        if cache_key in self._current_full_cache:
            return self._current_full_cache[cache_key]

        try:
            result = self._run_current_full_once(
                context,
                kT=kT,
                tolerance=tolerance,
                max_subdivisions=context.definition.max_subdivisions,
            )
        except Exception as exc:
            if is_quadrature_failure(exc):
                try:
                    result = self._run_current_full_once(
                        context,
                        kT=kT,
                        tolerance=tolerance,
                        max_subdivisions=RETRY_SUBDIVISIONS,
                    )
                    result.retry_note = (
                        f"retried at max_subdivisions={RETRY_SUBDIVISIONS} after failure at "
                        f"{context.definition.max_subdivisions}"
                    )
                except Exception as retry_exc:
                    result = CurrentFullResult(
                        kT=kT,
                        tolerance=tolerance,
                        max_subdivisions=RETRY_SUBDIVISIONS,
                        success=False,
                        runtime_s=0.0,
                        retry_note=(
                            f"retry failed at max_subdivisions={RETRY_SUBDIVISIONS} after initial "
                            f"failure at {context.definition.max_subdivisions}"
                        ),
                        failure=str(retry_exc),
                    )
            else:
                result = CurrentFullResult(
                    kT=kT,
                    tolerance=tolerance,
                    max_subdivisions=context.definition.max_subdivisions,
                    success=False,
                    runtime_s=0.0,
                    failure=str(exc),
                )

        self._current_full_cache[cache_key] = result
        return result

    def run_current_density(
        self,
        context: CaseContext,
        hamiltonian: dict[Any, np.ndarray],
        kT: float,
        tolerance: float,
    ) -> CurrentDensityResult:
        def run_once(max_subdivisions: int) -> CurrentDensityResult:
            start = time.perf_counter()
            rho, _, density_mu, density_info = self.current_pkg.density_matrix(
                hamiltonian,
                filling=context.filling,
                kT=kT,
                keys=context.keys,
                charge_tol=tolerance,
                density_atol=tolerance,
                max_mu_iterations=context.definition.max_mu_iterations,
                max_subdivisions=max_subdivisions,
            )
            return CurrentDensityResult(
                kT=kT,
                tolerance=tolerance,
                max_subdivisions=max_subdivisions,
                success=True,
                runtime_s=time.perf_counter() - start,
                rho=rho,
                meanfield=self.current_pkg.meanfield(rho, context.h_int),
                density_mu=float(density_mu),
                order_value=context.definition.order_function(rho, context.local_key),
                gap=compute_gap(hamiltonian, context.definition.dense_gap_nk),
                density_info=density_info,
            )

        try:
            return run_once(context.definition.max_subdivisions)
        except Exception as exc:
            if is_quadrature_failure(exc):
                try:
                    result = run_once(RETRY_SUBDIVISIONS)
                    result.retry_note = (
                        f"retried at max_subdivisions={RETRY_SUBDIVISIONS} after failure at "
                        f"{context.definition.max_subdivisions}"
                    )
                    return result
                except Exception as retry_exc:
                    return CurrentDensityResult(
                        kT=kT,
                        tolerance=tolerance,
                        max_subdivisions=RETRY_SUBDIVISIONS,
                        success=False,
                        runtime_s=0.0,
                        retry_note=(
                            f"retry failed at max_subdivisions={RETRY_SUBDIVISIONS} after initial "
                            f"failure at {context.definition.max_subdivisions}"
                        ),
                        failure=str(retry_exc),
                    )
            return CurrentDensityResult(
                kT=kT,
                tolerance=tolerance,
                max_subdivisions=context.definition.max_subdivisions,
                success=False,
                runtime_s=0.0,
                failure=str(exc),
            )


def compare_results(
    lhs: Any,
    rhs: Any,
    context: CaseContext,
    *,
    mu_attribute: str,
) -> ComparisonMetrics:
    gap_diff = abs(float(lhs.gap) - float(rhs.gap))
    order_diff = abs(float(lhs.order_value) - float(rhs.order_value))
    meanfield_diff = max_relative_tb_difference(lhs.meanfield, rhs.meanfield, context.keys)
    rho_diff = max_abs_tb_difference(lhs.rho, rhs.rho, context.keys)
    mu_diff = abs(float(getattr(lhs, mu_attribute)) - float(getattr(rhs, mu_attribute)))
    thresholds = context.definition.thresholds
    accepted = (
        gap_diff <= thresholds.gap
        and order_diff <= thresholds.order
        and meanfield_diff <= thresholds.meanfield
    )
    return ComparisonMetrics(
        gap_abs_diff=gap_diff,
        order_abs_diff=order_diff,
        meanfield_max_rel_diff=meanfield_diff,
        rho_max_abs_diff=rho_diff,
        mu_abs_diff=mu_diff,
        accepted=accepted,
    )


def serialise_main_result(result: MainFullResult) -> dict[str, Any]:
    return {
        "nk": result.nk,
        "success": result.success,
        "runtime_ms": 1e3 * result.runtime_s,
        "gap": result.gap,
        "order_value": result.order_value,
        "physical_mu": result.physical_mu,
        "failure": result.failure,
    }


def serialise_current_result(result: CurrentFullResult) -> dict[str, Any]:
    payload = {
        "kT": result.kT,
        "tolerance": result.tolerance,
        "success": result.success,
        "runtime_ms": 1e3 * result.runtime_s,
        "gap": result.gap,
        "order_value": result.order_value,
        "physical_mu": result.physical_mu,
        "retry_note": result.retry_note,
        "failure": result.failure,
    }
    if result.solver_info is not None:
        payload.update(
            {
                "iterations": result.solver_info.iterations,
                "residual_norm": result.solver_info.residual_norm,
                "charge_calls": result.solver_info.total_charge_integration_calls,
                "density_calls": result.solver_info.total_density_integration_calls,
                "kernel_evals": result.solver_info.total_kernel_evals,
                "evaluator_evals": result.solver_info.total_evaluator_evals,
            }
        )
    return payload


def pick_main_nk(
    context: CaseContext,
    runner: ComparisonRunner,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    sweep = [runner.run_main_full_scf(context, nk) for nk in context.definition.nk_values]
    notes = []
    chosen_nk = context.definition.nk_values[-1]

    for lower, higher in zip(sweep[:-1], sweep[1:]):
        if not lower.success or not higher.success:
            continue
        metrics = compare_results(lower, higher, context, mu_attribute="physical_mu")
        if metrics.accepted:
            chosen_nk = lower.nk
            notes.append(
                f"Selected nk={lower.nk} because it matches nk={higher.nk} within the "
                "acceptance thresholds."
            )
            break
    else:
        notes.append(
            f"No nk plateau found within thresholds; using the tightest grid nk={chosen_nk}."
        )

    return chosen_nk, [serialise_main_result(result) for result in sweep], notes


def pick_current_kT(
    context: CaseContext,
    runner: ComparisonRunner,
    tolerance: float,
) -> tuple[float, list[dict[str, Any]], list[str]]:
    ordered_kT = tuple(sorted(context.definition.kT_values))
    sweep = [runner.run_current_full_scf(context, kT, tolerance) for kT in ordered_kT]
    notes = []
    successful = [result for result in sweep if result.success]
    if not successful:
        raise RuntimeError(f"All current-branch kT runs failed for case {context.definition.name}.")

    chosen_kT = successful[-1].kT
    for smaller, larger in zip(sweep[:-1], sweep[1:]):
        if not smaller.success or not larger.success:
            continue
        metrics = compare_results(smaller, larger, context, mu_attribute="physical_mu")
        if metrics.accepted:
            chosen_kT = smaller.kT
            notes.append(
                f"Selected kT={smaller.kT} because it is the smallest sampled temperature on a "
                f"plateau with kT={larger.kT}."
            )
            break
    else:
        notes.append(
            f"No kT plateau found within thresholds; using the smallest successful kT={chosen_kT}."
        )

    return chosen_kT, [serialise_current_result(result) for result in sweep], notes


def pick_current_tolerance(
    context: CaseContext,
    runner: ComparisonRunner,
    kT: float,
) -> tuple[float, CurrentFullResult, list[dict[str, Any]], list[str]]:
    tolerances = tuple(sorted(context.definition.tolerance_values, reverse=True))
    sweep = [runner.run_current_full_scf(context, kT, tolerance) for tolerance in tolerances]
    notes = []
    baseline = next(
        (result for result in reversed(sweep) if result.success),
        None,
    )
    if baseline is None:
        raise RuntimeError(
            f"All current-branch tolerance runs failed for case {context.definition.name} at kT={kT}."
        )

    chosen = baseline
    for candidate in sweep:
        if not candidate.success:
            continue
        metrics = compare_results(candidate, baseline, context, mu_attribute="physical_mu")
        if metrics.accepted:
            chosen = candidate
            notes.append(
                f"Selected tolerance={candidate.tolerance} because it matches the tight reference "
                f"tolerance={baseline.tolerance} within thresholds."
            )
            break

    if chosen is baseline and baseline.tolerance != tolerances[0]:
        notes.append(
            f"Looser tolerances moved the observables; keeping tolerance={baseline.tolerance}."
        )
    elif chosen is baseline and baseline.tolerance == tolerances[0]:
        notes.append(f"Using the loosest sampled tolerance={chosen.tolerance}.")

    return (
        chosen.tolerance,
        chosen,
        [serialise_current_result(result) for result in sweep],
        notes,
    )


def density_stage_payload(
    main_result: MainDensityResult,
    current_result: CurrentDensityResult,
    comparison: ComparisonMetrics,
) -> dict[str, Any]:
    payload = {
        "metrics": {
            "gap_abs_diff": comparison.gap_abs_diff,
            "order_abs_diff": comparison.order_abs_diff,
            "meanfield_max_rel_diff": comparison.meanfield_max_rel_diff,
            "rho_max_abs_diff": comparison.rho_max_abs_diff,
            "mu_abs_diff": comparison.mu_abs_diff,
            "accepted": comparison.accepted,
        },
        "main": {
            "nk": main_result.nk,
            "runtime_ms": 1e3 * main_result.runtime_s,
            "density_mu": main_result.density_mu,
            "gap": main_result.gap,
            "order_value": main_result.order_value,
        },
        "current": {
            "kT": current_result.kT,
            "tolerance": current_result.tolerance,
            "runtime_ms": 1e3 * current_result.runtime_s,
            "density_mu": current_result.density_mu,
            "gap": current_result.gap,
            "order_value": current_result.order_value,
            "retry_note": current_result.retry_note,
            "charge_calls": (
                current_result.density_info.charge_integration_calls
                if current_result.density_info is not None
                else None
            ),
            "density_calls": (
                current_result.density_info.density_integration_calls
                if current_result.density_info is not None
                else None
            ),
            "kernel_evals": (
                current_result.density_info.n_kernel_evals
                if current_result.density_info is not None
                else None
            ),
            "evaluator_evals": (
                current_result.density_info.n_evaluator_evals
                if current_result.density_info is not None
                else None
            ),
        },
    }
    return payload


def full_stage_payload(
    main_result: MainFullResult,
    current_result: CurrentFullResult,
    comparison: ComparisonMetrics,
) -> dict[str, Any]:
    payload = {
        "metrics": {
            "gap_abs_diff": comparison.gap_abs_diff,
            "order_abs_diff": comparison.order_abs_diff,
            "meanfield_max_rel_diff": comparison.meanfield_max_rel_diff,
            "rho_max_abs_diff": comparison.rho_max_abs_diff,
            "mu_abs_diff": comparison.mu_abs_diff,
            "accepted": comparison.accepted,
        },
        "main": {
            "nk": main_result.nk,
            "runtime_ms": 1e3 * main_result.runtime_s,
            "physical_mu": main_result.physical_mu,
            "gap": main_result.gap,
            "order_value": main_result.order_value,
        },
        "current": {
            "kT": current_result.kT,
            "tolerance": current_result.tolerance,
            "runtime_ms": 1e3 * current_result.runtime_s,
            "physical_mu": current_result.physical_mu,
            "gap": current_result.gap,
            "order_value": current_result.order_value,
            "retry_note": current_result.retry_note,
            "iterations": (
                current_result.solver_info.iterations
                if current_result.solver_info is not None
                else None
            ),
            "residual_norm": (
                current_result.solver_info.residual_norm
                if current_result.solver_info is not None
                else None
            ),
            "charge_calls": (
                current_result.solver_info.total_charge_integration_calls
                if current_result.solver_info is not None
                else None
            ),
            "density_calls": (
                current_result.solver_info.total_density_integration_calls
                if current_result.solver_info is not None
                else None
            ),
            "kernel_evals": (
                current_result.solver_info.total_kernel_evals
                if current_result.solver_info is not None
                else None
            ),
            "evaluator_evals": (
                current_result.solver_info.total_evaluator_evals
                if current_result.solver_info is not None
                else None
            ),
        },
    }
    return payload


def evaluate_case(runner: ComparisonRunner, case_name: str) -> CaseComparisonSummary:
    context = runner.prepare_case(case_name)

    chosen_nk, nk_sweep, nk_notes = pick_main_nk(context, runner)
    chosen_main = runner.run_main_full_scf(context, chosen_nk)
    if not chosen_main.success:
        raise RuntimeError(f"Chosen main-branch run failed for case {case_name}.")

    tight_tolerance = min(context.definition.tolerance_values)
    chosen_kT, kT_sweep, kT_notes = pick_current_kT(context, runner, tight_tolerance)
    chosen_tolerance, chosen_current, tolerance_sweep, tolerance_notes = pick_current_tolerance(
        context, runner, chosen_kT
    )

    if not chosen_current.success:
        raise RuntimeError(f"Chosen current-branch run failed for case {case_name}.")

    main_density = runner.run_main_density(context, chosen_main.h_full, chosen_nk)
    if not main_density.success:
        raise RuntimeError(f"Density-only main run failed for case {case_name}.")

    current_density = runner.run_current_density(
        context,
        chosen_main.h_full,
        chosen_kT,
        chosen_tolerance,
    )
    if not current_density.success:
        raise RuntimeError(f"Density-only current run failed for case {case_name}.")

    density_metrics = compare_results(
        main_density,
        current_density,
        context,
        mu_attribute="density_mu",
    )
    full_metrics = compare_results(
        chosen_main,
        chosen_current,
        context,
        mu_attribute="physical_mu",
    )

    suitability_note = "gapped"
    acceptable = density_metrics.accepted and full_metrics.accepted
    if not np.isfinite(chosen_main.gap) or chosen_main.gap <= context.definition.gapless_floor:
        acceptable = False
        suitability_note = (
            f"unsuitable: main-branch gap {format_number(chosen_main.gap)} is below "
            f"the floor {format_number(context.definition.gapless_floor)}"
        )

    return CaseComparisonSummary(
        case=context.definition.name,
        label=context.definition.label,
        ref_sha=runner.ref_sha,
        chosen_nk=chosen_nk,
        chosen_kT=chosen_kT,
        chosen_tolerance=chosen_tolerance,
        retry_note=chosen_current.retry_note,
        acceptable=acceptable,
        suitability_note=suitability_note,
        density_only=density_stage_payload(main_density, current_density, density_metrics),
        full_scf=full_stage_payload(chosen_main, chosen_current, full_metrics),
        sweeps={
            "main_nk": nk_sweep,
            "current_kT": kT_sweep,
            "current_tolerance": tolerance_sweep,
        },
        selection_notes=[*nk_notes, *kT_notes, *tolerance_notes],
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def build_markdown_report(summary: dict[str, Any]) -> str:
    full_rows = []
    density_rows = []
    for case in summary["cases"]:
        full = case["full_scf"]
        density = case["density_only"]
        full_rows.append(
            [
                case["case"],
                case["ref_sha"][:12],
                str(case["chosen_nk"]),
                format_number(case["chosen_kT"]),
                format_number(case["chosen_tolerance"]),
                case["retry_note"],
                format_number(full["metrics"]["gap_abs_diff"]),
                format_number(full["metrics"]["order_abs_diff"]),
                format_number(full["metrics"]["meanfield_max_rel_diff"]),
                format_number(full["metrics"]["rho_max_abs_diff"]),
                format_number(full["metrics"]["mu_abs_diff"]),
                format_number(full["current"]["runtime_ms"]),
                str(full["current"]["charge_calls"]),
                str(full["current"]["density_calls"]),
                "yes" if case["acceptable"] else "no",
            ]
        )
        density_rows.append(
            [
                case["case"],
                case["ref_sha"][:12],
                str(case["chosen_nk"]),
                format_number(case["chosen_kT"]),
                format_number(case["chosen_tolerance"]),
                density["current"]["retry_note"],
                format_number(density["metrics"]["gap_abs_diff"]),
                format_number(density["metrics"]["order_abs_diff"]),
                format_number(density["metrics"]["meanfield_max_rel_diff"]),
                format_number(density["metrics"]["rho_max_abs_diff"]),
                format_number(density["metrics"]["mu_abs_diff"]),
                format_number(density["current"]["runtime_ms"]),
                str(density["current"]["charge_calls"]),
                str(density["current"]["density_calls"]),
                "yes" if density["metrics"]["accepted"] else "no",
            ]
        )

    lines = [
        "# Main vs Quadrature Comparison",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Reference ref: `{summary['ref']}`",
        f"- Reference SHA: `{summary['ref_sha']}`",
        "",
        "## Full SCF Summary",
        "",
        markdown_table(
            [
                "case",
                "ref sha",
                "nk",
                "kT",
                "tol",
                "retry/escalation",
                "abs delta gap",
                "abs delta order",
                "max rel diff(meanfield)",
                "max abs diff(rho[key])",
                "abs delta mu",
                "runtime ms",
                "charge calls",
                "density calls",
                "accepted",
            ],
            full_rows,
        ),
        "",
        "## Density-Only Summary",
        "",
        markdown_table(
            [
                "case",
                "ref sha",
                "nk",
                "kT",
                "tol",
                "retry/escalation",
                "abs delta gap",
                "abs delta order",
                "max rel diff(meanfield)",
                "max abs diff(rho[key])",
                "abs delta mu",
                "runtime ms",
                "charge calls",
                "density calls",
                "accepted",
            ],
            density_rows,
        ),
        "",
    ]

    for case in summary["cases"]:
        order_label = CASE_DEFINITIONS[case["case"]].order_label
        lines.extend(
            [
                f"## {case['label']}",
                "",
                f"- Suitability: `{case['suitability_note']}`",
                f"- Chosen config: `nk={case['chosen_nk']}`, `kT={case['chosen_kT']}`, "
                f"`tol={case['chosen_tolerance']}`",
                f"- Acceptance: `{'pass' if case['acceptable'] else 'fail'}`",
                "",
                "Selection notes:",
            ]
        )
        for note in case["selection_notes"]:
            lines.append(f"- {note}")
        lines.extend(
            [
                "",
                "Main nk sweep:",
                "",
                markdown_table(
                    ["nk", "success", "runtime ms", "gap", order_label, "mu", "failure"],
                    [
                        [
                            str(item["nk"]),
                            "yes" if item["success"] else "no",
                            format_number(item["runtime_ms"]),
                            format_number(item["gap"]),
                            format_number(item["order_value"]),
                            format_number(item["physical_mu"]),
                            item["failure"] or "",
                        ]
                        for item in case["sweeps"]["main_nk"]
                    ],
                ),
                "",
                "Current kT sweep:",
                "",
                markdown_table(
                    [
                        "kT",
                        "tol",
                        "success",
                        "runtime ms",
                        "gap",
                        order_label,
                        "mu",
                        "retry",
                        "failure",
                    ],
                    [
                        [
                            format_number(item["kT"]),
                            format_number(item["tolerance"]),
                            "yes" if item["success"] else "no",
                            format_number(item["runtime_ms"]),
                            format_number(item["gap"]),
                            format_number(item["order_value"]),
                            format_number(item["physical_mu"]),
                            item["retry_note"],
                            item["failure"] or "",
                        ]
                        for item in case["sweeps"]["current_kT"]
                    ],
                ),
                "",
                "Current tolerance sweep:",
                "",
                markdown_table(
                    [
                        "kT",
                        "tol",
                        "success",
                        "runtime ms",
                        "gap",
                        order_label,
                        "mu",
                        "retry",
                        "failure",
                    ],
                    [
                        [
                            format_number(item["kT"]),
                            format_number(item["tolerance"]),
                            "yes" if item["success"] else "no",
                            format_number(item["runtime_ms"]),
                            format_number(item["gap"]),
                            format_number(item["order_value"]),
                            format_number(item["physical_mu"]),
                            item["retry_note"],
                            item["failure"] or "",
                        ]
                        for item in case["sweeps"]["current_tolerance"]
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines)


def build_json_summary(ref: str, ref_sha: str, cases: list[CaseComparisonSummary]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ref": ref,
        "ref_sha": ref_sha,
        "cases": [
            {
                "case": case.case,
                "label": case.label,
                "ref_sha": case.ref_sha,
                "chosen_nk": case.chosen_nk,
                "chosen_kT": case.chosen_kT,
                "chosen_tolerance": case.chosen_tolerance,
                "retry_note": case.retry_note,
                "acceptable": case.acceptable,
                "suitability_note": case.suitability_note,
                "density_only": case.density_only,
                "full_scf": case.full_scf,
                "sweeps": case.sweeps,
                "selection_notes": case.selection_notes,
            }
            for case in cases
        ],
    }


def print_console_summary(summary: dict[str, Any]) -> None:
    print("Main vs quadrature summary")
    print(f"  ref: {summary['ref']} ({summary['ref_sha'][:12]})")
    for case in summary["cases"]:
        full = case["full_scf"]["metrics"]
        print(f"  {case['case']}:")
        print(
            f"    config nk={case['chosen_nk']} kT={case['chosen_kT']} "
            f"tol={case['chosen_tolerance']} accepted={case['acceptable']}"
        )
        print(
            f"    full SCF |Delta gap|={format_number(full['gap_abs_diff'])} "
            f"|Delta order|={format_number(full['order_abs_diff'])} "
            f"max rel diff(meanfield)={format_number(full['meanfield_max_rel_diff'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare main-branch k-grid mean-field generation against the quadrature implementation."
    )
    parser.add_argument("--ref", default="main", help="Git ref to load as the k-grid reference.")
    parser.add_argument(
        "--case",
        choices=("hubbard1d", "graphene", "all"),
        default="all",
        help="Case to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where summary.md and summary.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ComparisonRunner(args.ref)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = list(CASE_DEFINITIONS) if args.case == "all" else [args.case]
    summaries = [evaluate_case(runner, case_name) for case_name in case_names]
    json_summary = build_json_summary(args.ref, runner.ref_sha, summaries)
    markdown_summary = build_markdown_report(json_summary)

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    summary_json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(markdown_summary, encoding="utf-8")

    print_console_summary(json_summary)
    print(f"  wrote {summary_md_path}")
    print(f"  wrote {summary_json_path}")


if __name__ == "__main__":
    main()
