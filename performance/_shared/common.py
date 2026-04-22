from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_method_name(method: object) -> str:
    name = method.__class__.__name__
    return {
        "AdaptiveQuadrature": "adaptive_quadrature",
        "AdaptiveSimplex": "adaptive_simplex",
        "UniformGrid": "uniform_grid",
        "LinearMixing": "linear_mixing",
        "AndersonMixing": "anderson_mixing",
    }.get(name, name.lower())


def unique_eval_count(info: object, *, total: bool = False) -> int:
    if total:
        fields = ("total_unique_evals", "total_kernel_evals", "total_n_kpoints")
    else:
        fields = ("unique_evals", "n_kernel_evals", "n_kpoints")
    for field in fields:
        value = getattr(info, field, None)
        if value is not None:
            return int(value)
    return 0


def _base_record(
    *,
    scenario: str,
    method: str,
    workflow: str,
    kT: float,
    ndof: int,
    wall_s: float,
    peak_memory_bytes: int | None,
    unique_evals: int,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "method": method,
        "workflow": workflow,
        "kT": float(kT),
        "ndof": int(ndof),
        "wall_s": float(wall_s),
        "peak_memory_bytes": None if peak_memory_bytes is None else int(peak_memory_bytes),
        "unique_evals": int(unique_evals),
        "wall_per_unique_eval_s": None
        if unique_evals <= 0
        else float(wall_s / unique_evals),
    }


def density_record(
    *,
    scenario: str,
    workflow: str,
    integration: object,
    kT: float,
    ndof: int,
    benchmark_result,
    density_result,
    density_matrix_error: float | None,
    filling_error: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = density_result.info
    record = _base_record(
        scenario=scenario,
        method=canonical_method_name(integration),
        workflow=workflow,
        kT=kT,
        ndof=ndof,
        wall_s=benchmark_result.median_s,
        peak_memory_bytes=benchmark_result.peak_traced_bytes,
        unique_evals=unique_eval_count(info),
    )
    record.update(
        {
            "n_kernel_evals": getattr(info, "n_kernel_evals", None),
            "n_evaluator_evals": getattr(info, "n_evaluator_evals", None),
            "root_iterations": getattr(info, "root_iterations", None),
            "scf_iterations": None,
            "n_kpoints": getattr(info, "n_kpoints", None),
            "total_unique_evals": None,
            "density_matrix_error": None
            if density_matrix_error is None
            else float(density_matrix_error),
            "filling_error": None if filling_error is None else float(filling_error),
            "scf_residual": None,
        }
    )
    if extra:
        record.update(extra)
    return record


def scf_record(
    *,
    scenario: str,
    integration: object,
    scf_method: object,
    kT: float,
    ndof: int,
    benchmark_result,
    solver_result,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = solver_result.info
    record = _base_record(
        scenario=scenario,
        method=canonical_method_name(integration),
        workflow="scf",
        kT=kT,
        ndof=ndof,
        wall_s=benchmark_result.median_s,
        peak_memory_bytes=benchmark_result.peak_traced_bytes,
        unique_evals=unique_eval_count(info, total=True),
    )
    record.update(
        {
            "n_kernel_evals": getattr(
                solver_result.density_matrix_result.info,
                "n_kernel_evals",
                None,
            ),
            "n_evaluator_evals": getattr(info, "total_evaluator_evals", None),
            "root_iterations": None,
            "scf_iterations": getattr(info, "iterations", None),
            "n_kpoints": getattr(
                solver_result.density_matrix_result.info,
                "n_kpoints",
                None,
            ),
            "total_unique_evals": getattr(info, "total_unique_evals", None),
            "density_matrix_error": None,
            "filling_error": None,
            "scf_residual": float(info.residual_norm),
            "scf_method": canonical_method_name(scf_method),
        }
    )
    if extra:
        record.update(extra)
    return record


def write_records(records: list[dict[str, Any]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_summary(records: list[dict[str, Any]]) -> None:
    for record in records:
        wall = f"{record['wall_s']:.6f}s"
        unique = record["unique_evals"]
        per_unique = record["wall_per_unique_eval_s"]
        parts = [
            f"{record['scenario']}",
            f"[{record['method']}]",
            f"workflow={record['workflow']}",
            f"wall={wall}",
            f"unique_evals={unique}",
        ]
        if per_unique is not None:
            parts.append(f"wall/unique={per_unique:.6e}s")
        if record.get("density_matrix_error") is not None:
            parts.append(f"density_error={record['density_matrix_error']:.3e}")
        if record.get("scf_residual") is not None:
            parts.append(f"scf_residual={record['scf_residual']:.3e}")
        print(" ".join(parts))
