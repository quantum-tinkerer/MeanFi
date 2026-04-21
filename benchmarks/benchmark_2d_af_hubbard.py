from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from meanfi import Model, solver


def _bipartite_hubbard_2d(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {
        (0, 0): hop + hop.T.conj(),
        (1, 0): hop,
        (-1, 0): hop.T.conj(),
        (0, 1): hop,
        (0, -1): hop.T.conj(),
        (1, 1): hop,
        (-1, -1): hop.T.conj(),
    }
    h_int = {(0, 0): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def _antiferromagnetic_guess(delta: float):
    return {(0, 0): np.diag([-delta, delta, delta, -delta]).astype(complex)}


def _centered_diagonal(diag: np.ndarray) -> list[float]:
    centered = np.asarray(diag, dtype=float) - float(np.mean(diag))
    return [float(x) for x in centered]


def _af_split(diag: np.ndarray) -> float:
    centered = np.asarray(_centered_diagonal(diag), dtype=float)
    return float(0.5 * (centered[1] - centered[0]))


def _run_current_adaptive(args: argparse.Namespace) -> dict:
    h_0, h_int = _bipartite_hubbard_2d(args.U)
    guess = _antiferromagnetic_guess(args.delta)
    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=0.0,
        charge_tol=args.charge_tol,
        density_atol=args.density_atol,
        scf_tol=args.scf_tol,
        max_subdivisions=args.max_subdivisions,
    )
    solver_kwargs = dict(
        optimizer=None,
        optimizer_kwargs={"alpha": args.alpha},
        max_scf_steps=args.max_scf_steps,
        return_info=True,
    )

    for _ in range(args.warmups):
        solver(model, guess, **solver_kwargs)

    t0 = time.perf_counter()
    solution, info = solver(model, guess, **solver_kwargs)
    t1 = time.perf_counter()

    onsite_diag = np.real(np.diag(solution[(0, 0)]))
    return {
        "method": "adaptive_current",
        "wall_s": t1 - t0,
        "iterations": int(info.iterations),
        "residual_norm": float(info.residual_norm),
        "mu": float(info.mu),
        "total_charge_integration_calls": int(info.total_charge_integration_calls),
        "total_density_integration_calls": int(info.total_density_integration_calls),
        "total_kernel_evals": int(info.total_kernel_evals),
        "total_evaluator_evals": int(info.total_evaluator_evals),
        "wall_per_kernel_eval_s": float((t1 - t0) / info.total_kernel_evals),
        "onsite_diag_real": [float(x) for x in onsite_diag],
        "onsite_diag_centered_real": _centered_diagonal(onsite_diag),
        "af_split": _af_split(onsite_diag),
    }


_MAIN_BENCH_SNIPPET = r"""
import json
import time
import numpy as np

import meanfi
from meanfi.params.rparams import tb_to_rparams
from meanfi.tb.tb import add_tb


def bipartite_hubbard_2d(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {
        (0, 0): hop + hop.T.conj(),
        (1, 0): hop,
        (-1, 0): hop.T.conj(),
        (0, 1): hop,
        (0, -1): hop.T.conj(),
        (1, 1): hop,
        (-1, -1): hop.T.conj(),
    }
    h_int = {(0, 0): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def antiferromagnetic_guess(delta: float):
    return {(0, 0): np.diag([-delta, delta, delta, -delta]).astype(complex)}


stats = {}


def centered_local(tb, local_key):
    centered = dict(tb)
    local = np.array(centered[local_key], copy=True)
    shift = float(np.trace(local).real / local.shape[0])
    centered[local_key] = local - shift * np.eye(local.shape[0])
    return centered


def linear_mixing_optimizer(f, x0, alpha=0.5, maxiter=100, f_tol=1e-3):
    x = np.array(x0, copy=True)
    residual_norm = float("inf")
    for iteration in range(1, maxiter + 1):
        residual = np.asarray(f(x), dtype=float)
        residual_norm = float(np.linalg.norm(residual))
        stats["iterations"] = iteration
        stats["last_iteration_residual_norm"] = residual_norm
        if residual_norm <= f_tol:
            return x
        x = x + alpha * residual
    raise RuntimeError(f"No convergence after {maxiter} iterations; residual={residual_norm}")


U = float(os.environ["MEANFI_BENCH_U"])
delta = float(os.environ["MEANFI_BENCH_DELTA"])
nk = int(os.environ["MEANFI_BENCH_NK"])
alpha = float(os.environ["MEANFI_BENCH_ALPHA"])
scf_tol = float(os.environ["MEANFI_BENCH_SCF_TOL"])
max_scf_steps = int(os.environ["MEANFI_BENCH_MAX_SCF_STEPS"])
warmups = int(os.environ["MEANFI_BENCH_WARMUPS"])

h_0, h_int = bipartite_hubbard_2d(U)
guess = antiferromagnetic_guess(delta)
model = meanfi.Model(h_0, h_int, filling=2.0)
solver_kwargs = dict(
    nk=nk,
    optimizer=linear_mixing_optimizer,
    optimizer_kwargs={"alpha": alpha, "maxiter": max_scf_steps, "f_tol": scf_tol},
)

for _ in range(warmups):
    meanfi.solver(model, guess, **solver_kwargs)

stats.clear()
t0 = time.perf_counter()
solution = meanfi.solver(model, guess, **solver_kwargs)
t1 = time.perf_counter()

onsite_diag = np.real(np.diag(solution[(0, 0)]))
h_full = add_tb(h_0, solution)
rho_final, fermi_final = meanfi.density_matrix(h_full, filling=model.filling, nk=nk)
mf_recomputed = meanfi.meanfield(rho_final, h_int)
solution_recomputed = add_tb(
    mf_recomputed,
    {model._local_key: -float(fermi_final) * np.eye(model._ndof)},
)
posthoc_centered_residual = float(
    np.linalg.norm(
        tb_to_rparams(centered_local(solution_recomputed, model._local_key))
        - tb_to_rparams(centered_local(solution, model._local_key))
    )
)
grid_evals = int((stats["iterations"] + 2) * nk * nk)

print(
    json.dumps(
        {
            "method": "dense_main",
            "wall_s": t1 - t0,
            "iterations": int(stats["iterations"]),
            "last_iteration_residual_norm": float(stats["last_iteration_residual_norm"]),
            "posthoc_centered_correction_residual_norm": posthoc_centered_residual,
            "nk": nk,
            "kpoints_per_grid_eval": int(nk * nk),
            "total_kpoint_evals": grid_evals,
            "wall_per_kpoint_eval_s": float((t1 - t0) / grid_evals),
            "fermi_final": float(fermi_final),
            "onsite_diag_real": [float(x) for x in onsite_diag],
        },
        sort_keys=True,
    )
)
"""


def _run_main_dense(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    keep = args.keep_main_worktree
    worktree_dir = Path(
        tempfile.mkdtemp(prefix="meanfi-main-bench.", dir=args.tmp_dir)
        if args.tmp_dir is not None
        else tempfile.mkdtemp(prefix="meanfi-main-bench.")
    )
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_dir), "main"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{worktree_dir}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(worktree_dir)
        )
        env["MEANFI_BENCH_U"] = str(args.U)
        env["MEANFI_BENCH_DELTA"] = str(args.delta)
        env["MEANFI_BENCH_NK"] = str(args.dense_nk)
        env["MEANFI_BENCH_ALPHA"] = str(args.alpha)
        env["MEANFI_BENCH_SCF_TOL"] = str(args.scf_tol)
        env["MEANFI_BENCH_MAX_SCF_STEPS"] = str(args.max_scf_steps)
        env["MEANFI_BENCH_WARMUPS"] = str(args.warmups)
        completed = subprocess.run(
            [sys.executable, "-c", "import os\n" + _MAIN_BENCH_SNIPPET],
            cwd=worktree_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        onsite = np.asarray(result["onsite_diag_real"], dtype=float)
        result["onsite_diag_centered_real"] = _centered_diagonal(onsite)
        result["af_split"] = _af_split(onsite)
        if keep:
            result["main_worktree"] = str(worktree_dir)
        return result
    finally:
        if keep:
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the current adaptive kT=0 2D AF Hubbard solver against dense-grid main."
    )
    parser.add_argument("--U", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scf-tol", type=float, default=1e-3)
    parser.add_argument("--charge-tol", type=float, default=1e-3)
    parser.add_argument("--density-atol", type=float, default=1e-3)
    parser.add_argument("--max-scf-steps", type=int, default=100)
    parser.add_argument("--max-subdivisions", type=int, default=50_000)
    parser.add_argument("--dense-nk", type=int, default=201)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--keep-main-worktree", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default=None)
    args = parser.parse_args()

    adaptive = _run_current_adaptive(args)
    results = {"adaptive_current": adaptive}
    if not args.skip_main:
        dense = _run_main_dense(args)
        results["dense_main"] = dense
        results["comparison"] = {
            "wall_ratio_current_over_main": adaptive["wall_s"] / dense["wall_s"],
            "per_sample_ratio_current_over_main": (
                adaptive["wall_per_kernel_eval_s"] / dense["wall_per_kpoint_eval_s"]
            ),
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
