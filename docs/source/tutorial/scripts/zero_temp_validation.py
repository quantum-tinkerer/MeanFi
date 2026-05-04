from __future__ import annotations

import argparse
import json
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.optimize import brentq

import meanfi

GRAPHENE_REFERENCE_POINTS = {
    "metal": {"U": 0.2, "V": 0.2},
    "cdw": {"U": 0.2, "V": 1.2},
    "sdw": {"U": 3.0, "V": 0.2},
}

STRAINED_GRAPHENE_REFERENCE = {"U": 0.8}


def staggered_magnetization(local_density: np.ndarray) -> float:
    """Return the absolute staggered magnetization of a bipartite spinful cell."""

    occupations = np.real(np.diag(local_density))
    magnetization = 0.5 * (
        (occupations[0] - occupations[1]) + (occupations[3] - occupations[2])
    )
    return float(abs(magnetization))


def band_gap(
    h: dict[tuple[int, ...], np.ndarray], *, fermi_energy: float = 0.0, nk: int
) -> float:
    """Return the direct gap extracted from a uniform post-processing k-grid."""

    h_kgrid = meanfi.tb_to_kgrid(h, nk)
    eigenvalues = np.linalg.eigvalsh(h_kgrid)
    emax = np.max(eigenvalues[eigenvalues <= fermi_energy])
    emin = np.min(eigenvalues[eigenvalues > fermi_energy])
    return float(abs(emin - emax))


def gap_history_converged(
    history: Iterable[tuple[int, float]], *, atol: float, rtol: float
) -> bool:
    """Check whether the last two post-processing gap evaluations agree."""

    history = list(history)
    if len(history) < 2:
        return False
    _, previous_gap = history[-2]
    _, current_gap = history[-1]
    return abs(current_gap - previous_gap) <= max(
        float(atol),
        float(rtol) * max(abs(previous_gap), abs(current_gap)),
    )


def refined_band_gap(
    h: dict[tuple[int, ...], np.ndarray],
    *,
    fermi_energy: float = 0.0,
    nk_initial: int = 100,
    nk_max: int = 6400,
    atol: float = 5e-4,
    rtol: float = 5e-2,
) -> tuple[float, list[tuple[int, float]]]:
    """Refine the post-processing k-grid until the direct gap stops moving."""

    nk = int(nk_initial)
    history: list[tuple[int, float]] = []

    while True:
        gap = band_gap(h, fermi_energy=fermi_energy, nk=nk)
        history.append((nk, gap))
        if gap_history_converged(history, atol=atol, rtol=rtol) or nk >= nk_max:
            return gap, history
        nk *= 2


def resolved_hubbard_gap(
    h: dict[tuple[int, ...], np.ndarray],
    *,
    U: float,
    local_density: np.ndarray,
    fermi_energy: float = 0.0,
    nk_initial: int = 100,
    nk_max: int = 6400,
    gap_atol: float = 5e-4,
    gap_rtol: float = 5e-2,
    order_parameter_tol: float = 1e-5,
) -> tuple[float, dict[str, object]]:
    """Resolve the 1D Hubbard gap while avoiding a coarse-grid post-processing floor."""

    direct_gap, history = refined_band_gap(
        h,
        fermi_energy=fermi_energy,
        nk_initial=nk_initial,
        nk_max=nk_max,
        atol=gap_atol,
        rtol=gap_rtol,
    )
    magnetization = staggered_magnetization(local_density)
    order_parameter_gap = float(abs(U) * magnetization)
    use_order_parameter_gap = (
        order_parameter_gap <= order_parameter_tol
        or not gap_history_converged(history, atol=gap_atol, rtol=gap_rtol)
    )

    info = {
        "band_gap": direct_gap,
        "gap_history": history,
        "order_parameter_gap": order_parameter_gap,
        "staggered_magnetization": magnetization,
        "used_order_parameter_gap": use_order_parameter_gap,
    }
    if use_order_parameter_gap:
        return order_parameter_gap, info
    return direct_gap, info


def hubbard_reference_gap(U: float, *, nk: int = 400_000) -> float:
    """Return the dense zero-temperature Hartree-Fock gap for the 1D Hubbard chain."""

    hopp = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
    hkfunc = meanfi.tb_to_kfunc(h_0)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)[:, None]
    h_k = hkfunc(ks)
    spin_up_block = h_k[:, [0, 2]][:, :, [0, 2]]
    gamma_abs = np.abs(spin_up_block[:, 0, 1])

    def residual(delta: float) -> float:
        energy = np.sqrt(gamma_abs**2 + delta**2)
        return 1.0 - 0.5 * U * np.mean(1.0 / energy)

    if residual(1e-14) >= 0.0:
        return 0.0

    upper = max(10.0, 2.0 * U)
    delta = float(brentq(residual, 1e-14, upper))
    return 2.0 * delta


def _build_graphene_inputs():
    import kwant
    from meanfi.interop import kwant as utils

    s0 = np.identity(2)
    graphene = kwant.lattice.general(
        [(1, 0), (1 / 2, np.sqrt(3) / 2)],
        [(0, 0), (0, 1 / np.sqrt(3))],
        norbs=2,
    )
    a, b = graphene.sublattices
    bulk_graphene = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
    bulk_graphene[a.shape((lambda pos: True), (0, 0))] = 0 * s0
    bulk_graphene[b.shape((lambda pos: True), (0, 0))] = 0 * s0
    bulk_graphene[graphene.neighbors(1)] = s0
    h_0 = utils.builder_to_tb(bulk_graphene)

    def onsite_int(site, U):
        del site
        return U * np.array([[0, 1], [1, 0]])

    def nn_int(site1, site2, V):
        del site1, site2
        return V * np.ones((2, 2))

    builder_int = utils.build_interacting_syst(
        builder=bulk_graphene,
        lattice=graphene,
        func_onsite=onsite_int,
        func_hop=nn_int,
        max_neighbor=1,
    )
    h_int = utils.builder_to_tb(builder_int, GRAPHENE_REFERENCE_POINTS["metal"])
    return h_0, builder_int, frozenset(h_int), len(list(h_0.values())[0])


def graphene_reference_suite(
) -> dict[str, dict[str, float]]:
    """Run the graphene tutorial reference points with the zero-temperature settings."""

    from meanfi.interop import kwant as utils

    np.random.seed(0)
    h_0, builder_int, int_keys, ndof = _build_graphene_inputs()
    sz = np.diag([1, -1])
    s_list = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    ]
    cdw_operator = {(0, 0): np.kron(sz, np.eye(2))}

    results = {}
    for label, params in GRAPHENE_REFERENCE_POINTS.items():
        h_int = utils.builder_to_tb(builder_int, params)
        model = meanfi.Model(h_0, h_int, filling=2)
        guess = meanfi.guess_tb(int_keys, ndof)
        solver_result = meanfi.solver(
            model,
            guess,
            scf=meanfi.AndersonMixing(M=0, line_search="wolfe", max_iterations=80),
            scf_tol=2e-2,
            filling_tol=1e-2,
        )
        h_full = meanfi.add_tb(h_0, solver_result.mf)
        density_result = meanfi.density_matrix(
            h_full,
            filling=2,
            keys=[(0, 0)],
            filling_tol=1e-2,
        )
        rho = density_result.density_matrix

        cdw = abs(meanfi.expectation_value(rho, cdw_operator))
        sdw_sq = 0.0
        for s_i in s_list:
            sdw_sq += abs(
                meanfi.expectation_value(rho, {(0, 0): np.kron(sz, s_i)})
            ) ** 2

        results[label] = {
            "U": float(params["U"]),
            "V": float(params["V"]),
            "residual_norm": float(solver_result.info.residual_norm),
            "charge_error": float(abs(density_result.filling - 2.0)),
            "cdw": float(cdw),
            "sdw_sq": float(sdw_sq),
            "gap_nk_200": band_gap(h_full, nk=200),
        }

    return results


def _build_strained_graphene_inputs():
    from meanfi.interop import kwant as utils

    try:
        from .strained_graphene_kwant import create_system
    except ImportError:  # pragma: no cover - direct script execution fallback
        from strained_graphene_kwant import create_system

    h0_builder, lat, k_path = create_system(n=16)

    def interaction_hop(site1, site2):
        del site1, site2
        return 0 * np.ones((2, 2))

    def interaction_onsite(site, U):
        del site
        return U * np.ones((2, 2))

    int_builder = utils.build_interacting_syst(
        h0_builder, lat, interaction_onsite, interaction_hop, max_neighbor=0
    )
    h0_dense, data = utils.builder_to_tb(h0_builder, params={"xi": 7}, return_data=True)
    h_int_dense = utils.builder_to_tb(int_builder, STRAINED_GRAPHENE_REFERENCE)
    h0 = {key: sparse.csr_matrix(value) for key, value in h0_dense.items()}
    h_int = {key: sparse.csr_matrix(value) for key, value in h_int_dense.items()}
    filling = int(next(iter(h0.values())).shape[0]) // 2

    def guess_onsite(site):
        if site.family == lat.sublattices[0]:
            return np.diag([1.0, -1.0])
        return np.diag([-1.0, 1.0])

    guess_builder = utils.build_interacting_syst(
        h0_builder, lat, guess_onsite, interaction_hop, max_neighbor=0
    )
    guess_dense = utils.builder_to_tb(guess_builder)
    guess = {key: sparse.csr_matrix(value) for key, value in guess_dense.items()}
    return h0, h_int, guess, filling, data, k_path


def solve_strained_graphene_reference(
    *,
    max_scf_steps: int = 100,
) -> dict[str, float]:
    """Run the strained-graphene tutorial reference point."""

    h0, h_int, guess, filling, _, _ = _build_strained_graphene_inputs()
    integration = meanfi.UniformGrid(nk=2, density_matrix_tol=1e-1)
    model = meanfi.Model(h0, h_int, filling=filling, kT=0.01)
    solver_result = meanfi.solver(
        model,
        guess,
        integration=integration,
        scf=meanfi.AndersonMixing(
            M=10,
            line_search="armijo",
            max_iterations=max_scf_steps,
        ),
        scf_tol=2e-2,
        filling_tol=2.0,
    )
    mf_ham = meanfi.add_tb(h0, solver_result.mf)
    return {
        "residual_norm": float(solver_result.info.residual_norm),
        "charge_integrations": float(solver_result.info.total_charge_integration_calls),
        "gap_nk_40": band_gap(mf_ham, nk=40),
    }


def _build_hubbard_inputs(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hop + hop.T.conj(), (1,): hop, (-1,): hop.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    guess = meanfi.guess_tb(frozenset(h_int), 4)
    return h_0, h_int, guess


def solve_hubbard_reference(
    *,
    U: float = 4.0 / 29.0,
) -> dict[str, object]:
    """Run the 1D Hubbard zero-temperature reference point used by the regression tests."""

    np.random.seed(0)
    h_0, h_int, guess = _build_hubbard_inputs(U)
    model = meanfi.Model(h_0, h_int, filling=2.0)
    solver_result = meanfi.solver(
        model,
        guess,
        scf=meanfi.AndersonMixing(M=0, line_search="wolfe", max_iterations=80),
        scf_tol=2e-3,
        filling_tol=1e-3,
    )
    h_full = meanfi.add_tb(h_0, solver_result.mf)
    density_result = meanfi.density_matrix(
        h_full,
        filling=2.0,
        keys=[(0,)],
        filling_tol=1e-3,
    )
    resolved_gap, gap_info = resolved_hubbard_gap(
        h_full,
        U=U,
        local_density=density_result.density_matrix[(0,)],
    )
    return {
        "residual_norm": float(solver_result.info.residual_norm),
        "charge_error": float(abs(density_result.filling - 2.0)),
        "staggered_magnetization": staggered_magnetization(
            density_result.density_matrix[(0,)]
        ),
        "resolved_gap": float(resolved_gap),
        "reference_gap": float(hubbard_reference_gap(U)),
        "gap_info": gap_info,
    }


def _json_ready(obj):
    if isinstance(obj, dict):
        return {key: _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run zero-temperature tutorial validation probes."
    )
    parser.add_argument(
        "tutorial",
        choices=("hubbard", "graphene", "strained"),
        help="Tutorial validation target to run.",
    )
    args = parser.parse_args(argv)

    if args.tutorial == "hubbard":
        payload = solve_hubbard_reference()
    elif args.tutorial == "graphene":
        payload = graphene_reference_suite()
    else:
        payload = solve_strained_graphene_reference()

    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
