"""Separate quadratic interpolation, Schur, and affine-charge convergence.

Run against the production FermiSimplex package. Polynomial extrema and a
two-band Schur complement give analytic references. Absolute comparison
tolerance 2e-12 allows double-precision matrix assembly, well below the
smallest measured leading error. No algorithm selector or production change.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from fermisimplex import SpectralMesh


SIZES = (0.4, 0.2, 0.1, 0.05, 0.025)


def add_orders(rows, names):
    for previous, current in zip(rows, rows[1:]):
        for name in names:
            current[name + "_order"] = float(np.log2(previous[name] / current[name]))
    return rows


def interpolation_cases():
    rows = []
    for degree, origin in ((3, 0.0), (4, 0.0), (4, 0.7)):
        family = []
        for h in SIZES:
            polynomial = Polynomial([origin, h]) ** degree
            interpolant = (
                polynomial(0) * Polynomial([1, -3, 2])
                + polynomial(0.5) * Polynomial([0, 4, -4])
                + polynomial(1) * Polynomial([0, -1, 2])
            )
            residual = polynomial - interpolant
            extrema = [0.0, 1.0] + [
                z.real
                for z in residual.deriv().roots()
                if abs(z.imag) < 1e-12 and 0 < z.real < 1
            ]
            exact_error = float(max(abs(residual(t)) for t in extrema))

            def model(t):
                return np.array([[polynomial(t)]], complex)

            enclosure = SpectralMesh(model, root_level=0).occupation_enclosures(
                mu=0, depth=0
            )[0]
            assert exact_error <= enclosure.interpolation_error + 2e-12
            family.append(
                dict(
                    degree=degree,
                    origin=origin,
                    h=h,
                    exact_error=exact_error,
                    sampled_allowance=enclosure.interpolation_error,
                )
            )
        add_orders(family, ("exact_error", "sampled_allowance"))
        expected = 4 if degree == 4 and origin == 0 else 3
        assert abs(family[-1]["exact_error_order"] - expected) < 0.08
        assert abs(family[-1]["sampled_allowance_order"] - expected) < 0.08
        rows.extend(family)
    return rows


def coupled_case():
    rows = []
    gap, slope, coupling, crossing = 2.0, 0.3, 0.4, 0.37
    for h in SIZES:

        def model(t):
            x = h * t
            return np.array(
                [[x - crossing * h, coupling * x], [coupling * x, gap + slope * x]],
                complex,
            )

        mesh = SpectralMesh(model, root_level=0)
        enclosure = mesh.occupation_enclosures(mu=0, depth=0)[0]
        assert enclosure.active_dimension == 1
        eta = enclosure.interpolation_error
        x_bound = coupling * h / gap
        d_bound = slope * h + eta
        cubic_part = eta + 2 * eta * x_bound + d_bound * x_bound**2
        quartic_bound = (eta + d_bound * x_bound) ** 2 / enclosure.safe_gap
        assert abs(enclosure.model_error - cubic_part - quartic_bound) < 2e-12
        # Exact Schur error S-P increases on [0,h]; its maximum is at h.
        schur_error = coupling**2 * slope * h**3 / (gap * (gap + slope * h))
        # F=B-DX is quadratic, so the exact correction F*D^-1 F is quartic.
        exact_correction = (slope * h * x_bound) ** 2 / (gap + slope * h)
        assert schur_error <= enclosure.model_error + 2e-12
        assert exact_correction <= quartic_bound + 2e-12
        inferred_quartic = enclosure.model_error - cubic_part
        assert inferred_quartic > 0
        # det H=0 gives the sole occupied/unoccupied crossing analytically.
        roots = Polynomial(
            [-crossing * h * gap, gap - crossing * h * slope, slope - coupling**2]
        ).roots()
        root = next(
            float(z.real) for z in roots if abs(z.imag) < 1e-12 and 0 < z.real < h
        )
        affine_charge = mesh.estimate_charge_on_current_mesh(mu=0).value
        # Map the native unit interval back to the physical interval [0,h].
        charge_error = abs(h * affine_charge - root)
        rows.append(
            dict(
                h=h,
                interpolation_allowance=eta,
                schur_error=schur_error,
                model_allowance=enclosure.model_error,
                cubic_part=cubic_part,
                native_quartic_bound=inferred_quartic,
                exact_quartic_correction=exact_correction,
                physical_charge_error=charge_error,
            )
        )
    add_orders(
        rows,
        (
            "schur_error",
            "model_allowance",
            "native_quartic_bound",
            "exact_quartic_correction",
            "physical_charge_error",
        ),
    )
    for name, expected in (
        ("schur_error", 3),
        ("model_allowance", 3),
        ("native_quartic_bound", 4),
        ("exact_quartic_correction", 4),
        ("physical_charge_error", 2),
    ):
        assert abs(rows[-1][name + "_order"] - expected) < 0.08
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(interpolation=interpolation_cases(), coupled=coupled_case())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
