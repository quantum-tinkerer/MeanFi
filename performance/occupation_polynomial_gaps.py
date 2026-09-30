"""Cubic/quartic gap claims with analytic sign and occupation references."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from fermisimplex import CertificateStatus, SpectralMesh, certify_simplex


def extrema(poly):
    points = [0.0, 1.0]
    points += [
        float(z.real)
        for z in poly.deriv().roots()
        if abs(z.imag) < 1e-10 and 0 < z.real < 1
    ]
    values = poly(points)
    return float(min(values)), float(max(values))


def measure(poly):
    roots = [
        float(z.real) for z in poly.roots() if abs(z.imag) < 1e-10 and 0 < z.real < 1
    ]
    points = sorted([0.0, 1.0, *roots])
    return sum(b - a for a, b in zip(points, points[1:]) if poly((a + b) / 2) < 0)


def interval_cases():
    for degree in (3, 4):
        for h in (1.0, 0.1, 0.01):
            for center in (0.17, 0.31, 0.47, 0.63, 0.81):
                for radius in (0.015, 0.05, 0.12):
                    for shape in (-1.5, 0.6, 1.5):
                        for pocket in (True, False):
                            factor = (
                                Polynomial([1 - shape / 2, shape])
                                if degree == 3
                                else 1
                                + shape * Polynomial([-0.5, 1])
                                + Polynomial([-0.5, 1]) ** 2
                            )
                            poly = (
                                h**degree
                                * factor
                                * (
                                    Polynomial([-center, 1]) ** 2
                                    + (-1 if pocket else 1) * radius**2
                                )
                            )
                            yield (
                                dict(
                                    degree=degree,
                                    h=h,
                                    center=center,
                                    radius=radius,
                                    shape=shape,
                                    pocket=pocket,
                                ),
                                poly,
                            )
    for h in (1.0, 0.1, 0.01):
        for width in (0.01, 0.04, 0.1):
            poly = h**4 * Polynomial.fromroots(
                [0.21 - width, 0.21 + width, 0.73 - width, 0.73 + width]
            )
            yield (
                dict(degree=4, h=h, width=width, pocket=True, family="two_pockets"),
                poly,
            )


def run_intervals():
    rows = []
    for parameters, poly in interval_cases():

        def model(x):
            return np.array([[poly(x)]], complex)

        values, vectors = np.linalg.eigh(np.array([model(0), model(1)]))
        old = certify_simplex(values, vectors, linearization_error_bound=0)
        q2 = Polynomial.fit([0, 0.5, 1], poly([0, 0.5, 1]), 2).convert()
        defect_min, defect_max = extrema(poly - q2)
        exact_defect = max(abs(defect_min), abs(defect_max))
        exact_charge = measure(poly)
        lo, hi = extrema(poly)
        row = dict(
            **parameters,
            exact_charge=exact_charge,
            minimum=lo,
            maximum=hi,
            exact_remainder=exact_defect,
            exact_gapped=lo > 0 or hi < 0,
            affine_gapped=old.status is CertificateStatus.CertifiedGapped,
        )
        for depth in (2, 6):
            enclosure = SpectralMesh(model, root_level=0).occupation_enclosures(
                mu=0, depth=depth
            )[0]
            row[f"depth{depth}"] = dict(
                gapped=enclosure.fixed_occupation,
                lower=enclosure.charge_lower,
                upper=enclosure.charge_upper,
                sampled_remainder=enclosure.interpolation_error,
                remainder_covers=enclosure.interpolation_error >= exact_defect - 1e-14,
                charge_covered=enclosure.charge_lower - 1e-12
                <= exact_charge
                <= enclosure.charge_upper + 1e-12,
            )
        if row["exact_gapped"]:
            # Temporary subdivision cannot lower eta. Measure the persistent
            # refinement needed to establish these deliberately narrow gaps.
            mesh = SpectralMesh(model, root_level=0)
            historical = (
                {"method": "quadratic"}
                if "method" in inspect.signature(mesh.integrate_charge).parameters
                else {}
            )
            result = mesh.integrate_charge(
                mu=0, target_error=1e-12, max_refinements=2000, **historical
            )
            row["adaptive_gap"] = dict(
                charge=result.value,
                error=result.stopping_error,
                vertices=mesh.active_vertices,
                refinements=result.stats.refinements,
                inconclusive=result.inconclusive_simplices,
                all_gapped=all(
                    e.fixed_occupation for e in mesh.occupation_enclosures(mu=0)
                ),
            )
        rows.append(row)
    return rows


def run_bubbles():
    rows = []
    # First root triangle has lambda=(1-x,x-y,y). All edge probes vanish.
    # The quartic also vanishes at its centroid; its value at lambda=(.2,.5,.3)
    # is -0.009. A positive offset .001 gives a provable interior crossing.
    for degree in (3, 4):
        for h in (1.0, 0.1, 0.01):
            for offset in (0.0001, 0.001, 0.004):

                def model(x, y):
                    bubble = y * (1 - x) * (x - y)
                    factor = -1 if degree == 3 else 1 - 2 * x + y
                    return np.array([[h**degree * (offset + bubble * factor)]], complex)

                row = dict(
                    degree=degree,
                    h=h,
                    offset=offset,
                    exact_gapped=False,
                    negative_witness=float(model(0.8, 0.3)[0, 0].real),
                )
                assert row["negative_witness"] < 0 < model(0, 0)[0, 0].real
                for depth in (2, 6):
                    e = SpectralMesh(model, root_level=0).occupation_enclosures(
                        mu=0, depth=depth
                    )[0]
                    row[f"depth{depth}"] = dict(
                        gapped=e.fixed_occupation,
                        sampled_remainder=e.interpolation_error,
                        lower=e.charge_lower,
                        upper=e.charge_upper,
                    )
                rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(intervals=run_intervals(), triangles=run_bubbles())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for family, rows in result.items():
        for degree in (3, 4):
            selected = [r for r in rows if r["degree"] == degree]
            print(
                family,
                degree,
                len(selected),
                {
                    f"false_depth{d}": sum(
                        r[f"depth{d}"]["gapped"] and not r["exact_gapped"]
                        for r in selected
                    )
                    for d in (2, 6)
                },
            )
