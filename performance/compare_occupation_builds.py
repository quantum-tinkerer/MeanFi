"""Compare separate builds, alternating their order to reduce timing drift.

The baseline package must be built at FermiSimplex 98c5009 (which contains both
historical implementations). The current package exposes one algorithm.
Run this driver without concurrent builds or tests. Each worker pins one CPU,
uses one BLAS thread, and excludes import/model construction from its timing.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def compare(args):
    variants = [
        ("legacy", args.baseline),
        ("previous", args.baseline),
        ("current", args.current),
    ]
    cases = [
        ("spectators", n, rep)
        for n in (4, 12, 24, 36, 64, 96, 128, 192)
        for rep in ("tb", "callable")
    ]
    cases += [("replicas", n, "tb") for n in (4, 12, 36, 96)]
    rows = []
    for family, bands, representation in cases:
        for trial in range(args.rounds):
            order = variants if trial % 2 == 0 else variants[::-1]
            for variant, package in order:
                env = dict(
                    os.environ,
                    PYTHONPATH=str(package.resolve()),
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                )
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("occupation_scaling.py")),
                    "--models",
                    str(args.models.resolve()),
                    "--variant",
                    variant,
                    "--bands",
                    str(bands),
                    "--family",
                    family,
                    "--representation",
                    representation,
                    "--repeats",
                    "3",
                    "--batch-seconds",
                    ".03",
                ]
                try:
                    result = subprocess.run(
                        command,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                    )
                    row = json.loads(result.stdout)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    row = dict(
                        variant=variant,
                        bands=bands,
                        family=family,
                        representation=representation,
                        failure=str(exc),
                        stderr=exc.stderr,
                    )
                row["round"] = trial
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    compare(parser.parse_args())
