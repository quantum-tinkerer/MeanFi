"""Bounded larger-matrix comparison of independently built implementations.

Each worker pins one CPU and one BLAS thread. With fixed one-band physics,
unchanged vertex and refinement counts isolate matrix-size cost. Source model
generation and imports are outside the measured integration time.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    builds = [("legacy", args.legacy, "legacy"), ("ae3ce87", args.before, "current")]
    if args.after:
        builds.append(("optimized", args.after, "current"))
    rows = []
    for band_index, n in enumerate(args.bands):
        for repetition in range(args.rounds):
            offset = band_index % len(builds)
            order = builds[offset:] + builds[:offset]
            if repetition % 2:
                order.reverse()
            for label, package, variant in order:
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
                    str(n),
                    "--representation",
                    args.representation,
                    "--family",
                    args.family,
                    "--repeats",
                    str(args.repeats),
                    "--batch-seconds",
                    ".10",
                ]
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        env=env,
                        check=True,
                        timeout=180,
                    )
                    row = json.loads(result.stdout)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    stderr = exc.stderr
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    row = dict(
                        bands=n,
                        representation=args.representation,
                        family=args.family,
                        failure=str(exc),
                        stderr=stderr,
                    )
                row.update(build=label, round=repetition)
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bands", type=int, nargs="+", default=[96, 192, 384, 768, 1024]
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--representation", choices=("tb", "callable"), default="tb")
    parser.add_argument(
        "--family", choices=("spectators", "replicas"), default="spectators"
    )
    run(parser.parse_args())
