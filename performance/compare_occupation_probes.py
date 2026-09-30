"""Alternate separate builds for probe-degree and vertex-evaluation comparisons.

Run without concurrent builds/tests. Workers pin one CPU and one BLAS thread.
No option in this driver changes the installed production occupation algorithm.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def compare(args):
    rows = []
    directory = Path(__file__).parent
    if args.experiment == "probes":
        builds = [("cubic", args.comparison), ("quartic", args.quartic)]
        # Eight 1D/2D models at two targets, a 192-band model, and two 3D
        # models at three targets, plus four coupled matrix sizes at two targets.
        # occupation_probe_tradeoff.cases defines them.
        cases = args.case_indices if args.case_indices is not None else list(range(31))
    else:
        builds = [("reconstruct", args.comparison), ("evaluate", args.quartic)]
        cases = [(n, "tb") for n in (12, 36, 96, 192, 384, 768)]
        cases += [(n, rep) for rep in ("callable", "callable_dense") for n in (36, 192)]
    if args.comparison_label:
        builds[0] = (args.comparison_label, builds[0][1])
    if args.quartic_label:
        builds[1] = (args.quartic_label, builds[1][1])
    with TemporaryDirectory() as temporary:
        worker_output = Path(temporary) / "worker.json"
        for index, case in enumerate(cases):
            for trial in range(args.rounds):
                order = builds if (index + trial) % 2 == 0 else builds[::-1]
                for label, package in order:
                    env = dict(
                        os.environ,
                        PYTHONPATH=str(package.resolve()),
                        OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1",
                    )
                    if args.experiment == "probes":
                        command = [
                            sys.executable,
                            str(directory / "occupation_probe_tradeoff.py"),
                            "--case-index",
                            str(case),
                            "--output",
                            str(worker_output),
                            "--root-level",
                            str(args.root_level),
                        ]
                    else:
                        n, representation = case
                        command = [
                            sys.executable,
                            str(directory / "occupation_scaling.py"),
                            "--variant",
                            "current",
                            "--bands",
                            str(n),
                            "--representation",
                            representation,
                            "--batch-seconds",
                            ".06",
                        ]
                    command += [
                        "--models",
                        str(args.models.resolve()),
                        "--repeats",
                        str(args.repeats),
                    ]
                    result = subprocess.run(
                        command,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=180,
                    )
                    if args.experiment == "probes":
                        result = json.loads(worker_output.read_text())
                        row = result["charge"][0]
                        row["hidden_quartic_pockets"] = result["hidden_quartic_pockets"]
                        row["cpu"] = result["cpu"]
                    else:
                        row = json.loads(result.stdout)
                    row.update(build=label, round=trial)
                    rows.append(row)
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
                    print(
                        label,
                        case,
                        trial,
                        row.get("seconds", row.get("failure")),
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("probes", "evaluation"), required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="cubic build for probes; 9f06d36 build for evaluation",
    )
    parser.add_argument("--quartic", type=Path, required=True)
    parser.add_argument("--comparison-label")
    parser.add_argument("--quartic-label")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-indices", type=int, nargs="+")
    parser.add_argument("--root-level", type=int, default=2)
    compare(parser.parse_args())
