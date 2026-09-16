"""Bounded sparse SCF reconstruction benchmark; run sequentially on one CPU.

Model setup is outside the timed region. Each pass reconstructs active density,
recovers its parameters, and computes the interaction correction. No integration
or eigensolve is involved. Linux peak RSS includes imports and model setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkout", type=Path, default=Path(__file__).resolve().parents[2]
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--size", type=int, default=2000)
parser.add_argument("--repeat", type=int, default=3)
args = parser.parse_args()
sys.path.insert(0, str(args.checkout.resolve()))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from meanfi import Model  # noqa: E402

with threadpool_limits(1):
    size = args.size
    h = {(0,): sparse.eye(size, dtype=complex, format="csr")}
    interaction = {
        (0,): sparse.diags(
            [np.ones(size - 1), np.ones(size - 1)], [-1, 1], format="csr"
        )
    }
    model = Model(h, interaction, filling=size / 2)
    space = model._space if hasattr(model, "_space") else model.scf_space
    decode = (
        space.density_from_params
        if hasattr(space, "density_from_params")
        else space.meanfield_input_from_params
    )
    encode = (
        space.params_from_density
        if hasattr(space, "params_from_density")
        else space.params_from_meanfield_input
    )
    params = np.random.default_rng(42).standard_normal(space.num_params)

    def reconstruct():
        density = decode(params)
        recovered = encode(density)
        if hasattr(model, "mean_field"):
            correction = model.mean_field(density)
        else:
            from meanfi import meanfield  # Historical checkout under --checkout.

            correction = meanfield(density, interaction)
        return recovered, correction

    reconstruct()
    timings = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        recovered, correction = reconstruct()
        timings.append(time.perf_counter() - started)
    np.testing.assert_allclose(recovered, params, atol=1e-14)
    values = space.active_coordinates.values_from_tb(correction)
    output = dict(
        checkout=str(args.checkout.resolve()),
        python=sys.version,
        numpy=np.__version__,
        size=size,
        parameters=space.num_params,
        timings=timings,
        seconds=statistics.median(timings),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        correction_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output), flush=True)
