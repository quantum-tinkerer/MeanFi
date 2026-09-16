# Benchmarks

One script measures the core operations through the public API:

- Selected density entries with FermiSimplex, dense UniformGrid and optional sparse RationalFOE.
- A complete EDIIS solve at zero and finite temperature.

```bash
pixi run benchmark
pixi run -e test-sparse benchmark --sparse --sizes 100 200 --output build/benchmarks.json
```

Each case has one warm-up and three timed repetitions, with one BLAS thread.
Use `--repeat` to change the repetitions. The table reports median wall time,
the largest density-entry error and SCF iteration count. JSON also records every
timing, numerical estimates and the software environment. Density work counters
refer to the measured density call, or the final density evaluation in SCF.
Reports belong in ignored `build/` or CI artifacts; no generated results are tracked.

Density uses a coupled-orbital chain, `H(k) = A - 2 cos(k) I`. The reference
diagonalizes `A` once: zero-temperature occupations are analytic; thermal
occupations use 2048 points, checked against 1024 to within `1e-12`. These
references are computed outside the timings. All backends request the same
onsite diagonal and neighboring off-diagonal entries. UniformGrid uses 64 points;
FermiSimplex refines with the default tolerance policy. Use `--initial-nk 64` for
a separate starting-mesh comparison; it does not change the tolerance policy.
The reported integration estimate is an indicator, not a rigorous error bound;
compare it with the measured error in the JSON report.

SCF uses a half-filled spinful chain with `U=4`, whose translation-invariant
paramagnetic solution has onsite density `I/2`. Both solves use the default EDIIS
and error policy, starting from the same deterministic random correction.

These are reproducible baseline workloads, not a claim that one backend is
fastest for every Hamiltonian. Compare the same sizes, thread settings and
hardware, and inspect accuracy alongside runtime. For a detailed profile, run
the script with Python's standard `cProfile` tool; the numerical test suite
contains the broader normal/BdG accuracy checks.
