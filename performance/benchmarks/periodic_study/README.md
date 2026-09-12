# Historical periodic integration study

The [report](REPORT.md), [comparison table](results/comparison.csv), convergence
CSVs and environment metadata preserve the earlier 244-job study. The original
experimental implementation is archived in
`results/final-source-snapshot.tar.gz`; it is development evidence and may require
removed APIs and dependencies. It is not imported by MeanFi or included in release
artifacts. The source snapshot contains the original reproduction instructions.

These files were recovered from the sibling development worktree `a93c`: the
starting release checkout did not contain the study or `PeriodicQuadrature`.
Historical method names, source paths and budgets describe that earlier snapshot.
The report does not claim that periodic sampling is universally fastest. It
motivates mandatory shifted-grid validation and distinguishes final mesh size
from cumulative diagonalizations.

For the current two-family API, run the bounded regression harness in
`performance/benchmarks/release_regression.py`. Its results are separate from
these historical timings.
