Benchmark accuracy and timing tables

244 runs: 201 qualified, 1 measured accuracy failures, 42 unresolved or incomplete.

Overall targets are max selected density-entry error ≤ ε and physical filling error ≤ ε per supercell. Root residual and charge-integration budgets are each ε/4; strict physical filling results are separate. Rankings include only empirically qualified runs. Reference changes are practical uncertainty indicators, not rigorous integration bounds.

| Model | T | Density target | Filling target | Fastest qualified configuration | Seconds | Second / fastest | Strict filling winner |
|---|---:|---:|---:|---|---:|---:|---|
| alias1 | 0.002 | 1e-05 | 1e-05 | gk21-derivative | 0.118 | 1.02 | gk21-derivative |
| alias1 | 0.05 | 1e-05 | 1e-05 | periodic (max_n=8192) | 0.0124 | 3.06 | periodic (max_n=8192) |
| anisotropic48 | 0.01 | 1e-05 | 1e-05 | periodic-anisotropic (max_ref=30) | 1.56 | 23.7 | unresolved |
| anisotropic48 | 0.05 | 1e-07 | 1e-07 | periodic-anisotropic (max_ref=30) | 1.08 | 6.97 | periodic-anisotropic (max_ref=30) |
| anisotropic48 | 0.05 | 1e-05 | 1e-05 | periodic-anisotropic (max_ref=30) | 0.388 | 6.84 | periodic-anisotropic (max_ref=30) |
| anisotropic48 | 0.05 | 0.001 | 0.001 | gauss-bounded | 1.69 | 1 | gauss-bounded |
| anisotropic48 | 0.2 | 1e-05 | 1e-05 | periodic | 0.168 | 2.51 | periodic |
| bdg_chain48 | 0.05 | 1e-05 | 1e-05 | gauss | 0.125 | 1.15 | gauss |
| bdg_gapped48 | 0.05 | 1e-07 | 1e-07 | periodic | 7.68 | 3.99 | periodic |
| bdg_gapped48 | 0.05 | 1e-05 | 1e-05 | gauss | 5.84 | 1.24 | gauss |
| bdg_gapped48 | 0.05 | 0.001 | 0.001 | periodic | 1.84 | 1.12 | periodic |
| bdg_nodal48 | 0.01 | 1e-05 | 1e-05 | periodic-anisotropic (max_ref=30) | 40.7 | 1.45 | periodic-anisotropic (max_ref=30) |
| bdg_nodal48 | 0.05 | 1e-05 | 1e-05 | periodic-anisotropic (max_ref=30) | 4.93 | 1.36 | periodic-anisotropic (max_ref=30) |
| bdg_nodal48 | 0.2 | 1e-05 | 1e-05 | periodic | 1.83 | 2.41 | periodic |
| chain1 | 0.05 | 1e-05 | 1e-05 | gm-derivative | 0.00646 | 1.84 | gm-derivative |
| chain48 | 0.01 | 1e-05 | 1e-05 | periodic | 0.0269 | 1.39 | periodic |
| chain48 | 0.05 | 1e-05 | 1e-05 | periodic (nk=2) | 0.00881 | 1.15 | periodic (nk=2) |
| chain48 | 0.2 | 1e-05 | 1e-05 | gk21-derivative | 0.0103 | 1.25 | gk21-derivative |
| cubic48 | 0.05 | 1e-07 | 1e-07 | unresolved | — | — | unresolved |
| cubic48 | 0.05 | 1e-05 | 1e-05 | periodic-eigenvalue-priority | 218 | — | periodic-eigenvalue-priority |
| cubic48 | 0.05 | 1e-05 | 0.001 | unresolved | — | — | unresolved |
| cubic48 | 0.05 | 0.001 | 0.001 | periodic | 28.4 | — | periodic |
| gapped48 | 0.05 | 1e-05 | 1e-05 | periodic (nk=2) | 0.0148 | 2.33 | periodic (nk=2) |
| pocket1 | 0.002 | 1e-05 | 1e-05 | gk21-derivative | 0.325 | 4.08 | gk21-derivative |
| pocket1 | 0.05 | 1e-05 | 1e-05 | periodic | 0.00849 | 1.98 | periodic |
| square1 | 0.002 | 1e-05 | 1e-05 | gauss-bounded (max_n=4096, max_points=16777216) | 3.68 | 1.97 | gauss-bounded (max_n=4096, max_points=16777216) |
| square1 | 0.05 | 1e-05 | 1e-05 | gk21-derivative | 0.149 | 1.05 | gk21-derivative |
| square48 | 0.01 | 1e-05 | 1e-05 | periodic | 12.3 | 2.68 | periodic |
| square48 | 0.05 | 1e-07 | 1e-07 | periodic | 3.05 | 2.32 | periodic |
| square48 | 0.05 | 1e-05 | 1e-05 | periodic-anisotropic (max_ref=30) | 1.53 | 1.11 | periodic-anisotropic (max_ref=30) |
| square48 | 0.05 | 1e-05 | 0.001 | periodic (filling_target=0.001) | 0.677 | 1.36 | periodic (filling_target=0.001) |
| square48 | 0.05 | 0.001 | 0.001 | periodic | 0.67 | 1.79 | periodic |
| square48 | 0.2 | 1e-05 | 1e-05 | gk21-derivative | 0.16 | 1.04 | gk21-derivative |
| vanhove48 | 0.05 | 1e-05 | 1e-05 | periodic | 0.66 | 2.58 | periodic |
| wire48 | 0.01 | 1e-05 | 1e-05 | gk21-derivative (max_n=4096) | 1.43 | 1 | gk21-derivative (max_n=4096) |
| wire48 | 0.05 | 1e-05 | 1e-05 | gk21-derivative (max_n=4096) | 0.285 | 1.22 | gk21-derivative (max_n=4096) |
| wire48 | 0.2 | 1e-05 | 1e-05 | periodic (max_n=4096) | 0.101 | 1.02 | periodic (max_n=4096) |

All runs

| Model | T | ε | Configuration | Execution / accuracy | Seconds | ρ error | Charge error | Diagonalizations | LAPACK ratio |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| alias1 | 0.002 | 1e-05 | gauss | error / unknown | 0.013 | — | — | 1016 | 100 |
| alias1 | 0.002 | 1e-05 | gk21-derivative | success / pass | 0.118 | 1.72e-11 | 1.75e-11 | 18123 | 51 |
| alias1 | 0.002 | 1e-05 | gm-derivative | success / pass | 0.121 | 1.72e-11 | 1.75e-11 | 18123 | 52.3 |
| alias1 | 0.002 | 1e-05 | periodic | error / unknown | 0.0182 | — | — | 528 | 271 |
| alias1 | 0.002 | 1e-05 | periodic (max_n=65536) | success / pass | 0.168 | 2.42e-06 | 2.42e-06 | 131088 | 10 |
| alias1 | 0.05 | 1e-05 | gauss | error / unknown | 0.0133 | — | — | 1016 | 102 |
| alias1 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.0382 | 2.22e-11 | 2.22e-11 | 5691 | 52.6 |
| alias1 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.038 | 2.22e-11 | 2.22e-11 | 5691 | 52.2 |
| alias1 | 0.05 | 1e-05 | periodic | error / unknown | 0.0163 | — | — | 528 | 241 |
| alias1 | 0.05 | 1e-05 | periodic (max_n=8192) | success / pass | 0.0124 | 2.07e-06 | 2.07e-06 | 4112 | 23.6 |
| alias1 | 0.05 | 1e-05 | periodic-unvalidated | success / fail | 0.00137 | 0.661 | 0.556 | 16 | 671 |
| anisotropic48 | 0 | 0.0001 | fermi (max_ref=100000) | success / unknown | 8.16 | — | — | — | — |
| anisotropic48 | 0.01 | 1e-05 | gauss-bounded | timeout / unknown | 120 | — | — | 587840 | 1.41 |
| anisotropic48 | 0.01 | 1e-05 | gk21-derivative | memory_limit / unknown | 70.2 | — | — | — | — |
| anisotropic48 | 0.01 | 1e-05 | gk21-lean | success / pass | 60.8 | 6.71e-09 | 2.61e-07 | 351918 | 1.33 |
| anisotropic48 | 0.01 | 1e-05 | gm-derivative | success / pass | 44.3 | 7.13e-09 | 2.58e-07 | 141117 | 1.6 |
| anisotropic48 | 0.01 | 1e-05 | gm-lean | success / pass | 36.9 | 6.98e-09 | 2.58e-07 | 214098 | 1.32 |
| anisotropic48 | 0.01 | 1e-05 | periodic | success / pass | 57.5 | 7.02e-09 | 2.73e-07 | 259712 | 1.5 |
| anisotropic48 | 0.01 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 1.56 | 7.02e-09 | 2.73e-07 | 4096 | 1.94 |
| anisotropic48 | 0.01 | 1e-05 | periodic-eigenvalue-priority | success / pass | 47.8 | 7.02e-09 | 2.73e-07 | 184896 | 1.53 |
| anisotropic48 | 0.05 | 1e-07 | gauss-bounded | success / pass | 7.52 | 1.47e-13 | 5.56e-12 | 23872 | 1.68 |
| anisotropic48 | 0.05 | 1e-07 | gk21-derivative | success / pass | 10.3 | 6.17e-11 | 2.38e-09 | 32193 | 1.63 |
| anisotropic48 | 0.05 | 1e-07 | gm-derivative | success / pass | 19.7 | 6.17e-11 | 2.38e-09 | 61149 | 1.64 |
| anisotropic48 | 0.05 | 1e-07 | periodic | success / pass | 10.8 | 1.5e-15 | 2.13e-14 | 34880 | 1.63 |
| anisotropic48 | 0.05 | 1e-07 | periodic-anisotropic (max_ref=30) | success / pass | 1.08 | 1.28e-15 | 2.13e-14 | 2048 | 2.69 |
| anisotropic48 | 0.05 | 1e-05 | gauss | success / pass | 6.82 | 1.47e-13 | 5.55e-12 | 21824 | 1.59 |
| anisotropic48 | 0.05 | 1e-05 | gauss-bounded | success / pass | 7.78 | 1.47e-13 | 5.55e-12 | 23872 | 1.74 |
| anisotropic48 | 0.05 | 1e-05 | gk21 | success / pass | 10.2 | 6.17e-11 | 2.38e-09 | 32193 | 1.61 |
| anisotropic48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 3.04 | 2.19e-08 | 8.42e-07 | 9261 | 1.67 |
| anisotropic48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 3.63 | 6.17e-11 | 2.38e-09 | 9261 | 2 |
| anisotropic48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 2.95 | 6.17e-11 | 2.38e-09 | 9261 | 1.63 |
| anisotropic48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 2.95 | 6.17e-11 | 2.38e-09 | 9261 | 1.63 |
| anisotropic48 | 0.05 | 1e-05 | gk21-fresh | success / pass | 10.8 | 6.17e-11 | 2.38e-09 | 41454 | 1.33 |
| anisotropic48 | 0.05 | 1e-05 | gk21-lean | success / pass | 4.05 | 6.17e-11 | 2.38e-09 | 18522 | 1.5 |
| anisotropic48 | 0.05 | 1e-05 | gm | success / pass | 18.6 | 6.17e-11 | 2.39e-09 | 52649 | 1.8 |
| anisotropic48 | 0.05 | 1e-05 | gm-charge-only | success / pass | 5.37 | 2.2e-08 | 8.5e-07 | 16745 | 1.64 |
| anisotropic48 | 0.05 | 1e-05 | gm-derivative | success / pass | 4.57 | 4.16e-10 | 6.2e-09 | 14229 | 1.64 |
| anisotropic48 | 0.05 | 1e-05 | gm-derivative | success / pass | 4.56 | 4.16e-10 | 6.2e-09 | 14229 | 1.63 |
| anisotropic48 | 0.05 | 1e-05 | gm-derivative | success / pass | 4.6 | 4.16e-10 | 6.2e-09 | 14229 | 1.65 |
| anisotropic48 | 0.05 | 1e-05 | gm-fresh | success / pass | 14.8 | 9.72e-08 | 2.39e-09 | 55386 | 1.37 |
| anisotropic48 | 0.05 | 1e-05 | gm-lean | success / pass | 4.34 | 1.38e-09 | 6.2e-09 | 24038 | 1.32 |
| anisotropic48 | 0.05 | 1e-05 | periodic | success / pass | 2.65 | 2.13e-09 | 8.21e-08 | 8192 | 1.65 |
| anisotropic48 | 0.05 | 1e-05 | periodic | success / pass | 2.65 | 2.13e-09 | 8.21e-08 | 8192 | 1.65 |
| anisotropic48 | 0.05 | 1e-05 | periodic | success / pass | 2.65 | 2.13e-09 | 8.21e-08 | 8192 | 1.65 |
| anisotropic48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 0.387 | 2.13e-09 | 8.21e-08 | 1024 | 1.93 |
| anisotropic48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 0.388 | 2.13e-09 | 8.21e-08 | 1024 | 1.93 |
| anisotropic48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 0.4 | 2.13e-09 | 8.21e-08 | 1024 | 1.99 |
| anisotropic48 | 0.05 | 0.001 | gauss-bounded | success / pass | 1.69 | 1.01e-07 | 3.88e-06 | 5440 | 1.59 |
| anisotropic48 | 0.05 | 0.001 | gk21-derivative | success / pass | 2.96 | 6.17e-11 | 2.38e-09 | 9261 | 1.63 |
| anisotropic48 | 0.05 | 0.001 | gm-derivative | success / pass | 1.7 | 7.52e-08 | 1.91e-06 | 3349 | 2.59 |
| anisotropic48 | 0.05 | 0.001 | periodic | success / pass | 2.66 | 2.14e-09 | 8.21e-08 | 8192 | 1.65 |
| anisotropic48 | 0.2 | 1e-05 | gauss-bounded | success / pass | 0.42 | 1.58e-11 | 6.18e-10 | 1344 | 1.6 |
| anisotropic48 | 0.2 | 1e-05 | gk21-derivative | success / pass | 0.727 | 7.62e-14 | 3.05e-12 | 2205 | 1.68 |
| anisotropic48 | 0.2 | 1e-05 | gm-derivative | success / pass | 0.483 | 1.56e-11 | 4.64e-11 | 1445 | 1.7 |
| anisotropic48 | 0.2 | 1e-05 | periodic | success / pass | 0.168 | 5.84e-09 | 2.28e-07 | 512 | 1.67 |
| bdg_chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.125 | 5.06e-09 | 2.43e-07 | 104 | 1.29 |
| bdg_chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.125 | 5.06e-09 | 2.43e-07 | 104 | 1.28 |
| bdg_chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.125 | 5.06e-09 | 2.43e-07 | 104 | 1.28 |
| bdg_chain48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 0.19 | 5.06e-09 | 2.43e-07 | 168 | 1.21 |
| bdg_chain48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 0.193 | 5.06e-09 | 2.43e-07 | 168 | 1.23 |
| bdg_chain48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 0.19 | 5.06e-09 | 2.43e-07 | 168 | 1.21 |
| bdg_chain48 | 0.05 | 1e-05 | gm-charge-only | success / pass | 0.19 | 5.06e-09 | 2.43e-07 | 168 | 1.21 |
| bdg_chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.146 | 5.06e-09 | 2.43e-07 | 120 | 1.3 |
| bdg_chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.144 | 5.06e-09 | 2.43e-07 | 120 | 1.28 |
| bdg_chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.144 | 5.06e-09 | 2.43e-07 | 120 | 1.28 |
| bdg_gapped48 | 0.05 | 1e-07 | gk21-charge-only | success / pass | 30.7 | 1.11e-16 | 3.55e-15 | 25578 | 1.29 |
| bdg_gapped48 | 0.05 | 1e-07 | gm-charge-only | timeout / unknown | 90.1 | — | — | — | — |
| bdg_gapped48 | 0.05 | 1e-07 | periodic | success / pass | 7.68 | 4.12e-13 | 1.98e-11 | 6016 | 1.37 |
| bdg_gapped48 | 0.05 | 1e-05 | gauss | success / pass | 5.84 | 6.13e-09 | 2.94e-07 | 4992 | 1.26 |
| bdg_gapped48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 16 | 1.04e-09 | 4.98e-08 | 14112 | 1.22 |
| bdg_gapped48 | 0.05 | 1e-05 | gm-charge-only | success / pass | 16.4 | 1.04e-09 | 4.9e-08 | 14365 | 1.23 |
| bdg_gapped48 | 0.05 | 1e-05 | periodic | success / pass | 7.23 | 4.12e-13 | 1.98e-11 | 5952 | 1.3 |
| bdg_gapped48 | 0.05 | 0.001 | gk21-charge-only | success / pass | 4.43 | 3.02e-06 | 0.000145 | 3969 | 1.2 |
| bdg_gapped48 | 0.05 | 0.001 | gm-charge-only | success / pass | 2.07 | 3.02e-06 | 0.000145 | 1768 | 1.26 |
| bdg_gapped48 | 0.05 | 0.001 | periodic | success / pass | 1.84 | 3.08e-06 | 0.000148 | 1536 | 1.29 |
| bdg_nodal48 | 0.01 | 1e-05 | gauss-bounded | success / pass | 112 | 8.71e-10 | 4.18e-08 | 98688 | 1.22 |
| bdg_nodal48 | 0.01 | 1e-05 | gk21-charge-only | timeout / unknown | 90.1 | — | — | — | — |
| bdg_nodal48 | 0.01 | 1e-05 | gk21-derivative | success / pass | 109 | 9.37e-13 | 4.52e-11 | 68355 | 1.72 |
| bdg_nodal48 | 0.01 | 1e-05 | gm-charge-only | timeout / unknown | 90.1 | — | — | — | — |
| bdg_nodal48 | 0.01 | 1e-05 | gm-derivative | success / pass | 59 | 2.54e-11 | 1.19e-09 | 39168 | 1.62 |
| bdg_nodal48 | 0.01 | 1e-05 | periodic | timeout / unknown | 90 | — | — | — | — |
| bdg_nodal48 | 0.01 | 1e-05 | periodic | success / pass | 128 | 3.42e-11 | 1.64e-09 | 106432 | 1.29 |
| bdg_nodal48 | 0.01 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 40.7 | 1.81e-09 | 8.69e-08 | 33178 | 1.32 |
| bdg_nodal48 | 0.05 | 1e-05 | gauss | success / pass | 6.71 | 3.5e-09 | 1.68e-07 | 4928 | 1.47 |
| bdg_nodal48 | 0.05 | 1e-05 | gk21 | success / pass | 46 | 1.46e-14 | 7.14e-13 | 29106 | 1.7 |
| bdg_nodal48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 43.2 | 8.22e-09 | 3.94e-07 | 37485 | 1.24 |
| bdg_nodal48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 17.5 | 1.47e-14 | 7.46e-13 | 11907 | 1.59 |
| bdg_nodal48 | 0.05 | 1e-05 | gk21-fresh | success / pass | 36.2 | 5.74e-11 | 7.14e-13 | 23814 | 1.64 |
| bdg_nodal48 | 0.05 | 1e-05 | gk21-lean | success / pass | 36.4 | 8.22e-09 | 3.94e-07 | 33957 | 1.15 |
| bdg_nodal48 | 0.05 | 1e-05 | gm | timeout / unknown | 90.1 | — | — | — | — |
| bdg_nodal48 | 0.05 | 1e-05 | gm-charge-only | success / pass | 30.3 | 8.22e-09 | 3.94e-07 | 26333 | 1.24 |
| bdg_nodal48 | 0.05 | 1e-05 | gm-derivative | success / pass | 34 | 4.57e-11 | 1.69e-09 | 23171 | 1.58 |
| bdg_nodal48 | 0.05 | 1e-05 | gm-fresh | timeout / unknown | 90.1 | — | — | — | — |
| bdg_nodal48 | 0.05 | 1e-05 | gm-lean | success / pass | 31.5 | 8.29e-09 | 3.94e-07 | 27047 | 1.25 |
| bdg_nodal48 | 0.05 | 1e-05 | periodic | success / pass | 7.9 | 6.16e-12 | 2.96e-10 | 5888 | 1.44 |
| bdg_nodal48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 4.93 | 3.29e-08 | 1.58e-06 | 3968 | 1.34 |
| bdg_nodal48 | 0.2 | 1e-05 | gk21-charge-only | success / pass | 4.41 | 1.79e-08 | 8.57e-07 | 3969 | 1.2 |
| bdg_nodal48 | 0.2 | 1e-05 | gm-charge-only | success / pass | 12.8 | 1.79e-08 | 8.57e-07 | 11135 | 1.24 |
| bdg_nodal48 | 0.2 | 1e-05 | periodic | success / pass | 1.83 | 1.79e-08 | 8.57e-07 | 1536 | 1.28 |
| chain1 | 0.05 | 1e-05 | gauss | success / pass | 0.0255 | 2.2e-06 | 2.2e-06 | 1016 | 232 |
| chain1 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.0125 | 2.71e-11 | 2.71e-11 | 609 | 190 |
| chain1 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.00646 | 2.71e-11 | 2.71e-11 | 609 | 98 |
| chain1 | 0.05 | 1e-05 | periodic | success / pass | 0.0119 | 2.46e-08 | 2.46e-08 | 1024 | 107 |
| chain48 | 0 | 0.0001 | fermi (max_ref=100000) | success / unknown | 0.0336 | — | — | — | — |
| chain48 | 0.01 | 1e-05 | gk21-derivative | success / pass | 0.0374 | 9.1e-12 | 1.57e-10 | 105 | 1.81 |
| chain48 | 0.01 | 1e-05 | gm-derivative | success / pass | 0.0375 | 9.1e-12 | 1.57e-10 | 105 | 1.81 |
| chain48 | 0.01 | 1e-05 | periodic | success / pass | 0.0269 | 5.63e-11 | 9.74e-10 | 64 | 2.13 |
| chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.0099 | 1.49e-10 | 3.4e-09 | 24 | 2.09 |
| chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.0191 | 1.49e-10 | 3.4e-09 | 24 | 4.03 |
| chain48 | 0.05 | 1e-05 | gauss | success / pass | 0.0101 | 1.49e-10 | 3.4e-09 | 24 | 2.14 |
| chain48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.0109 | 3.8e-12 | 8.68e-11 | 21 | 2.64 |
| chain48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.011 | 3.8e-12 | 8.68e-11 | 21 | 2.67 |
| chain48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.0202 | 3.8e-12 | 8.68e-11 | 21 | 4.88 |
| chain48 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.0112 | 3.8e-12 | 8.68e-11 | 21 | 2.7 |
| chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.0128 | 3.8e-12 | 8.68e-11 | 32 | 2.03 |
| chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.0146 | 3.8e-12 | 8.68e-11 | 32 | 2.31 |
| chain48 | 0.05 | 1e-05 | periodic | success / pass | 0.0127 | 3.8e-12 | 8.68e-11 | 32 | 2.01 |
| chain48 | 0.05 | 1e-05 | periodic (nk=2) | success / pass | 0.00881 | 6.54e-09 | 1.5e-07 | 16 | 2.8 |
| chain48 | 0.05 | 1e-05 | periodic (nk=4) | success / pass | 0.0139 | 4.64e-09 | 1.06e-07 | 16 | 4.42 |
| chain48 | 0.2 | 1e-05 | gk21-derivative | success / pass | 0.0103 | 6.27e-10 | 2.05e-08 | 21 | 2.48 |
| chain48 | 0.2 | 1e-05 | gm-derivative | success / pass | 0.0185 | 6.27e-10 | 2.05e-08 | 21 | 4.47 |
| chain48 | 0.2 | 1e-05 | periodic | success / pass | 0.0128 | 6.27e-10 | 2.05e-08 | 32 | 2.03 |
| cubic48 | 0.05 | 1e-07 | gk21-derivative | timeout / unknown | 90.1 | — | — | — | — |
| cubic48 | 0.05 | 1e-07 | gm-derivative | memory_limit / unknown | 60.4 | — | — | — | — |
| cubic48 | 0.05 | 1e-07 | periodic | timeout / unknown | 90.1 | — | — | — | — |
| cubic48 | 0.05 | 1e-05 | gauss | error / unknown | 17.7 | — | — | 37376 | 2.38 |
| cubic48 | 0.05 | 1e-05 | gauss-bounded | error / unknown | 131 | — | — | 569088 | 1.53 |
| cubic48 | 0.05 | 1e-05 | gauss-bounded (filling_target=0.001) | timeout / unknown | 120 | — | — | 539904 | 1.5 |
| cubic48 | 0.05 | 1e-05 | gk21-derivative | timeout / unknown | 90.1 | — | — | — | — |
| cubic48 | 0.05 | 1e-05 | gk21-lean | timeout / unknown | 180 | — | — | 1.71298e+06 | 1.07 |
| cubic48 | 0.05 | 1e-05 | gk21-lean (filling_target=0.001) | timeout / unknown | 120 | — | — | 767338 | 1.31 |
| cubic48 | 0.05 | 1e-05 | gm-derivative | memory_limit / unknown | 49.5 | — | — | — | — |
| cubic48 | 0.05 | 1e-05 | gm-lean | error / unknown | 56.5 | — | — | 528033 | 1.1 |
| cubic48 | 0.05 | 1e-05 | periodic | timeout / unknown | 90 | — | — | — | — |
| cubic48 | 0.05 | 1e-05 | periodic | timeout / unknown | 240 | — | — | 1.23776e+06 | 1.43 |
| cubic48 | 0.05 | 1e-05 | periodic (cache_MiB=128) | timeout / unknown | 119 | — | — | 957696 | 1.18 |
| cubic48 | 0.05 | 1e-05 | periodic (filling_target=0.001) | timeout / unknown | 120 | — | — | 872448 | 1.24 |
| cubic48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | timeout / unknown | 180 | — | — | 869296 | 1.53 |
| cubic48 | 0.05 | 1e-05 | periodic-eigenvalue-priority | success / pass | 218 | 1.89e-08 | 7.3e-07 | 793216 | 1.65 |
| cubic48 | 0.05 | 1e-05 | periodic-eigenvalue-priority (cache_MiB=128) | timeout / unknown | 120 | — | — | 532224 | 1.5 |
| cubic48 | 0.05 | 0.001 | gk21-derivative | timeout / unknown | 90.1 | — | — | — | — |
| cubic48 | 0.05 | 0.001 | gk21-lean | timeout / unknown | 120 | — | — | 766570 | 1.31 |
| cubic48 | 0.05 | 0.001 | gm-derivative | memory_limit / unknown | 48.7 | — | — | — | — |
| cubic48 | 0.05 | 0.001 | gm-lean | error / unknown | 56.3 | — | — | 528033 | 1.09 |
| cubic48 | 0.05 | 0.001 | periodic | success / pass | 28.4 | 4.87e-06 | 0.00019 | 105984 | 1.67 |
| gapped48 | 0.05 | 1e-05 | gauss | success / pass | 0.108 | 6.66e-16 | 0 | 320 | 1.72 |
| gapped48 | 0.05 | 1e-05 | gauss | success / pass | 0.107 | 6.66e-16 | 0 | 320 | 1.69 |
| gapped48 | 0.05 | 1e-05 | gauss | success / pass | 0.108 | 6.66e-16 | 0 | 320 | 1.72 |
| gapped48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.156 | 3.33e-16 | 0 | 441 | 1.8 |
| gapped48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.157 | 3.33e-16 | 0 | 441 | 1.81 |
| gapped48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.157 | 3.33e-16 | 0 | 441 | 1.81 |
| gapped48 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.0345 | 7.98e-10 | 0 | 85 | 2.06 |
| gapped48 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.0347 | 7.98e-10 | 0 | 85 | 2.07 |
| gapped48 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.0343 | 7.98e-10 | 0 | 85 | 2.05 |
| gapped48 | 0.05 | 1e-05 | periodic | success / pass | 0.166 | 7.77e-16 | 0 | 512 | 1.64 |
| gapped48 | 0.05 | 1e-05 | periodic | success / pass | 0.167 | 7.77e-16 | 0 | 512 | 1.66 |
| gapped48 | 0.05 | 1e-05 | periodic | success / pass | 0.174 | 7.77e-16 | 0 | 512 | 1.72 |
| gapped48 | 0.05 | 1e-05 | periodic (nk=2) | success / pass | 0.0148 | 6.12e-11 | 0 | 32 | 2.35 |
| gapped48 | 0.05 | 1e-05 | periodic (nk=4) | success / pass | 0.0456 | 3.33e-16 | 0 | 128 | 1.81 |
| pocket1 | 0.002 | 1e-05 | gauss | error / unknown | 0.506 | — | — | 349504 | 11.4 |
| pocket1 | 0.002 | 1e-05 | gauss-bounded (max_n=4096, max_points=16777216) | success / pass | 1.33 | 2.18e-06 | 2.18e-06 | 1.39808e+06 | 7.49 |
| pocket1 | 0.002 | 1e-05 | gk21-derivative | success / pass | 0.325 | 7.06e-09 | 7.51e-09 | 157437 | 16.3 |
| pocket1 | 0.002 | 1e-05 | gm-derivative | error / unknown | 0.353 | — | — | 136017 | 20.5 |
| pocket1 | 0.002 | 1e-05 | gm-derivative (max_ref=32768) | success / pass | 3.1 | 5.81e-07 | 6.78e-07 | 480981 | 50.8 |
| pocket1 | 0.002 | 1e-05 | periodic | error / unknown | 0.337 | — | — | 262144 | 10.1 |
| pocket1 | 0.002 | 1e-05 | periodic (max_n=4096, max_points=16777216) | success / pass | 1.83 | 1.8e-08 | 1.8e-08 | 2.09715e+06 | 6.88 |
| pocket1 | 0.05 | 1e-05 | gauss | success / pass | 0.0325 | 4.04e-07 | 4.04e-07 | 21824 | 11.7 |
| pocket1 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.0336 | 2.22e-09 | 2.22e-09 | 12789 | 20.7 |
| pocket1 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.0168 | 1.59e-08 | 1.24e-08 | 2601 | 51 |
| pocket1 | 0.05 | 1e-05 | periodic | success / pass | 0.00849 | 6.11e-07 | 6.11e-07 | 8192 | 8.17 |
| square1 | 0.002 | 1e-05 | gauss | error / unknown | 0.659 | — | — | 349504 | 15.7 |
| square1 | 0.002 | 1e-05 | gauss-bounded (max_n=4096, max_points=16777216) | success / pass | 3.68 | 1.19e-06 | 1.24e-06 | 5.59238e+06 | 5.49 |
| square1 | 0.002 | 1e-05 | gk21-derivative | success / pass | 7.24 | 1.04e-08 | 1.04e-08 | 2.77521e+06 | 21.8 |
| square1 | 0.002 | 1e-05 | gm-derivative | error / unknown | 0.36 | — | — | 136017 | 22.1 |
| square1 | 0.002 | 1e-05 | gm-derivative (max_ref=32768) | success / pass | 11.3 | 1.02e-08 | 1.08e-08 | 2.01472e+06 | 46.9 |
| square1 | 0.002 | 1e-05 | periodic | error / unknown | 0.374 | — | — | 262144 | 11.9 |
| square1 | 0.002 | 1e-05 | periodic (max_n=4096, max_points=16777216) | success / pass | 9.65 | 2.06e-06 | 2.12e-06 | 8.38861e+06 | 9.61 |
| square1 | 0.05 | 1e-05 | gauss | success / pass | 0.156 | 5.02e-07 | 5.02e-07 | 87360 | 14.9 |
| square1 | 0.05 | 1e-05 | gk21-derivative | success / pass | 0.149 | 2.96e-09 | 4.1e-10 | 58653 | 21.2 |
| square1 | 0.05 | 1e-05 | gm-derivative | success / pass | 0.343 | 2.61e-09 | 3.02e-09 | 58701 | 48.8 |
| square1 | 0.05 | 1e-05 | periodic | success / pass | 0.224 | 2.43e-08 | 2.43e-08 | 131072 | 14.3 |
| square48 | 0 | 0.0001 | fermi (max_ref=100000) | timeout / unknown | 90.1 | — | — | — | — |
| square48 | 0.01 | 1e-05 | gauss-bounded | success / pass | 32.9 | 2.81e-09 | 7.34e-08 | 141376 | 1.47 |
| square48 | 0.01 | 1e-05 | gk21-derivative | success / pass | 35.5 | 2.06e-12 | 1.09e-11 | 92169 | 1.96 |
| square48 | 0.01 | 1e-05 | gm-derivative | success / pass | 58.3 | 1.06e-11 | 2.19e-10 | 175525 | 1.69 |
| square48 | 0.01 | 1e-05 | periodic | success / pass | 12.3 | 1.07e-08 | 2.81e-07 | 34880 | 1.85 |
| square48 | 0.05 | 1e-07 | gauss-bounded | success / pass | 7.09 | 1.09e-14 | 2.98e-13 | 23872 | 1.58 |
| square48 | 0.05 | 1e-07 | gk21-derivative | success / pass | 9.03 | 5.71e-12 | 1.67e-10 | 28665 | 1.6 |
| square48 | 0.05 | 1e-07 | gm-derivative | success / pass | 29.4 | 5.61e-12 | 1.65e-10 | 89505 | 1.67 |
| square48 | 0.05 | 1e-07 | periodic | success / pass | 3.05 | 5.74e-11 | 1.68e-09 | 8192 | 1.9 |
| square48 | 0.05 | 1e-05 | gauss | success / pass | 1.7 | 2.79e-09 | 8.18e-08 | 5440 | 1.59 |
| square48 | 0.05 | 1e-05 | gauss | success / pass | 1.71 | 2.79e-09 | 8.18e-08 | 5440 | 1.6 |
| square48 | 0.05 | 1e-05 | gauss | success / pass | 1.7 | 2.79e-09 | 8.18e-08 | 5440 | 1.6 |
| square48 | 0.05 | 1e-05 | gauss-bounded | success / pass | 1.71 | 2.79e-09 | 8.18e-08 | 5440 | 1.6 |
| square48 | 0.05 | 1e-05 | gauss-bounded (filling_target=0.001) | success / pass | 1.7 | 2.79e-09 | 8.18e-08 | 5440 | 1.59 |
| square48 | 0.05 | 1e-05 | gk21 | success / pass | 6.88 | 5.71e-12 | 1.67e-10 | 21609 | 1.62 |
| square48 | 0.05 | 1e-05 | gk21-charge-only | success / pass | 3.02 | 5.06e-11 | 1.49e-09 | 9261 | 1.66 |
| square48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 3.64 | 5.71e-12 | 1.67e-10 | 9261 | 2 |
| square48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 2.93 | 5.71e-12 | 1.67e-10 | 9261 | 1.61 |
| square48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 2.96 | 5.71e-12 | 1.67e-10 | 9261 | 1.63 |
| square48 | 0.05 | 1e-05 | gk21-fresh | success / pass | 5.88 | 5.76e-10 | 1.67e-10 | 23814 | 1.26 |
| square48 | 0.05 | 1e-05 | gk21-lean | success / pass | 3.85 | 5.71e-12 | 1.67e-10 | 18522 | 1.42 |
| square48 | 0.05 | 1e-05 | gk21-lean (filling_target=0.001) | success / pass | 0.921 | 5.52e-06 | 0.000162 | 4410 | 1.43 |
| square48 | 0.05 | 1e-05 | gm | success / pass | 25.2 | 5.51e-12 | 1.59e-10 | 77197 | 1.66 |
| square48 | 0.05 | 1e-05 | gm-charge-only | success / pass | 7.39 | 1.96e-10 | 5.54e-09 | 21981 | 1.71 |
| square48 | 0.05 | 1e-05 | gm-derivative | success / pass | 7 | 1.85e-10 | 4.51e-09 | 21301 | 1.67 |
| square48 | 0.05 | 1e-05 | gm-derivative | success / pass | 7.03 | 1.85e-10 | 4.51e-09 | 21301 | 1.68 |
| square48 | 0.05 | 1e-05 | gm-derivative | success / pass | 7.87 | 1.85e-10 | 4.51e-09 | 21301 | 1.88 |
| square48 | 0.05 | 1e-05 | gm-fresh | success / pass | 20.8 | 4.73e-08 | 1.59e-10 | 81566 | 1.3 |
| square48 | 0.05 | 1e-05 | gm-lean | success / pass | 7.53 | 3.53e-10 | 4.51e-09 | 39474 | 1.34 |
| square48 | 0.05 | 1e-05 | periodic | success / pass | 2.64 | 5.74e-11 | 1.68e-09 | 8192 | 1.64 |
| square48 | 0.05 | 1e-05 | periodic | success / pass | 2.63 | 5.74e-11 | 1.68e-09 | 8192 | 1.64 |
| square48 | 0.05 | 1e-05 | periodic | success / pass | 2.64 | 5.74e-11 | 1.68e-09 | 8192 | 1.64 |
| square48 | 0.05 | 1e-05 | periodic (filling_target=0.001) | success / pass | 0.677 | 7.06e-07 | 2.07e-05 | 2048 | 1.68 |
| square48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 1.53 | 1.33e-10 | 3.89e-09 | 4096 | 1.91 |
| square48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 1.52 | 1.33e-10 | 3.89e-09 | 4096 | 1.89 |
| square48 | 0.05 | 1e-05 | periodic-anisotropic (max_ref=30) | success / pass | 1.54 | 1.33e-10 | 3.89e-09 | 4096 | 1.91 |
| square48 | 0.05 | 0.001 | gauss-bounded | success / pass | 1.72 | 2.79e-09 | 8.18e-08 | 5440 | 1.61 |
| square48 | 0.05 | 0.001 | gk21-derivative | success / pass | 1.2 | 5.52e-06 | 0.000162 | 2205 | 2.76 |
| square48 | 0.05 | 0.001 | gm-derivative | success / pass | 1.59 | 5.47e-06 | 0.000158 | 4845 | 1.68 |
| square48 | 0.05 | 0.001 | periodic | success / pass | 0.67 | 7.06e-07 | 2.07e-05 | 2048 | 1.67 |
| square48 | 0.2 | 1e-05 | gauss-bounded | success / pass | 0.424 | 7.61e-13 | 2.8e-11 | 1344 | 1.61 |
| square48 | 0.2 | 1e-05 | gk21-derivative | success / pass | 0.16 | 7.17e-14 | 1.14e-12 | 441 | 1.85 |
| square48 | 0.2 | 1e-05 | gm-derivative | success / pass | 0.896 | 9.49e-12 | 2.1e-10 | 2737 | 1.67 |
| square48 | 0.2 | 1e-05 | periodic | success / pass | 0.166 | 5.6e-10 | 2.06e-08 | 512 | 1.65 |
| vanhove48 | 0.05 | 1e-05 | gauss | success / pass | 1.7 | 1.28e-09 | 6.17e-08 | 5440 | 1.61 |
| vanhove48 | 0.05 | 1e-05 | gk21-derivative | success / pass | 2.93 | 3.6e-11 | 1.73e-09 | 9261 | 1.63 |
| vanhove48 | 0.05 | 1e-05 | gm-derivative | success / pass | 7.91 | 3.6e-11 | 8.1e-10 | 24429 | 1.67 |
| vanhove48 | 0.05 | 1e-05 | periodic | success / pass | 0.66 | 1.47e-13 | 7.01e-12 | 2048 | 1.66 |
| wire48 | 0.01 | 1e-05 | gauss-bounded (max_n=4096) | success / pass | 1.44 | 1.78e-09 | 6.32e-08 | 4088 | 1.81 |
| wire48 | 0.01 | 1e-05 | gk21-derivative (max_n=4096) | success / pass | 1.43 | 2.84e-13 | 1.01e-11 | 4683 | 1.57 |
| wire48 | 0.01 | 1e-05 | periodic (max_n=4096) | success / pass | 1.47 | 4.29e-10 | 1.52e-08 | 4096 | 1.85 |
| wire48 | 0.05 | 1e-05 | gauss-bounded (max_n=4096) | success / pass | 0.347 | 3.37e-09 | 1.19e-07 | 1016 | 1.76 |
| wire48 | 0.05 | 1e-05 | gauss-bounded (max_n=4096) | success / pass | 0.348 | 3.37e-09 | 1.19e-07 | 1016 | 1.76 |
| wire48 | 0.05 | 1e-05 | gauss-bounded (max_n=4096) | success / pass | 0.366 | 3.37e-09 | 1.19e-07 | 1016 | 1.86 |
| wire48 | 0.05 | 1e-05 | gk21-derivative (max_n=4096) | success / pass | 0.285 | 4.47e-13 | 1.55e-11 | 903 | 1.62 |
| wire48 | 0.05 | 1e-05 | gk21-derivative (max_n=4096) | success / pass | 0.294 | 4.47e-13 | 1.55e-11 | 903 | 1.68 |
| wire48 | 0.05 | 1e-05 | gk21-derivative (max_n=4096) | success / pass | 0.282 | 4.47e-13 | 1.55e-11 | 903 | 1.61 |
| wire48 | 0.05 | 1e-05 | periodic (max_n=4096) | success / pass | 0.379 | 2.45e-11 | 8.65e-10 | 1024 | 1.9 |
| wire48 | 0.05 | 1e-05 | periodic (max_n=4096) | success / pass | 0.444 | 2.45e-11 | 8.65e-10 | 1024 | 2.23 |
| wire48 | 0.05 | 1e-05 | periodic (max_n=4096) | success / pass | 0.692 | 2.45e-11 | 8.65e-10 | 1024 | 3.48 |
| wire48 | 0.2 | 1e-05 | gauss-bounded (max_n=4096) | success / pass | 0.171 | 7.79e-09 | 2.76e-07 | 248 | 3.56 |
| wire48 | 0.2 | 1e-05 | gk21-derivative (max_n=4096) | success / pass | 0.104 | 1.25e-13 | 4.42e-12 | 315 | 1.7 |
| wire48 | 0.2 | 1e-05 | periodic (max_n=4096) | success / pass | 0.101 | 3.95e-11 | 1.4e-09 | 256 | 2.04 |

LAPACK ratio = total runtime / (full-eigensystem count × fastest tested full LAPACK cost + eigenvalues-only count × fastest tested values-only LAPACK cost). Representative sampled baselines can vary with the actual nodes and μ values. Scalar controls are not evidence about dense-matrix implementation overhead.
