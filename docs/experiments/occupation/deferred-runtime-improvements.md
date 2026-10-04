# Deferred charge-runtime improvements

Recorded at the user's request on 2026-10-04. The later authorization to prepare the residual-matrix method for production includes the cleanup below. The additional mathematical and subdivision changes remain deferred.

- Implemented during production cleanup: remove obsolete safe-variation calculations and duplicate extraction of active blocks. This preserved mesh, charge and density exactly and reduced difficult-state charge time from 6.28 to 5.90 seconds.
- Keep the quartic transformed Hamiltonian Y in the existing degree-six Schur envelopes instead of replacing Y-P by a scalar remainder.
- Keep the direct Hamiltonian interpolation error in the matrix metric eta(I+X†X), and use Young's matrix inequality to retain the residual perturbation error in this metric. Combined with cleanup, difficult-state charge time was 5.43 seconds with 1914 instead of 2181 triangles.
- Consider internal charge depth one as a separate performance decision. Together with the above changes it gave 4.74 seconds charge and 6.23 seconds charge plus density, compared with 6.28/7.75 seconds. The combination saved 25–46% charge time across six graphene fields but was about 6% slower on a simple 12-band exact model. Do not assume a universal improvement.

Validation: 351 Python tests, 14 native tests per candidate, eight analytic polynomial crossing/pocket cases, and 57,600 direct matrix-envelope checks per mathematical variant. Existing finite-probe remainder assumptions remain. Full tutorial runtime for these newer fixes is unmeasured.

Rejected candidates: replacing scaled spectral margins with positive Gershgorin margins; preliminary inherited-basis sign checks; center-diagonal weights; cached polynomial product coefficients; eliminating all internal subdivision. They did not give enough benefit in controlled tests.

Detailed measurements and temporary sources: /tmp/meanfi-schur-prototype/runtime-candidates/RESULTS.md and recommended-source/ in that directory. The next tolerance study uses the earlier matrix-source prototype, without these fixes.
