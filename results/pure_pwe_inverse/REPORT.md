# Pure-PWE gap-only inverse tests

No ML training or COMSOL spectral fitting. Targets come from saved PWE or an explicit requested width.
Full frozen M=20, normal grid 256, 281 k-points, two bands; no reduced-fidelity search.

Positive saved-target retrieval checks: 61.
Nonpositive signed gaps excluded as width targets: 3.
Retrieval checks are software sanity checks, not independent inverse validation.

|Experiment|Target GHz|L1 um|L2 um|Achieved GHz|Absolute error GHz|Trial evaluations (may reuse cache)|Status|
|---|---:|---:|---:|---:|---:|---:|---|
|Held-out D028|15.426659367|168.008928334|139.307403077|15.447613824|0.020954457|1|matched_new_pwe|
|Requested off-grid gap|20.000000000|168.008928334|115.715378144|19.994493703|0.005506297|1|matched_new_pwe|

Elapsed seconds: 188.169.

One solution per target; geometry recovery is not required for a nonunique scalar inverse.
Only one held-out design tested, not a full LODO benchmark.
Matching tolerance is not a bound on physical accuracy. New geometries have PWE-only verification.
Gaps are sampled-path gaps, not certified full-Brillouin-zone gaps.
Existing learned-engine parameter-recovery acceptance gates are not certified by this experiment.
