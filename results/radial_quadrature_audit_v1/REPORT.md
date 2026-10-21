# Direct radial-projector quadrature audit

Separate geometry-only direct-projector-quadrature audit. No COMSOL labels, fitting, optimizer or production replacement. Five-point gaps are proxies; full paths are sampled-path gaps, not Brillouin-zone certificates.

Completed jobs: 61; protected files verified: 326.
Predeclared spectral angular order: 128.
Projector length derivatives vanish by construction. Finite-basis derivative checks are not continuum certificates.
Gap derivatives use active sampled edges; inspect uniqueness, mode separation and cutoff stability.

|Scope|Design|M change|Frequency relative change|Gap change GHz|Gradient relative change|
|---|---|---|---|---|---|
|base|0|5 to 9|0.00139171|-0.112818|0.0293982|
|base|0|9 to 12|0.000216343|0.0529371|0.00877162|
|base|0|12 to 15|0.000229505|-0.00741864|0.00609854|
|base|0|15 to 20|7.63274e-05|-0.00549085|0.00496455|
|base|0|20 to 25|6.95603e-05|-0.00126138|0.00631916|
|full|0|20 to 25|6.95616e-05|-0.00126138|0.00214419|
|base|1|5 to 9|0.00190924|-0.0956373|0.0337677|
|base|1|9 to 12|0.000393545|0.0563885|0.0105478|
|base|1|12 to 15|0.000137936|-0.0327795|0.00736311|
|base|1|15 to 20|0.000172395|0.0118729|0.00797134|
|base|1|20 to 25|4.44394e-05|0.00193954|0.00290319|
|full|1|20 to 25|5.13806e-05|0.00193954|0.00196098|
|base|2|5 to 9|0.00244309|0.140652|0.0260185|
|base|2|9 to 12|0.000441339|-0.0700709|0.0120629|
|base|2|12 to 15|0.000214283|-0.00209474|0.00963123|
|base|2|15 to 20|0.000151888|0.014857|0.0081358|
|base|2|20 to 25|0.000107781|-0.0112914|0.00401554|
|full|2|20 to 25|0.000112473|-0.0112914|0.00401554|

Detailed quadrature, derivative and angular-frequency diagnostics are in results.json and cases/.
