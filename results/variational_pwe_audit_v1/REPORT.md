# Independent variational Hz PWE audit

Three geometries and five k samples. Gaps are five-point proxies, never full-path or full-Brillouin-zone certificates. Analytic derivative agreement and consecutive-cutoff convergence are separate diagnostics.

Completed checkpointed jobs: 34. Protected files unchanged: 326.
Unchanged gradient tolerance: 0.0001.
No projectors, ML, COMSOL spectra or fitting. Production files/results are unchanged.
Nested-basis eigenvalue monotonicity applies to exact Galerkin assembly; numerical violations are recorded.
Derivative agreement at finite M is not a continuum accuracy certificate.
Base/end-point simplicity checks do not exclude intermediate crossings.
Gap derivatives refer to selected active edges; unchanged indices alone do not prove unique extrema.

|Design|M change|Frequency max relative change|Gap proxy change GHz|Gradient max relative change|Monotonicity violations|
|---|---|---|---|---|---|
|0|5 to 9|0.018996|-2.47971|0.222201|0|
|0|9 to 12|0.00613715|-0.82926|0.0501268|0|
|0|12 to 15|0.00369601|-0.486473|0.0468361|0|
|0|15 to 20|0.00377049|-0.518067|0.0651666|0|
|0|20 to 25|0.00222565|-0.280857|0.0389202|0|
|1|5 to 9|0.0207873|-2.1576|0.158305|0|
|1|9 to 12|0.00676138|-0.774244|0.100547|0|
|1|12 to 15|0.00408056|-0.457924|0.0485271|0|
|1|15 to 20|0.00409099|-0.501239|0.0127934|0|
|1|20 to 25|0.00247365|-0.289962|0.0547423|0|
|2|5 to 9|0.0234278|-1.96435|0.107929|0|
|2|9 to 12|0.00764016|-1.02061|0.0397845|0|
|2|12 to 15|0.00469636|-0.472172|0.112301|0|
|2|15 to 20|0.00476068|-0.465719|0.0162027|0|
|2|20 to 25|0.0028744|-0.259604|0.00897856|0|

|M|Parameter|FD step um|Max relative derivative error|Pass existing tolerance|
|---|---|---|---|---|
|5|L1|0.01|1.4111197974552038e-08|True|
|5|L1|0.001|2.968580858990017e-08|True|
|5|L2|0.01|1.3383511164972425e-08|True|
|5|L2|0.001|9.485409786459895e-08|True|
|20|L1|0.01|1.2029895041892202e-07|True|
|20|L1|0.001|3.0435477052351765e-06|True|
|20|L2|0.01|3.8007243065551505e-07|True|
|20|L2|0.001|1.6279533962103448e-06|True|
