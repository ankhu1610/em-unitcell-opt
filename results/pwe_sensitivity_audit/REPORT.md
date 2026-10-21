# Pure-PWE L1/L2 sensitivity audit

Five-point spectrum and five-point gap proxy; not a full 281-point gap sensitivity certification. M=20 only for middle geometry. Reference gradient tolerance is read unchanged from criteria.yaml.

Analytic material shape derivative uses boundary transport, not finite differences.
Branch derivative differentiates the complete energy operator with fixed normal assignments.
Full-rebuild differences recompute material AND normal projectors at perturbed lengths.
Projector-secanted chain is a finite-step diagnostic, NOT an independently analytic projector derivative.
Unchanged criteria.yaml relative gradient threshold: 0.0001.
Validated solver/results hashes preserved: True.
No ML, COMSOL spectral data or fitting. No inverse optimizer enabled.

|L1 um|L2 um|M|Grid|Derivative|Material rel error*|Frozen rel error*|Full rel error*|Switches*|FD step spread GHz/um|
|---|---|---|---|---|---|---|---|---|---|
|147.00781|98.00521|5|256|L1|6.9e-11|5.05e-07|0.999|949|530.244|
|147.00781|98.00521|5|256|L2|9.49e-11|8.11e-07|1|1220|325.273|
|147.00781|98.00521|5|512|L1|6.9e-11|7.04e-07|0.999|3773|521.46|
|147.00781|98.00521|5|512|L2|9.49e-11|1.69e-06|1|4794|380.296|
|161.00856|112.00595|5|256|L1|6.61e-11|3e-07|1|2152|1217.47|
|161.00856|112.00595|5|256|L2|1.19e-10|2.3e-07|1|1267|447.188|
|161.00856|112.00595|5|512|L1|6.61e-11|2.83e-07|1|8476|1203.67|
|161.00856|112.00595|5|512|L2|1.19e-10|1.18e-06|1|5009|447.872|
|182.00967|133.00707|5|256|L1|1.04e-10|4.43e-07|1.02|940|271.383|
|182.00967|133.00707|5|256|L2|5.8e-11|6.67e-07|1.54|73|39.7457|
|182.00967|133.00707|5|512|L1|1.04e-10|2.3e-07|1.01|3831|250.79|
|182.00967|133.00707|5|512|L2|5.8e-11|8.43e-07|1.45|260|41.5575|
|161.00856|112.00595|20|256|L1|8.31e-11|1.59e-05|1.03|2152|75.2732|
|161.00856|112.00595|20|256|L2|1.25e-10|3.11e-05|1.05|1267|51.0183|
|161.00856|112.00595|20|512|L1|8.31e-11|1.73e-05|1.04|8476|67.2004|
|161.00856|112.00595|20|512|L2|1.25e-10|3.32e-06|1.04|5009|50.0027|

*At the smallest perturbation. Full relative error excludes near-degenerate base modes.
Other rows, both signs, all mode sensitivities, eigenpair residuals and band-edge locations are in results.json.
Finite-grid derivatives can match at tiny steps yet fail to converge under grid refinement.
Gap values here are five-point proxies, not the full 281-point validated band gap.
Analytic branches are not globally smooth at nearest-interface assignment switches.
