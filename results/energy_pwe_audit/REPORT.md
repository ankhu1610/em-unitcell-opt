# Hermitian energy-projector PWE

No training, fitting, reference-frequency datasets, or COMSOL calls.

## Discretization

The original ordered product has a nonzero commutator and is not Hermitian.
This new variant uses B_block + P* Delta_block P, where B=inverse(T(epsilon)),
Delta=T(1/epsilon)-B, and P=T(n n^T). It is the matrix of a real positive
quadratic energy for positive scalar materials. It is NOT claimed to be the
published Model B formula, nor a proven continuum error bound.
No imaginary eigenvalue parts are discarded; no assembled operator is averaged with its adjoint.
Hermiticity, material consistency, negative squared frequencies, and eigenpair residuals are checked.

## Refinement

First two bands, all sampled k points; Gamma zero excluded from relative changes.
These are successive-basis changes, not errors against a known 2D solution.

|Sides / (a/sqrt(3))|Cutoff|Energy change %|Direct PWE change %|Hermitian defect|Eigenpair residual|
|---|---:|---:|---:|---:|---:|
|[1.0, 0.65]|3|nan|nan|1.547e-16|3.602e-16|
|[1.0, 0.65]|5|0.484119|2.697700|2.383e-16|5.210e-16|
|[1.0, 0.65]|7|0.136303|1.150691|3.026e-16|5.388e-16|
|[1.0, 0.65]|9|0.052597|0.692307|3.297e-16|6.082e-16|
|[1.0, 0.65]|12|0.036365|0.593101|4.073e-16|7.576e-16|
|[1.0, 0.65]|15|0.020603|0.347987|4.381e-16|8.266e-16|
|[1.0, 0.65]|20|0.010863|0.352662|5.018e-16|8.351e-16|
|[1.15, 0.8]|3|nan|nan|1.463e-16|4.807e-16|
|[1.15, 0.8]|5|0.904149|3.399705|2.372e-16|5.183e-16|
|[1.15, 0.8]|7|0.148168|1.431785|3.374e-16|5.653e-16|
|[1.15, 0.8]|9|0.107023|0.734448|3.659e-16|5.974e-16|
|[1.15, 0.8]|12|0.032867|0.680741|4.389e-16|6.866e-16|
|[1.15, 0.8]|15|0.021002|0.409728|4.694e-16|8.380e-16|
|[1.15, 0.8]|20|0.012212|0.410780|5.376e-16|8.560e-16|
|[1.35, 1.0]|3|nan|nan|1.569e-16|6.391e-16|
|[1.35, 1.0]|5|1.947632|7.005083|2.632e-16|4.971e-16|
|[1.35, 1.0]|7|0.196645|1.436029|3.653e-16|7.026e-16|
|[1.35, 1.0]|9|0.076302|1.190863|3.903e-16|5.837e-16|
|[1.35, 1.0]|12|0.052239|0.799274|4.664e-16|9.600e-16|
|[1.35, 1.0]|15|0.039483|0.564875|5.174e-16|1.176e-15|
|[1.35, 1.0]|20|0.013306|0.493322|6.083e-16|1.077e-15|

## Selected-point gaps and cross-method check

No full-zone band-gap claim. Cross-method disagreement is not an error bound.

|Geometry|Final direct K GHz|Last change GHz|Final indirect M/K GHz|Last change GHz|Direct/energy frequency disagreement %|
|---|---:|---:|---:|---:|---:|
|[1.0, 0.65]|21.596507|0.004620|4.243064|0.004268|1.085646|
|[1.15, 0.8]|25.248321|0.009759|15.426659|0.001892|1.251583|
|[1.35, 1.0]|31.067883|0.008153|31.221958|0.011847|1.519293|

## Independent numerical checks

### Geometry [1.0, 0.65]

```json
{
  "normal_grid_doubling_change_pct": 0.000655302204676562,
  "time_reversal_max_abs_GHz": 1.3534418030758388e-10,
  "arc_quadrature_doubling_max_abs_eta": 5.721958498152797e-17,
  "refined_grid_first6_GHz": [
    [
      0.00020240932162542464,
      487.31381864158055,
      510.65803750498327,
      550.6027722893358,
      550.6816158991619,
      643.4668856278934
    ],
    [
      103.51966396081059,
      442.21956664122104,
      474.76507712187754,
      543.3833224311626,
      591.0640835068328,
      661.1085977259871
    ],
    [
      257.6608774534574,
      302.97318710176245,
      438.73026136536436,
      530.783565321526,
      685.364401021034,
      732.0081589421541
    ],
    [
      272.2867240801558,
      311.44077824520025,
      392.8176475759423,
      573.5025085944288,
      660.9582462452817,
      703.9305451658674
    ],
    [
      298.7295241578549,
      320.3236494376331,
      346.8358929441789,
      612.750721773801,
      638.3882430738691,
      684.330672879417
    ],
    [
      125.88191722035366,
      434.83596841026457,
      449.6112925301109,
      557.6762705142456,
      591.1718187687142,
      672.7526891152613
    ]
  ]
}
```

Independent general eigensolver (not constrained to return real eigenvalues):

```json
{
  "cutoff": 3,
  "max_eigenvalue_imaginary_over_spectral_scale": 3.757778992517005e-17,
  "general_vs_hermitian_eigenvalue_difference_over_scale": 4.805663254733034e-15
}
```

### Geometry [1.15, 0.8]

```json
{
  "normal_grid_doubling_change_pct": 0.0011162842981988737,
  "time_reversal_max_abs_GHz": 1.7507773009128869e-10,
  "arc_quadrature_doubling_max_abs_eta": 7.244417918960238e-17,
  "refined_grid_first6_GHz": [
    [
      4.866045023655803e-05,
      500.6709135825741,
      533.2480413081955,
      592.2611128001093,
      592.2852391276925,
      713.739571522428
    ],
    [
      110.10873984500653,
      464.34695681815134,
      496.8448511236585,
      587.2431218954181,
      627.302000713483,
      728.7126111981787
    ],
    [
      269.2509480028198,
      328.3625162731255,
      458.6962394569928,
      577.5948698020195,
      715.9055748808496,
      793.1733273898769
    ],
    [
      285.1140548799297,
      335.535375245738,
      417.0627061291604,
      615.2709868431075,
      695.793739983919,
      767.8709988814787
    ],
    [
      312.93659540794437,
      338.18206125520896,
      377.0253929123019,
      651.6097905938284,
      674.0670236717868,
      750.6725425845366
    ],
    [
      133.844516963529,
      456.299561139073,
      475.1295073324626,
      598.0555316340384,
      630.3911084923816,
      737.3798051289082
    ]
  ]
}
```

Independent general eigensolver (not constrained to return real eigenvalues):

```json
{
  "cutoff": 3,
  "max_eigenvalue_imaginary_over_spectral_scale": 4.7859623487081544e-17,
  "general_vs_hermitian_eigenvalue_difference_over_scale": 7.582222835396209e-15
}
```

### Geometry [1.35, 1.0]

```json
{
  "normal_grid_doubling_change_pct": 0.0008998397936344301,
  "time_reversal_max_abs_GHz": 1.091606804948242e-09,
  "arc_quadrature_doubling_max_abs_eta": 9.934984129320279e-17,
  "refined_grid_first6_GHz": [
    [
      0.0,
      533.7130205549402,
      580.4761958912352,
      668.4243373068953,
      668.4904173334727,
      833.7518643992635
    ],
    [
      122.00079677949671,
      505.599708960127,
      542.6890875904588,
      664.5865327017178,
      697.6634929307855,
      846.882301937809
    ],
    [
      292.3439278626062,
      372.1348970006124,
      501.0853922921213,
      656.569228463362,
      780.0399095809292,
      884.4433456203868
    ],
    [
      310.35942418209936,
      377.06296053141483,
      463.1719608693995,
      690.715529995682,
      763.4742538513767,
      868.0070970560536
    ],
    [
      340.91303796487455,
      371.98560245233085,
      429.34987015760197,
      728.4392138070465,
      739.8791786527036,
      853.6917152233065
    ],
    [
      148.22462953341594,
      497.34657883926036,
      523.2147453319644,
      672.9794507657715,
      701.7609988673853,
      853.7970017549173
    ]
  ]
}
```

Independent general eigensolver (not constrained to return real eigenvalues):

```json
{
  "cutoff": 3,
  "max_eigenvalue_imaginary_over_spectral_scale": 4.7280962129143756e-17,
  "general_vs_hermitian_eigenvalue_difference_over_scale": 7.482287792628887e-15
}
```

## Analytic layered test

The energy variant is run against an independent analytic transfer-matrix solution.
It also reduces algebraically to the flat-interface inverse rule:

```json
{
  "analytical_first4_GHz": [
    0.050796348525113506,
    0.11787364898721686,
    0.23570091382342054,
    0.27899165525360714
  ],
  "runs": [
    {
      "cutoff": 3,
      "factorized_max_relative_error_pct": 5.258254352236107,
      "direct_max_relative_error_pct": 34.35614120844296
    },
    {
      "cutoff": 7,
      "factorized_max_relative_error_pct": 0.08373809727265817,
      "direct_max_relative_error_pct": 5.869945735534044
    },
    {
      "cutoff": 15,
      "factorized_max_relative_error_pct": 0.008597682932530694,
      "direct_max_relative_error_pct": 2.316354483186466
    },
    {
      "cutoff": 31,
      "factorized_max_relative_error_pct": 0.001023805141005232,
      "direct_max_relative_error_pct": 1.1504274034790796
    }
  ]
}
```

## Limits and reproduction

The normal extension has medial-axis discontinuities and requires grid refinement.
No full-zone gap or universal accuracy claim follows from selected-point convergence.
Original production Torch code and earlier audit results are unchanged.
No pre-existing acceptance gates are changed or claimed to pass.
Run with NumPy from Tushar:

```
python tests/test_energy_pwe.py
python scripts/audit_energy_pwe.py
```
