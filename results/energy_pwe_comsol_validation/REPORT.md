# Frozen pure-PWE validation against existing COMSOL

Validation only: no solver execution, training, fitting, material adjustment, or cutoff selection.
Primary settings: cutoff 20, normal grid 256.
Shared k parameters: [0.37, 1.0, 1.53, 2.0, 2.61]. Gamma is excluded because COMSOL has no saved Gamma target.
Recovered source geometries are matched by unrounded dimensions; source spectra align with processed para.
Only the first two frequency-sorted bands are compared; no eigenfield-based mode tracking.

## Overall frequency agreement

```json
{
  "count": 30,
  "mae_GHz": 0.06183851295116985,
  "rmse_GHz": 0.06808551753841913,
  "max_abs_GHz": 0.12377736039883303,
  "mean_abs_relative_pct": 0.01990496531703108,
  "rms_relative_pct": 0.02032783848998565,
  "max_abs_relative_pct": 0.027550330765530425,
  "signed_mean_GHz": 0.06183851295116985
}
```

## Frequency agreement by geometry

|Design|MAE GHz|RMSE GHz|Max error GHz|Mean relative error %|Max relative error %|Direct PWE MAE at same cutoff GHz|
|---|---:|---:|---:|---:|---:|---:|
|D001|0.043899|0.046700|0.063460|0.015659|0.020163|2.173925|
|D028|0.065049|0.070971|0.114656|0.021164|0.024698|2.893038|
|D064|0.076567|0.081787|0.123777|0.022892|0.027550|4.379829|

## Local band-gap comparisons

Direct = f2(K)-f1(K). Indirect M/K = f2(M)-f1(K). These are NOT verified full-path/full-zone gaps.

|Design|Gap definition|COMSOL GHz|PWE GHz|Absolute error GHz|Relative error %|
|---|---|---:|---:|---:|---:|
|D001|direct_K|21.596267|21.596507|0.000240|0.001112|
|D001|indirect_M_K|4.220609|4.243064|0.022455|0.532036|
|D028|direct_K|25.250948|25.248321|0.002627|0.010405|
|D028|indirect_M_K|15.415014|15.426659|0.011645|0.075545|
|D064|direct_K|31.080053|31.067883|0.012170|0.039157|
|D064|indirect_M_K|31.220945|31.221958|0.001013|0.003245|

## Individual frequencies

|Design|k|Band|COMSOL GHz|PWE GHz|Signed difference GHz|
|---|---:|---:|---:|---:|---:|
|D001|0.37|1|103.500957|103.519477|0.018520|
|D001|0.37|2|442.157689|442.221149|0.063460|
|D001|1.00|1|257.624767|257.661499|0.036731|
|D001|1.00|2|302.911170|302.972246|0.061076|
|D001|1.53|1|272.249485|272.287053|0.037568|
|D001|1.53|2|311.380340|311.440498|0.060157|
|D001|2.00|1|298.690561|298.729182|0.038621|
|D001|2.00|2|320.286828|320.325689|0.038861|
|D001|2.61|1|125.859903|125.881913|0.022011|
|D001|2.61|2|434.776828|434.838818|0.061989|
|D028|0.37|1|110.085183|110.108055|0.022872|
|D028|0.37|2|464.235972|464.350628|0.114656|
|D028|1.00|1|269.184096|269.250347|0.066251|
|D028|1.00|2|328.293510|328.361580|0.068070|
|D028|1.53|1|285.046827|285.113288|0.066460|
|D028|1.53|2|335.471122|335.534550|0.063428|
|D028|2.00|1|312.878496|312.934921|0.056425|
|D028|2.00|2|338.129444|338.183242|0.053798|
|D028|2.61|1|133.816964|133.844014|0.027050|
|D028|2.61|2|456.193178|456.304655|0.111477|
|D064|0.37|1|121.973750|122.001737|0.027987|
|D064|0.37|2|505.472950|505.596727|0.123777|
|D064|1.00|1|292.264986|292.345506|0.080520|
|D064|1.00|2|372.056110|372.137618|0.081507|
|D064|1.53|1|310.278758|310.360652|0.081894|
|D064|1.53|2|376.987364|377.065415|0.078051|
|D064|2.00|1|340.835165|340.915659|0.080494|
|D064|2.00|2|371.915218|371.983542|0.068324|
|D064|2.61|1|148.195180|148.225069|0.029889|
|D064|2.61|2|497.228873|497.342104|0.113230|

## Grid sensitivity and baseline

Doubled-grid predictions were also generated before this comparison; they are disclosed, not selected by COMSOL error.
```json
{
  "direct_PWE_same_cutoff": {
    "count": 30,
    "mae_GHz": 3.1489309714742637,
    "rmse_GHz": 3.4715708734345654,
    "max_abs_GHz": 5.814552628150523,
    "mean_abs_relative_pct": 1.0261771538441977,
    "rms_relative_pct": 1.062366375644914,
    "max_abs_relative_pct": 1.541532704765336,
    "signed_mean_GHz": 3.1489309714742637
  },
  "refined_normal_grid": {
    "count": 30,
    "mae_GHz": 0.061435642803641125,
    "rmse_GHz": 0.06767496539532412,
    "max_abs_GHz": 0.1267589736111745,
    "mean_abs_relative_pct": 0.01980084600732742,
    "rms_relative_pct": 0.020226340672557845,
    "max_abs_relative_pct": 0.02701023140294906,
    "signed_mean_GHz": 0.061435642803641125
  }
}
```

## Limitations

Agreement is with saved COMSOL results, not an independently certified continuum solution.
COMSOL mesh error and remaining PWE truncation error may both contribute.
These geometries appeared in earlier diagnostic work: this is not a blind holdout.
No claim is made for all geometries or all k points. Sparse samples cannot certify extrema or group velocity.
The full-path metrics contract is intentionally not evaluated on a five-point subset.
Per-band metrics, processed/source differences, basis history and input hashes are in results.json.
No existing acceptance thresholds or solver files were changed.
