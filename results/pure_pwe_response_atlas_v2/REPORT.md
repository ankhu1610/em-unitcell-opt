# Adaptive pure-PWE response atlas

Status: complete_for_requested_budget.
Validated M20 PWE seed observations: 46.
New M20 PWE evaluations: 12.
Distinct PWE points in the interpolation: 58.

The PWE contour panel is fit only to pure-PWE M20 sampled-path-gap values.
The COMSOL panel is a separately labelled view of data/processed/bands.npz; it is not used
to train, select, calibrate or accept PWE candidates. A visible disagreement is evidence, not an error to hide.
Neither panel certifies a full-Brillouin-zone gap or continuum convergence.
Any contour-derived design must still receive the existing full-path M20 and independent M25 checks.

|New point|L1 um|L2 um|M20 sampled-path gap GHz|Selection|
|---|---|---|---|---|
|1|148.17454|113.87272|6.21318735|target_aware_refinement|
|2|162.87532|130.20692|13.5065505|target_aware_refinement|
|3|140.00744|123.67324|0.13465199|overshoot_containment|
|4|179.20952|118.77298|28.3192324|target_aware_refinement|
|5|159.60848|102.43878|16.1738651|target_aware_refinement|
|6|140.00744|140.00744|-0.955854771|overshoot_containment|
|7|179.20952|140.00744|23.4021848|target_aware_refinement|
|8|143.27428|97.538517|5.40838312|target_aware_refinement|
|9|149.80796|91.004836|11.0071725|overshoot_containment|
|10|166.14216|97.538517|22.2174853|target_aware_refinement|
|11|148.17454|107.33904|7.18648112|target_aware_refinement|
|12|189.01004|91.004836|43.6252309|overshoot_containment|
