# Offline atlas interpolant comparison

Post-launch offline comparison only. No changes to the running worker, frozen proposal surface, physics or acceptance tolerance.

|Model|Interior held-out groups|MAE GHz|RMSE GHz|Worst absolute GHz|81×81 range GHz|Range excursions|
|---|---|---|---|---|---|---|
|frozen_RBF_no_polynomial|52|0.304789|0.576205|2.192636|-2.41231 to 43.83279|15|
|RBF_linear_polynomial|52|0.040663|0.074981|0.276783|-1.11394 to 43.53535|4|
|piecewise_linear|52|0.122331|0.253728|1.368737|-0.95585 to 43.62523|0|

Nonuniform adaptive data; grouped leave-one-out is not an independent spatial test set. Boundary extrapolation is reported separately, and piecewise-linear interpolation intentionally returns no prediction outside its hull. Better CV does not certify the 0.05 GHz tolerance. Alternative fits are not deployed by this comparison.

All fits use the same saved pure-PWE M20 observations. Numerically equivalent coordinates are removed together in each holdout. Both RBF variants use the existing smoothing value, with no tuning search; only the polynomial tail differs. No COMSOL values are used. The existing four atlas starts and all validation outcomes are untouched.

|Frozen start|Existing RBF prediction GHz|Linear-tail RBF prediction GHz|Piecewise-linear prediction GHz|
|---|---|---|---|
|[0.21250000000000002, 0.5125000000000001]|7.5014956|7.3706024|7.3984319|
|[0.5, 0.7625000000000001]|14.9921987|14.9543630|15.0345185|
|[0.5375, 0.125]|22.5062598|22.4562824|22.4675476|
|[0.7875000000000001, 0.5875]|27.5111824|27.5996875|27.5959469|
