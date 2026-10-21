# Separate pure-PWE projector experiments

Three representative para geometries, two experimental projector extensions. Full-path gaps remain sampled-path gaps, not full-Brillouin-zone certificates. Frozen criteria.yaml supplies gradient tolerance and M=15-relative frequency diagnostic. Higher-M and grid comparisons are reported without changing thresholds. No optimizer is enabled automatically.

Completed checkpointed jobs: 120.
Protected files unchanged: 320 (SHA-256 verified).
Existing gradient tolerance: 0.0001. Existing frequency-refinement tolerance: 0.005, reference M=15.
No ML, COMSOL spectra or fitting. The production solver and historical results are unchanged.
Stable-nearest derivatives describe fixed assignments only; radial derivatives include all length dependence of that discrete solver.
Base and perturbed endpoint separations are checked; an intermediate crossing is not ruled out by this alone.
A finite-M gradient check is not a continuum-error certificate.

|Method|Design|M|Grid|Parameter|Small-step relative error|Changed projector samples|Pass all tested steps|
|---|---|---|---|---|---|---|---|
|homothetic_radial|0|5|256|L1|3.296467415201798e-07|0|True|
|homothetic_radial|0|5|256|L2|9.169247803171385e-07|0|True|
|homothetic_radial|0|5|512|L1|1.3777061266552856e-06|0|True|
|homothetic_radial|0|5|512|L2|1.54175439982858e-06|0|True|
|homothetic_radial|1|20|256|L1|3.052320828197808e-05|0|True|
|homothetic_radial|1|20|256|L2|1.845713968337917e-05|0|True|
|homothetic_radial|1|20|512|L1|1.2393998339038957e-05|0|True|
|homothetic_radial|1|20|512|L2|1.4314095812815256e-05|0|True|
|homothetic_radial|1|5|256|L1|1.1057901433009906e-06|0|True|
|homothetic_radial|1|5|256|L2|8.766084464180941e-07|0|True|
|homothetic_radial|1|5|512|L1|5.583136197272817e-07|0|True|
|homothetic_radial|1|5|512|L2|6.699086958707818e-07|0|True|
|homothetic_radial|2|5|256|L1|5.066005607720771e-07|0|True|
|homothetic_radial|2|5|256|L2|8.583090081989348e-07|0|True|
|homothetic_radial|2|5|512|L1|5.1222652855042e-07|0|True|
|homothetic_radial|2|5|512|L2|1.249359965939649e-06|0|True|
|stable_nearest|0|5|256|L1|4.42387707409073e-07|0|False|
|stable_nearest|0|5|256|L2|1.383598851563769e-06|0|False|
|stable_nearest|0|5|512|L1|6.523476862311021e-07|0|False|
|stable_nearest|0|5|512|L2|1.0458508150884544e-06|0|False|
|stable_nearest|1|20|256|L1|1.2322449679303e-05|0|False|
|stable_nearest|1|20|256|L2|1.4223436218254299e-05|0|False|
|stable_nearest|1|20|512|L1|1.6579934459365927e-05|0|False|
|stable_nearest|1|20|512|L2|2.659538654550122e-05|0|False|
|stable_nearest|1|5|256|L1|4.6112394718690186e-07|0|False|
|stable_nearest|1|5|256|L2|5.337808976204074e-07|0|False|
|stable_nearest|1|5|512|L1|5.192531668333279e-07|0|False|
|stable_nearest|1|5|512|L2|1.1223164384583231e-06|0|False|
|stable_nearest|2|5|256|L1|2.8236010273175594e-07|0|False|
|stable_nearest|2|5|256|L2|8.685621271409652e-07|0|False|
|stable_nearest|2|5|512|L1|4.4196681323526167e-07|0|False|
|stable_nearest|2|5|512|L2|5.247465672748251e-07|0|False|

## Fourier-cutoff refinement

|Method|Design|Grid|Full path|M transition|Max frequency relative difference|Gap change GHz|
|---|---|---|---|---|---|---|
|stable_nearest|0|256|False|5 to 9|0.00168855|-0.11088|
|stable_nearest|0|256|False|9 to 12|0.000197317|-0.00808819|
|stable_nearest|0|256|False|12 to 15|0.000214711|-0.00174406|
|stable_nearest|0|256|False|15 to 20|0.000108344|-0.00415885|
|stable_nearest|0|256|False|20 to 25|4.57416e-05|-0.00326525|
|stable_nearest|0|512|False|5 to 9|0.00168782|-0.110909|
|stable_nearest|0|512|False|9 to 12|0.000193657|-0.00945074|
|stable_nearest|0|512|False|12 to 15|0.000216981|-0.000603152|
|stable_nearest|0|512|False|15 to 20|0.000110053|-0.00401884|
|stable_nearest|0|512|False|20 to 25|4.76985e-05|-0.00354469|
|stable_nearest|0|512|True|15 to 20|0.000110053|-0.00401884|
|stable_nearest|0|512|True|20 to 25|4.88976e-05|-0.00354469|
|stable_nearest|1|256|False|5 to 9|0.00186843|-0.15138|
|stable_nearest|1|256|False|9 to 12|0.000355763|0.0176912|
|stable_nearest|1|256|False|12 to 15|0.000193811|-0.0316974|
|stable_nearest|1|256|False|15 to 20|0.000133347|0.00515078|
|stable_nearest|1|256|False|20 to 25|7.47886e-05|0.00177728|
|stable_nearest|1|512|False|5 to 9|0.00186246|-0.149256|
|stable_nearest|1|512|False|9 to 12|0.000351094|0.0169756|
|stable_nearest|1|512|False|12 to 15|0.000196979|-0.0307255|
|stable_nearest|1|512|False|15 to 20|0.000131773|0.00570419|
|stable_nearest|1|512|False|20 to 25|7.24143e-05|0.00272766|
|stable_nearest|1|512|True|15 to 20|0.000139173|0.00570419|
|stable_nearest|1|512|True|20 to 25|7.30516e-05|0.00272766|
|stable_nearest|2|256|False|5 to 9|0.00208527|0.145023|
|stable_nearest|2|256|False|9 to 12|0.0005264|0.000688285|
|stable_nearest|2|256|False|12 to 15|0.00024474|0.0109861|
|stable_nearest|2|256|False|15 to 20|0.000187563|-0.00636475|
|stable_nearest|2|256|False|20 to 25|6.8014e-05|-0.00341294|
|stable_nearest|2|512|False|5 to 9|0.00209501|0.147339|
|stable_nearest|2|512|False|9 to 12|0.000519454|-0.000356922|
|stable_nearest|2|512|False|12 to 15|0.000244068|0.0123618|
|stable_nearest|2|512|False|15 to 20|0.000189751|-0.00830952|
|stable_nearest|2|512|False|20 to 25|7.02825e-05|-0.00263968|
|stable_nearest|2|512|True|15 to 20|0.000190288|-0.00830952|
|stable_nearest|2|512|True|20 to 25|7.72614e-05|-0.00263968|
|homothetic_radial|0|256|False|5 to 9|0.00139821|-0.115163|
|homothetic_radial|0|256|False|9 to 12|0.000214111|0.0539765|
|homothetic_radial|0|256|False|12 to 15|0.000243386|-0.00776301|
|homothetic_radial|0|256|False|15 to 20|7.51681e-05|-0.00587369|
|homothetic_radial|0|256|False|20 to 25|6.9602e-05|-0.000948514|
|homothetic_radial|0|512|False|5 to 9|0.00139457|-0.114031|
|homothetic_radial|0|512|False|9 to 12|0.000217657|0.0534581|
|homothetic_radial|0|512|False|12 to 15|0.000234006|-0.00705734|
|homothetic_radial|0|512|False|15 to 20|7.6498e-05|-0.00589855|
|homothetic_radial|0|512|False|20 to 25|6.97048e-05|-0.00135461|
|homothetic_radial|0|512|True|15 to 20|8.21171e-05|-0.00589855|
|homothetic_radial|0|512|True|20 to 25|6.97162e-05|-0.00135461|
|homothetic_radial|1|256|False|5 to 9|0.00189565|-0.0950467|
|homothetic_radial|1|256|False|9 to 12|0.000388393|0.0554373|
|homothetic_radial|1|256|False|12 to 15|0.000137499|-0.0330311|
|homothetic_radial|1|256|False|15 to 20|0.000166531|0.0131541|
|homothetic_radial|1|256|False|20 to 25|4.76285e-05|0.00282977|
|homothetic_radial|1|512|False|5 to 9|0.00190445|-0.0960833|
|homothetic_radial|1|512|False|9 to 12|0.00038754|0.0551352|
|homothetic_radial|1|512|False|12 to 15|0.000138405|-0.032778|
|homothetic_radial|1|512|False|15 to 20|0.000169973|0.0121302|
|homothetic_radial|1|512|False|20 to 25|4.83444e-05|0.00269207|
|homothetic_radial|1|512|True|15 to 20|0.000171121|0.0121302|
|homothetic_radial|1|512|True|20 to 25|5.15204e-05|0.00269207|
|homothetic_radial|2|256|False|5 to 9|0.00237055|0.142893|
|homothetic_radial|2|256|False|9 to 12|0.000448584|-0.0754104|
|homothetic_radial|2|256|False|12 to 15|0.00021822|0.00208503|
|homothetic_radial|2|256|False|15 to 20|0.000142744|0.0119951|
|homothetic_radial|2|256|False|20 to 25|0.000109596|-0.0105633|
|homothetic_radial|2|512|False|5 to 9|0.00240714|0.141542|
|homothetic_radial|2|512|False|9 to 12|0.000444828|-0.0717568|
|homothetic_radial|2|512|False|12 to 15|0.000215919|-0.00102838|
|homothetic_radial|2|512|False|15 to 20|0.000145216|0.0134865|
|homothetic_radial|2|512|False|20 to 25|0.000109744|-0.0106606|
|homothetic_radial|2|512|True|15 to 20|0.000148328|0.0134865|
|homothetic_radial|2|512|True|20 to 25|0.000114747|-0.0106606|

## Normal-grid refinement

|Method|Design|M|Grid transition|Max frequency relative difference|Gap proxy change GHz|
|---|---|---|---|---|---|
|stable_nearest|0|5|256 to 512|9.54734e-06|0.00385419|
|stable_nearest|0|9|256 to 512|6.65449e-06|0.00382556|
|stable_nearest|0|12|256 to 512|6.34617e-06|0.00246301|
|stable_nearest|0|15|256 to 512|6.54923e-06|0.00360391|
|stable_nearest|0|20|256 to 512|7.44923e-06|0.00374392|
|stable_nearest|0|20|512 to 1024|3.44353e-06|0.00106509|
|stable_nearest|0|25|256 to 512|5.64775e-06|0.00346449|
|stable_nearest|1|5|256 to 512|1.56088e-05|-0.00649386|
|stable_nearest|1|9|256 to 512|1.27098e-05|-0.00437019|
|stable_nearest|1|12|256 to 512|8.07832e-06|-0.00508576|
|stable_nearest|1|15|256 to 512|1.00794e-05|-0.00411389|
|stable_nearest|1|20|256 to 512|1.04474e-05|-0.00356047|
|stable_nearest|1|20|512 to 1024|1.24728e-06|-6.79695e-05|
|stable_nearest|1|25|256 to 512|8.56953e-06|-0.0026101|
|stable_nearest|2|5|256 to 512|1.0801e-05|-0.0028464|
|stable_nearest|2|9|256 to 512|1.61585e-05|-0.000530638|
|stable_nearest|2|12|256 to 512|8.08718e-06|-0.00157584|
|stable_nearest|2|15|256 to 512|6.33467e-06|-0.00020017|
|stable_nearest|2|20|256 to 512|7.64787e-06|-0.00214494|
|stable_nearest|2|20|512 to 1024|2.74549e-06|-0.00101663|
|stable_nearest|2|25|256 to 512|7.45581e-06|-0.00137168|
|homothetic_radial|0|5|256 to 512|1.37426e-05|0.0014162|
|homothetic_radial|0|9|256 to 512|7.67501e-06|0.00254743|
|homothetic_radial|0|12|256 to 512|1.05488e-05|0.00202909|
|homothetic_radial|0|15|256 to 512|6.96124e-06|0.00273476|
|homothetic_radial|0|20|256 to 512|6.73922e-06|0.0027099|
|homothetic_radial|0|20|512 to 1024|1.39895e-06|0.000397231|
|homothetic_radial|0|25|256 to 512|4.23473e-06|0.00230381|
|homothetic_radial|1|5|256 to 512|9.03037e-06|0.00102076|
|homothetic_radial|1|9|256 to 512|2.48297e-06|-1.58014e-05|
|homothetic_radial|1|12|256 to 512|3.37848e-06|-0.000317905|
|homothetic_radial|1|15|256 to 512|1.48537e-06|-6.47515e-05|
|homothetic_radial|1|20|256 to 512|3.54706e-06|-0.00108864|
|homothetic_radial|1|20|512 to 1024|1.27949e-06|0.000340652|
|homothetic_radial|1|25|256 to 512|3.69237e-06|-0.00122634|
|homothetic_radial|2|5|256 to 512|3.37435e-05|0.000876379|
|homothetic_radial|2|9|256 to 512|1.08506e-05|-0.000474488|
|homothetic_radial|2|12|256 to 512|7.70682e-06|0.0031791|
|homothetic_radial|2|15|256 to 512|4.45562e-06|6.57014e-05|
|homothetic_radial|2|20|256 to 512|3.37161e-06|0.0015571|
|homothetic_radial|2|20|512 to 1024|1.65868e-06|4.32756e-05|
|homothetic_radial|2|25|256 to 512|2.70598e-06|0.00145977|

## Full-path gap sensitivity

|Parameter|Step um|Analytic GHz/um|FD GHz/um|Relative difference|Edges unchanged|
|---|---|---|---|---|---|
|L1|0.01|0.736385487|0.736385484|4.69896e-09|True|
|L1|0.001|0.736385487|0.736385462|3.38739e-08|True|
|L2|0.01|-0.186403697|-0.186403694|1.59681e-08|True|
|L2|0.001|-0.186403697|-0.18640364|3.02772e-07|True|
