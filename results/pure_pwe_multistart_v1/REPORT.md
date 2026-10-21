# Pure-PWE anchor-free multistart reliability experiment

The existing full-path evaluator and inverse controls are unchanged.
Each target/start pair is an independent local search; all attempts are retained.
No ML, COMSOL fitting or historical response anchors are used.
A signed sampled-path gap is not a full-Brillouin-zone or continuum certificate.

Status: complete.
Completed cases: 12/12.

|Target GHz|Start|Initial L1 um|Initial L2 um|Final L1 um|Final L2 um|M20 gap GHz|M25 gap GHz|M25 error GHz|Error plus refinement GHz|Outcome|
|---|---|---|---|---|---|---|---|---|---|---|
|7.5|center|164.508742|115.506138|151.994429|122.258806|7.50044526|7.4980759|0.0019241|0.00281463|success|
|7.5|low_l1_high_l2|144.907701|135.10718|153.542575|129.489868|7.5000387|7.49639137|0.00360863|0.00368604|success|
|7.5|high_l1_low_l2|184.109784|95.9050966|152.365846|123.978435|7.50007687|7.49753347|0.00246653|0.00262026|success|
|15|center|164.508742|115.506138|161.514034|116.278564|15.0000088|15.002469|0.00246898|0.00246898|success|
|15|low_l1_high_l2|144.907701|135.10718|162.3831|119.728594|15.000008|15.0042948|0.00429484|0.00429484|success|
|15|high_l1_low_l2|184.109784|95.9050966|160.666612|112.913972|15.0005214|15.0018801|0.00188014|0.00188014|success|
|22.5|center|164.508742|115.506138|169.845491|110.490428|22.5007591|22.5127772|0.0127772|0.0127772|success|
|22.5|low_l1_high_l2|144.907701|135.10718|170.978578|114.737356|22.5031551|22.5150861|0.0150861|0.0150861|success|
|22.5|high_l1_low_l2|184.109784|95.9050966|168.975852|107.192336|22.5007233|22.510198|0.010198|0.010198|success|
|27.5|center|164.508742|115.506138|175.552087|108.981794|27.5001028|27.5041414|0.00414142|0.00414142|success|
|27.5|low_l1_high_l2|144.907701|135.10718|175.100425|107.295887|27.5000069|27.5040685|0.00406852|0.00406852|success|
|27.5|high_l1_low_l2|184.109784|95.9050966|173.763772|102.225783|27.5000329|27.5018763|0.00187626|0.00187626|success|

|Target GHz|Successful / planned|Completed|Unfinished|Success / completed|
|---|---|---|---|---|
|7.5|3/3|3|0|1|
|15|3/3|3|0|1|
|22.5|3/3|3|0|1|
|27.5|3/3|3|0|1|

Success requires the unchanged internal search tolerance, the unchanged final M25
tolerance, and the empirical guard abs(G20-target)+abs(G25-G20).
M25 checks the unchanged candidate, including locally unresolved searches; it never fits the geometry.
Unresolved local searches do not prove infeasibility. Different successful geometries are
expected because a scalar target generally does not uniquely determine two lengths.
Only identical, integrity-checked evaluations inside this new run are reused.
