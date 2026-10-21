# Independent verification: pure-PWE multistart inverse

Status: verified_complete. Verified cases: 12/12; targets: 4; starts per target: 3.
Existing final tolerance: 0.05 GHz. M25 matches: 12/12; empirical refinement guards passed: 12/12.

Combined success (internal search, M25 match and refinement guard): 12/12.

|Target GHz|Start|Initial L1 um|Initial L2 um|Returned L1 um|Returned L2 um|M25 gap GHz|M25 error GHz|M20-to-M25 change GHz|Guard|Search status|
|---|---|---|---|---|---|---|---|---|---|---|
|7.5|center|164.5087423|115.5061382|151.9944294|122.2588065|7.498075896|0.0019241|0.00236937|True|matched|
|7.5|low_l1_high_l2|144.9077007|135.1071799|153.5425752|129.4898679|7.496391366|0.00360863|0.00364734|True|matched|
|7.5|high_l1_low_l2|184.109784|95.90509659|152.3658462|123.9784347|7.497533472|0.00246653|0.00254339|True|matched|
|15|center|164.5087423|115.5061382|161.514034|116.2785643|15.00246898|0.00246898|0.00246013|True|matched|
|15|low_l1_high_l2|144.9077007|135.1071799|162.3830998|119.7285943|15.00429484|0.00429484|0.00428683|True|matched|
|15|high_l1_low_l2|184.109784|95.90509659|160.6666119|112.9139715|15.00188014|0.00188014|0.00135871|True|matched|
|22.5|center|164.5087423|115.5061382|169.845491|110.4904281|22.51277723|0.0127772|0.0120182|True|matched|
|22.5|low_l1_high_l2|144.9077007|135.1071799|170.9785776|114.7373559|22.51508607|0.0150861|0.0119309|True|matched|
|22.5|high_l1_low_l2|184.109784|95.90509659|168.9758519|107.1923362|22.51019803|0.010198|0.00947469|True|matched|
|27.5|center|164.5087423|115.5061382|175.5520869|108.9817943|27.50414142|0.00414142|0.0040386|True|matched|
|27.5|low_l1_high_l2|144.9077007|135.1071799|175.1004251|107.2958873|27.50406852|0.00406852|0.00406165|True|matched|
|27.5|high_l1_low_l2|184.109784|95.90509659|173.7637717|102.2257832|27.50187626|0.00187626|0.00184334|True|matched|

## Starting-point sensitivity

|Target GHz|Combined success / starts|M25 matches / starts|Guards passed / starts|Returned L1 span um|Returned L2 span um|Maximum pairwise length distance um|
|---|---|---|---|---|---|---|
|7.5|3/3|3/3|3/3|1.54814579|7.23106143|7.39493102|
|15|3/3|3/3|3/3|1.71648791|6.8146228|7.02747569|
|22.5|3/3|3/3|3/3|2.00272573|7.54501971|7.80629443|
|27.5|3/3|3/3|3/3|1.78831516|6.75601108|6.98868778|

Length spread includes all returned candidates, including unresolved searches. Different lengths can realize the same scalar gap; these spans are not parameter-recovery errors.
Unresolved cases under the combined success criterion: [].
A local search or refinement-guard failure is recorded as unresolved, not as proof of infeasibility.

## Numerical evidence

Fresh distinct full-path evaluations: 46; cutoff counts: {'20': 34, '25': 12}.
Search calls: 53; reused exact-point calls: 19.
Summed saved evaluation time: 8275.8933 seconds; last-invocation elapsed time: 8287.3386 seconds.
Per-evaluation elapsed time includes assembly, the sampled path, checks and derivatives. Summed work time need not equal wall time; last invocation excludes earlier resumed invocations.
Worst CPU/device frequency difference: 8.57667e-09 GHz.
Worst relative eigenpair residual: 6.1172926e-09.
Verified immutable source/input files: 673.
Preflight tests: 235 passed; 0 failures; 0 errors.

Verification checked saved checkpoint identities, payload hashes, all path diagnostics, every search trace and its prescribed starting geometry, and independently recomputed both cutoff errors and the unchanged-candidate refinement guard. Verification did not execute PWE.
Only exact-point checkpoints generated within this new scientific run are shared; historical anchor libraries are not used.

No ML training, COMSOL execution, COMSOL spectra, fitting, historical anchor initialization, or changes to protected earlier solver/results are part of this experiment.
Finite deterministic target/start matrix, not a statistical guarantee over all requested gaps or starting geometries.
The gaps refer to the existing sampled path. Two-cutoff agreement is empirical, not a continuum error bound or a complete Brillouin-zone gap certificate. No uniqueness is asserted.
