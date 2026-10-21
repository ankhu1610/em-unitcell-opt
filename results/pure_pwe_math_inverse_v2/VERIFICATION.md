# Independent saved-result verification: anchor-free PWE inverse

Status: verified_complete. Targets verified: 3.
Existing tolerance: 0.05 GHz. M25 matches: 3; empirical refinement guards passed: 3.

|Target GHz|L1 um|L2 um|M20 gap GHz|M25 gap GHz|M20 error GHz|M25 error GHz|Cutoff change GHz|Error + change GHz|Guard|
|---|---|---|---|---|---|---|---|---|---|
|10|155.5996843|121.385011|10.00003416|9.997321282|3.41621e-05|0.00267872|0.00271288|0.00274704|True|
|20|167.7662683|114.6659119|20.00001249|20.01160894|1.24942e-05|0.0116089|0.0115964|0.0116089|True|
|30|178.3104194|108.2534862|30.00040374|29.99856979|0.00040374|0.00143021|0.00183395|0.00223769|True|

Distinct full-path physics evaluations: 12; by cutoff: {'20': 9, '25': 3}.
Fixed initial normalized coordinates for every request: [0.5, 0.5].
Search calls: 11; cached calls: 2.
Saved evaluation work time: 1971.84 seconds; mean: 164.32 seconds.
Saved per-evaluation elapsed time includes assembly, sampled path, checks and derivatives; summed work time is not necessarily experiment wall time.

Worst CPU/device frequency difference: 6.52041e-09 GHz.
Worst relative eigenpair residual: 5.43027e-09.
Recorded path-point backends: {'cuda': 3372}.
Immutable input/source files verified: 635.
Preflight tests: 153 passed; 0 failures; 0 errors.

Verification recomputed assessments from saved frequencies and checked checkpoint identities, checksums, full-path diagnostics, accepted traces, unchanged refinement geometries and protected hashes.
It did not execute PWE, ML, COMSOL, fitting or inverse optimization.
Repeated identical points within this experiment share checked PWE checkpoints; these are not historical saved-design anchors.

These are sampled-path gaps and empirical two-cutoff checks, not certified continuum errors or complete Brillouin-zone gaps. A scalar gap need not identify a unique pair of side lengths; unresolved local searches do not prove a target infeasible.
