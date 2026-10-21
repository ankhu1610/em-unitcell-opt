# Atlas-derived pure-PWE candidate validation

Status: complete.
Starts are deterministic proposals from the pure-PWE v2 atlas and deliberately separated
from the earlier M25-validated multistart solutions. The atlas interpolation never accepts a result.

|Target GHz|Atlas start L1 um|Atlas start L2 um|Final L1 um|Final L2 um|M20 gap GHz|M25 gap GHz|M25 error GHz|Guard|Outcome|
|---|---|---|---|---|---|---|---|---|---|
|7.5|150.420494|116.118671|150.630244|116.071883|7.50017763|7.49920813|0.000791875|True|success|
|15|164.508742|128.369322|164.557866|128.356887|15.000011|15.0013845|0.00138452|True|success|
|22.5|166.34634|97.1301617|166.387962|97.1196987|22.5000061|22.5068172|0.00681717|True|success|
|27.5|178.596991|119.793866|178.515224|119.81663|27.5000248|27.4920886|0.00791144|True|success|

Every M25 calculation evaluates the unchanged M20 candidate. A positive local outcome
does not establish uniqueness, global feasibility, full-Brillouin-zone behavior, or continuum accuracy.
