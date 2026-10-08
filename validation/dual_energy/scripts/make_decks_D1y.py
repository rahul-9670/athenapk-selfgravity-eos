"""D1y: the DE-F3 candidate fix hydro/dual_energy_flux = density on the D1x contact cases (and the D1b smooth flows,
to check that it does not degrade them). Same decks, override hydro/dual_energy_flux=density; dual ON only (OFF never
reads the key). Writes inputs/D1y_runs.txt."""
import json, os
from de_common import INP
DEN = "hydro/dual_energy_flux=density"
runs = []
for line in open(os.path.join(INP, "D1x_runs.txt")):
    d, k, o = [x.strip() for x in line.split("|")]
    if d.endswith("_on") and "_dc_" not in d and "eta1e-3" not in d:
        runs.append((d.replace("D1x/", "D1y/").replace("D1x_", "D1y_"), k, (o + " " + DEN).strip()))
for line in open(os.path.join(INP, "D1_runs.txt")):
    d, k, o = [x.strip() for x in line.split("|")]
    if d.endswith("_on") and (d.startswith("D1a/") or ("_N1024_" in d or "_N4096_" in d or "boost" in d)):
        runs.append((d.replace("D1a/", "D1y/").replace("D1b/", "D1y/").replace("_on", "_den"), k, (o + " " + DEN).strip()))
open(os.path.join(INP, "D1y_runs.txt"), "w").write("".join(" | ".join(r) + "\n" for r in runs))
print(len(runs), "runs")
