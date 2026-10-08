"""D6: switch-threshold sensitivity. Existing D1/D2/D3 decks with CLI overrides: eta1 in {1e-3, 1e-2, 5e-2 (prod),
0.1, 0.2} (eta2 0.1 = prod), plus eta2 = 0 at eta1 5e-2 (one-sided switch). Cases: D1a_M100 (contact), D1b_M100_N4096
(resolved hypersonic), D1b_M1000_N1024 (under-resolved), D1b_M100_N256_boost, D3_M10_N1024 / D3_M30_N1024 (Noh),
D2b_b0.01_br1_M0_N128 (div B blob). The known answers are those of D1a/D1b/D2b/D3; the total-energy drift (hst tot-E)
measures what the switch costs in conservation. Writes inputs/D6_runs.txt."""
import os
from de_common import INP
cases = [("D1a_M100", "D1a_M100.in"), ("D1b_M100_N4096", "D1b_M100_N4096.in"), ("D1b_M1000_N1024", "D1b_M1000_N1024.in"),
         ("D1b_M100_N256_boost", "D1b_M100_N256_boost.in"), ("D3_M10_N1024", "D3_M10_N1024.in"),
         ("D3_M30_N1024", "D3_M30_N1024.in"), ("D2b_b0.01_br1_M0_N128", "D2b_b0.01_br1_M0_N128.in")]
runs = []
for name, deck in cases:
    for e1 in ("1e-3", "1e-2", "5e-2", "0.1", "0.2"):
        runs.append((f"D6/{name}_eta1_{e1}", deck, f"hydro/dual_energy_eta1={e1}"))
    runs.append((f"D6/{name}_eta1_5e-2_eta2off", deck, "hydro/dual_energy_eta1=5e-2 hydro/dual_energy_eta2=0.0"))
open(os.path.join(INP, "D6_runs.txt"), "w").write("".join(" | ".join(r) + "\n" for r in runs))
print(len(runs), "runs")
