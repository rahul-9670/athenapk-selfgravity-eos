#!/usr/bin/env python
"""D6: a STRUCTURED dust field through refine / derefine events (AUDIT_DUST item 7a: F-21 covered only standing
coarse-fine faces with a uniform f_dg).

2D oblique periodic slabs (eos_test slab, mx = my = 1), root 64^2, 16^2 blocks, refinement = adaptive, numlevel 3
(finest = 256^2 equivalent), Parthenon derivative_order_1 on prim component 9 (= scalar 0 = f_dg) so the mesh follows
the moving dust fronts: blocks refine ahead of a front and derefine behind it every few cycles. 5 crossings.
  D6a  = D4b density contact x10 carrying the D4a f_dg / a_c front (freeze_growth): exact sum(rho f dV) is conserved by
         transport, prolongation and restriction (all conservative), so ANY change is React clamping an overshoot
         above f_ref (apply_sublimation) -- the quantity this test exists to measure; plus the bounds.
  D6c  = D4c (temperature contact across T_subl, freeze_growth): dust lost to numerically mixed hot gas, AMR vs the
         uniform 128^2 / 256^2 runs (D4c_2d_N128 and D6c_uni256 here).
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
sys.path.insert(0, _VT + "/dust/scripts")
from du_common import DU, write
import make_decks_D4 as D4


def amr(deck, rho_dense):
    deck = deck.replace("<parthenon/mesh>\nrefinement = none\n",
                        "<parthenon/mesh>\nrefinement = adaptive\nnumlevel = 3\n")
    # AthenaPK's driver tags refinement PER BLOCK (Refinement::Tag<MeshBlockData>, hydro_driver.cpp), and that path
    # never evaluates Parthenon's generic <parthenon/refinementN> criteria (they act once, at initialisation only:
    # first attempt 2026-10-12 refined at t = 0 and collapsed to the root grid after). So the refinement must come
    # from an AthenaPK criterion: maxdensity on the DENSE slab, which carries one side of the dust front. Blocks
    # holding dense gas refine to level 2, blocks without it derefine: the refined region travels with the slab, with
    # refine events at its leading contact and derefine events at its trailing contact -- exactly where f_dg jumps.
    return deck + f"""
<refinement>
type = maxdensity
maxdensity_refine_above = {0.7 * rho_dense!r}
maxdensity_deref_below = {0.5 * rho_dense!r}
"""


def main():
    meta4 = json.load(open(f"{DU}/inputs/D4_meta.json"))
    runs, meta = [], {}
    # D6a: the D4b density contact (x10, uniform P, growth frozen) with the D4a dust front ON it (f_ref | f_ref/2,
    # fixed grain number), so the structured f_dg co-moves with the density step the criterion follows.
    for v, src in (("D6a", "D4b_1d_N64"), ("D6c", "D4c_1d_N64")):
        m = dict(meta4[src])
        if v == "D6a":
            m["scR"] = list(meta4["D4a_1d_N64"]["scR"])
            m["over"] = {"freeze_growth": "true"}
        tlim = float(m["tlim"]) / 2 / (2 ** 0.5)   # 5 crossings of the oblique wavelength 1/sqrt(2)
        for tag, N, mb, adaptive in (("amr", 64, 16, True), ("uni256", 256, 32, False)):
            name = f"{v}_{tag}"
            d = D4.deck(name, 2, N, m["L"], m["R"], m["scL"], m["scR"], m["u"], tlim, m["over"], 16)
            d = d.replace("nx1 = 32\nnx2 = 32\nnx3 = 1\n", f"nx1 = {mb}\nnx2 = {mb}\nnx3 = 1\n")
            if adaptive:
                d = amr(d, max(m["L"][0], m["R"][0]))
            write(f"{DU}/inputs/D6/{name}.in", d)
            runs.append(f"D6/{name} | D6/{name}.in | | 16")
            meta[name] = dict(m, variant=v, tlim=tlim, adaptive=adaptive, N=N)
    write(f"{DU}/inputs/D6_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D6_meta.json", json.dumps(meta, indent=1))
    print("\n".join(runs))


if __name__ == "__main__":
    main()
