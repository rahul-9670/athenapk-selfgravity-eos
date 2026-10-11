#!/usr/bin/env python
"""GATE K4: does Sn reproduce the diffusion limit, and up to what tau_cell?

Pure scattering (kappa_a=0) in a static uniform medium conserves Er and spreads a Gaussian slab
as Var(t) = Var0 + 2 D t with D = chat/(3 rho kappa_s) -- exact, no free parameters. We measure
the spreading rate and divide by the analytic D.

WHAT EACH NUMBER MEANS
  D_meas/D_ana ~ 1   the scheme reproduces the diffusion limit at that tau_cell
  D_meas/D_ana > 1   step differencing's numerical diffusion (~c*dx) is dominating the physical
                     one; predicted to grow as ~1 + O(tau_cell)

THE tau_cell = 0.1 ARM GRADES THE APPARATUS, NOT THE CODE. The mean free path is resolved 10x
there, so the scheme must be accurate; if that arm does not return ~1 the measurement itself is
wrong and no other row means anything. This is the control the ladder is anchored on.

Er CONSERVATION is checked independently: pure scattering must conserve total Er to solver
tolerance. If it does not, the run is not the problem this analysis assumes and the variance is
being measured on a decaying pulse.
"""
import sys, glob, re, h5py, numpy as np

CHAT = 100.0   # c_code / creduc from deck_b.in
RHO  = 1.0

LBOX = 4.0     # x1max - x1min in deck_b.in; the fundamental wavenumber is 2*pi/LBOX

def moments(fn):
    """Return (t, Var_x, Er_total, |A1|) from one dump.

    TWO ESTIMATORS OF D, AND ONLY ONE OF THEM IS EXACT ON A PERIODIC BOX.
      Var_x  : second moment, Var = Var0 + 2 D t.  BIASED LOW on a periodic domain -- as the
               pulse widens its periodic images overlap and the second moment saturates toward
               L^2/12, so the fitted slope under-reads D.  Measured on these very dumps
               (2026-09-14): refitting with later windows made the ratio FALL (0.974 -> 0.965
               -> 0.953), the opposite of a start-up transient, and every K4 number published
               before this date (SN_TRANSPORT_DESIGN.md sections 13, 16) carries the bias.
      |A1|   : amplitude of the fundamental Fourier mode k1 = 2 pi / L.  For the diffusion
               equation on the ring every mode decays EXACTLY as exp(-D k1^2 t), images and
               all, so ln|A1| = const - D k1^2 t is the estimator to fit.  Kept as the
               primary; the second moment is printed alongside, labelled biased, so the old
               tables can be compared against it.
    """
    with h5py.File(fn, "r") as h:
        t  = float(h["Info"].attrs["Time"])
        Er = np.array(h["rad.Er"])                    # (nb, k, j, i)
        x  = np.array(h["VolumeLocations"]["x"])      # (nb, i) cell centres
    # broadcast x over (k, j)
    xb = x[:, None, None, :] * np.ones_like(Er)
    k1 = 2.0 * np.pi / LBOX
    A1 = abs((Er * np.exp(-1j * k1 * xb)).sum())      # uniform cells: equal weights
    w  = Er - Er.min()                                # remove the E0 floor
    W  = w.sum()
    if W <= 0: return t, np.nan, Er.sum(), A1
    xbar = (w * xb).sum() / W
    var  = (w * (xb - xbar) ** 2).sum() / W
    return t, var, Er.sum(), A1

def analyse(arm, kappa_s):
    fns = sorted(glob.glob(f"{arm}/parthenon.out0.*.phdf"))
    if len(fns) < 4: return None, f"only {len(fns)} dumps"
    rows = [moments(f) for f in fns]
    rows.sort(key=lambda r: r[0])
    t   = np.array([r[0] for r in rows])
    var = np.array([r[1] for r in rows])
    tot = np.array([r[2] for r in rows])
    A1  = np.array([r[3] for r in rows])
    cons = abs(tot[-1] - tot[0]) / abs(tot[0])
    # drop the first fifth: Fr starts at 0 and needs a moment to reach Fick's law
    k0 = max(1, len(t) // 5)
    slope = np.polyfit(t[k0:], var[k0:], 1)[0]
    D_mom = slope / 2.0                                          # BIASED (see moments())
    k1 = 2.0 * np.pi / LBOX
    D_meas = -np.polyfit(t[k0:], np.log(A1[k0:]), 1)[0] / k1**2  # exact ring estimator
    D_ana  = CHAT / (3.0 * RHO * kappa_s)
    return (D_meas, D_ana, D_meas / D_ana, cons, var[0], var[-1], D_mom / D_ana), None

# dd_tau3 is listed but usually absent: it costs ~32 min of GPU on its own (step measured 1911
# wsec_total for it, job 1724827) so it is run in a separate job. A missing arm prints "only 0
# dumps" and is skipped, which is why it can sit in the list unconditionally.
ARMS = [("dd_tau0p2", 25.6, 0.2), ("dd_tau0p5", 64.0, 0.5), ("dd_tau1", 128.0, 1.0),
        ("dd_tau3", 384.0, 3.0),
        ]

print(f"{'arm':8s} {'tau_cell':>9s} {'kappa_s':>9s} {'D_analytic':>12s} {'D_measured':>12s}"
      f" {'ratio':>8s} {'dEr/Er':>10s} {'ratio(2nd-mom, BIASED)':>24s}")
ok = True
rows = []
for arm, ks, tau in ARMS:
    res, err = analyse(arm, ks)
    if res is None:
        print(f"{arm:8s} {tau:9.2f} {ks:9.1f}   {err}")
        continue
    D_meas, D_ana, ratio, cons, v0, v1, rmom = res
    print(f"{arm:8s} {tau:9.2f} {ks:9.1f} {D_ana:12.4e} {D_meas:12.4e} {ratio:8.3f} {cons:10.2e}"
          f" {rmom:24.3f}")
    rows.append((tau, ratio, cons))
    if cons > 1e-6:
        print(f"         ^ Er NOT CONSERVED to {cons:.1e} -- pure scattering must conserve it;"
              " the variance is being measured on a decaying pulse")
        ok = False

print()
cal = [r for r in rows if r[0] <= 0.21]
if not cal:
    print("NO CALIBRATION ARM -- the ladder is unanchored and means nothing."); sys.exit(1)
c = cal[0][1]
print(f"CALIBRATION (tau_cell = 0.2): ratio = {c:.3f}", end="   ")
if abs(c - 1.0) < 0.25:
    print("OK -- the measurement is sound, so the rest of the ladder is meaningful")
else:
    print("*** the APPARATUS is wrong, not the code -- fix this before reading any other row ***")
    ok = False

print("\nPredicted signature of step differencing: ratio ~ 1 + O(tau_cell).")
for tau, ratio, _ in rows:
    print(f"   tau_cell {tau:6.2f} -> excess numerical diffusion {100*(ratio-1):+8.1f} %")
print("\nGATE K4:", "measurement VALID" if ok else "INVALID")
sys.exit(0 if ok else 1)
