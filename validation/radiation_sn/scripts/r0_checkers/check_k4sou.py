#!/usr/bin/env python
"""GATE K4 for the sou/step arms of jobs 2749734 / 2749735, reusing check_k4dd.py's analyse()
verbatim (same variance fit, same Er conservation test, same first-fifth discard) so the
numbers are comparable to the published ladder rather than to a second implementation."""
import sys, glob, importlib.util, os
spec = importlib.util.spec_from_file_location("k4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_k4dd.py"))
k4 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
import builtins
_exit = sys.exit
sys.exit = lambda *a: None          # check_k4dd runs its own ladder at import; neutralise it
try:
    spec.loader.exec_module(k4)     # it runs its own (empty) ladder at import and may raise
except Exception:
    pass
sys.exit = _exit
print("\n" + "=" * 78)
print("SOU LADDER (this job)")
# f_* arms = the SHIPPED binary (athenaPK_sou3/4, two-kernel SnApplyA), jobs 2749804/2749805.
ARMS = [(a, ks, tau) for a, ks, tau in
        [("f_step_tau0p2", 25.6, 0.2), ("f_step_tau0p5", 64.0, 0.5), ("f_step_tau1", 128.0, 1.0),
         ("f_step_tau3", 384.0, 3.0),
         ("f_sou1_tau0p2", 25.6, 0.2), ("f_sou1_tau0p5", 64.0, 0.5), ("f_sou1_tau1", 128.0, 1.0),
         ("f_sou1_tau3", 384.0, 3.0), ("f_sou1_tau1_nb32", 128.0, 1.0),
         # kappa dial (job 2749782) and angular-resolution arms (job 2749818), earlier binaries
         ("sou094_tau0p2", 25.6, 0.2), ("sou094_tau1", 128.0, 1.0), ("sou000_tau1", 128.0, 1.0),
         ("ang_sou_n6p8", 128.0, 1.0), ("ang_sou_n8p12", 128.0, 1.0),
         ("ang_sou_n10p16", 128.0, 1.0), ("ang_step_n10p16", 128.0, 1.0)]]
print(f"{'arm':16s} {'tau_cell':>9s} {'D_analytic':>12s} {'D_meas(Fourier)':>15s} {'ratio':>8s}"
      f" {'dEr/Er':>10s} {'ratio(2nd-mom,BIASED)':>22s}")
bad = []
for arm, ks, tau in ARMS:
    res, err = k4.analyse(arm, ks)
    if res is None:
        print(f"{arm:16s} {tau:9.2f}   {err}")
        continue
    D_meas, D_ana, ratio, cons, v0, v1, rmom = res
    flag = "" if cons <= 1e-6 else "  *** Er NOT CONSERVED ***"
    print(f"{arm:16s} {tau:9.2f} {D_ana:12.4e} {D_meas:15.4e} {ratio:8.4f} {cons:10.2e} {rmom:22.4f}{flag}")
    if cons > 1e-6: bad.append(arm)
print("\nEr conservation requirement is 1e-6; violations:", bad if bad else "none")
