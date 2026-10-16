"""Shared helpers: load the production EOS table (v2, cgs) and evaluate it exactly as the runtime does
(bilinear, clamped) in CODE units of the default (non-BE) unit system."""
import numpy as np
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
TABLE = _os.environ.get("EOS_TABLE", _REPO + "/src/eos/eos_table_hires_v5_cold.bin")  # EOS_TABLE: another table
RHO_U, V_U = 5.467e-19, 1.9e4          # physical_units.hpp defaults for a non-BE pgen


def load(path=TABLE, rho_u=RHO_U, v_u=V_U):
    with open(path, "rb") as f:
        assert np.fromfile(f, np.int64, 1)[0] == 0x454F535441424C31
        ver, flags, nr, ne, nT = (int(x) for x in np.fromfile(f, np.int64, 5))
        lr0, dlr, le0, dle, lT0, dlT, _, _ = np.fromfile(f, np.float64, 8)
        P = np.fromfile(f, np.float64, nr * ne).reshape(nr, ne) / (rho_u * v_u ** 2)
        cs2 = np.fromfile(f, np.float64, nr * ne).reshape(nr, ne) / v_u ** 2
        logT = np.fromfile(f, np.float64, nr * ne).reshape(nr, ne)
        espT = np.fromfile(f, np.float64, nr * nT).reshape(nr, nT) / v_u ** 2
    return dict(nr=nr, ne=ne, nT=nT, lr0=lr0 - np.log10(rho_u), dlr=dlr, le0=le0 - np.log10(v_u ** 2), dle=dle,
                lT0=lT0, dlT=dlT, P=P, cs2=cs2, logT=logT, espT=espT)


def bilin(A, x0, dx, y0, dy, x, y):
    n1, n2 = A.shape
    fi = (x - x0) / dx; i = int(min(max(np.floor(fi), 0), n1 - 2)); ti = min(max(fi - i, 0.0), 1.0)
    fj = (y - y0) / dy; j = int(min(max(np.floor(fj), 0), n2 - 2)); tj = min(max(fj - j, 0.0), 1.0)
    return (1 - ti) * (1 - tj) * A[i, j] + ti * (1 - tj) * A[i + 1, j] + (1 - ti) * tj * A[i, j + 1] + ti * tj * A[i + 1, j + 1]


def esp_of(t, rho, T):  # code units
    return bilin(t["espT"], t["lr0"], t["dlr"], t["lT0"], t["dlT"], np.log10(rho), np.log10(T))


def P_of(t, rho, esp):
    return bilin(t["P"], t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(esp))


def cs2_of(t, rho, esp):
    return bilin(t["cs2"], t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(esp))


def T_of(t, rho, esp):
    return 10 ** bilin(t["logT"], t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(esp))
