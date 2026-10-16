#!/usr/bin/env python3
"""
Full protostellar-EOS table generator for AthenaPK (H2 dissociation + H ionization +
He/He+ ionization + H2 rotational/vibrational + inert-thermal He), in the shared FHC
code units.  Computes P, e_internal, c_s^2, T on a (log rho, log e_specific) grid for the
hydro path and e_specific(log rho, log T) for the RT path, and writes them to an HDF5 file
(for analysis) and to the flat binary <out>.bin that the runtime reads (src/eos/eos_table.hpp,
format src/eos/eos_table_format.hpp; v2, cgs).

Physics (per unit mass; gas = H mass-fraction X + He mass-fraction Y):
  species H2, HI, HII, e, HeI, HeII, HeIII solved from coupled Saha at (rho,T):
    H2 <-> 2 HI      K_d  = (pi m_H kT/h^2)^{3/2} 16 (Zrot_ns Zvib)^-1 exp(-chi_d/kT)
    HI <-> HII + e   K_H  = (2 pi m_e kT/h^2)^{3/2} exp(-chi_H/kT)         [gII/gI net 1]
    HeI<-> HeII+ e   K1   = 4 (2 pi m_e kT/h^2)^{3/2} exp(-chi_He1/kT)
    HeII<->HeIII+e   K2   = 1 (2 pi m_e kT/h^2)^{3/2} exp(-chi_He2/kT)
  charge neutrality  n_e = nHII + nHeII + 2 nHeIII ; nuclei conserved.
  internal energy (ref: cold ground-state H2 + neutral He):
    e = 3/2 kT n_tot  + nH2 * e_rotvib(T)  + chi_d*(nHI+nHII)/2
        + chi_H*nHII  + chi_He1*(nHeII+nHeIII) + chi_He2*nHeIII
  pressure P = n_tot k T.  c_s^2 = (dP/drho)_s = (dP/drho)_esp + (P/rho^2)(dP/desp)_rho, with the two
  derivatives taken by np.gradient of the tabulated P on the (log rho, log esp) grid (see build_table).

Validation prints gamma_eff(T) at a fixed density: ~7/5 warm-molecular, dip <4/3 at H2
dissociation (~2000 K) and again at H ionization (~1e4 K), ->5/3 atomic/ionized.
"""
import numpy as np

# ---- cgs constants ----
m_H = 1.6726219e-24; m_e = 9.1093837e-28; k_B = 1.380649e-16
h_pl = 6.62607015e-27; eV = 1.602176634e-12
chi_d   = 4.4781 * eV       # H2 dissociation
chi_H   = 13.598 * eV       # H ionization
chi_He1 = 24.587 * eV       # He  -> He+
chi_He2 = 54.418 * eV       # He+ -> He++
theta_rot = 85.4            # H2 rotational temperature [K]
theta_vib = 5987.0          # H2 vibrational temperature [K]

# ---- composition (mu=2.33 molecular: 1/mu = X/2 + Y/4) ----
X = 0.716; Y = 1.0 - X

# ---- FHC code units ----
# --- table file format (see eos_table_format.hpp; both must agree) ---
EOS_MAGIC = 0x454F535441424C31   # ASCII "EOSTABL1"; cannot collide with a legacy file, whose
                                 # first int64 is nr (a modest positive grid count)
EOS_VERSION = 2
EOS_FLAG_CGS = 1                 # bit 0: axes + arrays stored in cgs, converted at load

# rho0/v0 below set the code units the physics is COMPUTED in. They are rounded copies of
# scales the run derives exactly (units/be_normalization.hpp), which used to leak into the
# emitted file and bias every lookup by +0.00315 % (audit N14). Since v2 writes CGS and the
# loader converts with the run's own units, they no longer reach the output -- they only set
# the internal working scale, and the conversion back out is exact. Do not use them to
# normalize anything that is written.
rho0 = 5.467e-19            # g/cm^3
v0   = 1.9e4               # cm/s
e_unit = rho0 * v0 * v0    # erg/cm^3
esp_unit = e_unit / rho0   # erg/g  (specific-energy code unit = v0^2)

def lam3(T, m):            # (2 pi m k T / h^2)^{3/2}
    return (2.0*np.pi*m*k_B*T/(h_pl*h_pl))**1.5

def Zrot_ns(T):
    # Nuclear-spin-weighted H2 rotational partition function for the DISSOCIATION Saha:
    # para (even J, nuclear-spin weight 1) + ortho (odd J, weight 3), energies from the
    # para J=0 level (consistent with chi_d = D0 referenced to the v=0, J=0 state).
    # High-T limit: ~ 2*sum_J (2J+1)exp(-...) = 4 * [T/(2 theta_rot)] (spin 4 x sym nr 2).
    J = np.arange(0, 60)
    w = np.where(J % 2 == 0, 1.0, 3.0)
    x = J*(J+1)*theta_rot/np.maximum(T, 1.0)
    return np.sum(w*(2*J+1)*np.exp(-x))
def _erot_species(T, Jvals):    # rot energy per molecule of one nuclear-spin species [erg]
    # energy referenced to the species' OWN ground state (para: J=0, ortho: J=1) so
    # e_rot -> 0 as T -> 0 (the ortho J=1 zero-point is not dynamically available energy).
    E = (Jvals*(Jvals+1) - Jvals[0]*(Jvals[0]+1))*k_B*theta_rot
    x = Jvals*(Jvals+1)*theta_rot/np.maximum(T,1.0); w = (2*Jvals+1)*np.exp(-x)
    Z = np.sum(w); return np.sum(w*E)/Z if Z > 0 else 0.0
def erot_per_H2(T):        # rotational internal energy per molecule [erg]
    # FROZEN 3:1 ortho:para mixture (the ISM ratio; ortho<->para conversion is slow, so the
    # gas is NOT in rotational spin equilibrium). para = even J, ortho = odd J.
    Jeven = np.arange(0, 60, 2); Jodd = np.arange(1, 60, 2)
    return 0.25*_erot_species(T, Jeven) + 0.75*_erot_species(T, Jodd)
def Zvib(T):
    return 1.0/(1.0 - np.exp(-theta_vib/np.maximum(T,1.0)))
def evib_per_H2(T):        # harmonic vibrational energy per molecule (ref ground) [erg]
    # errstate only. Below ~4 K, theta_vib/T > 1500 and the exp overflows to +inf, giving
    # exactly 0.0 -- which is the correct T->0 limit (vibration frozen out), just noisy. The
    # 2026-08-18 downward axis extension goes to ~1.9 K and made every low-T build print a
    # wall of RuntimeWarnings. Suppressed rather than rewritten as exp(-x)/(1-exp(-x)):
    # that form is algebraically identical but differs in the last ulp at moderate x, which
    # would break the guarantee that an unpadded build reproduces the shipped table
    # bit-for-bit.
    with np.errstate(over='ignore'):
        return k_B*theta_vib/(np.exp(theta_vib/np.maximum(T,1.0)) - 1.0)

def solve_saha(rho, T):
    """Return per-volume number densities of all species + n_e at (rho,T)."""
    nH_tot  = rho * X / m_H          # H nuclei per cm^3
    nHe_tot = rho * Y / (4.0*m_H)
    le = lam3(T, m_e)
    # H2 <-> 2H law of mass action: n_H^2/n_H2 = (pi m_H kT/h^2)^{3/2}
    #   * g_H^2 / (g_el,H2 * Zrot_ns * Zvib) * exp(-chi_d/kT)
    # with g_H = 4 (2 electron-spin x 2 nuclear-spin), g_el(H2 X^1Sig_g+) = 1, and the
    # nuclear-spin statistics of H2 inside Zrot_ns (para w=1 / ortho w=3). Cross-checked:
    # gives the textbook ~8% dissociation of H2 at 3000 K, 1 atm (JANAF); dropping the
    # 16/(ortho-para weights) underestimates K_d by exactly 8x at high T.
    Kd = (np.pi*m_H*k_B*T/(h_pl*h_pl))**1.5 * 16.0/(Zrot_ns(T)*Zvib(T)) \
         * np.exp(-chi_d/(k_B*T))
    KH  = le * np.exp(-chi_H  /(k_B*T))
    K1  = 4.0*le*np.exp(-chi_He1/(k_B*T))
    K2  = 1.0*le*np.exp(-chi_He2/(k_B*T))
    # iterate on n_e (bisection in log space on neutrality residual)
    def neutrality(ne):
        ne = max(ne, 1e-300)
        # H side: 2 nHI^2/Kd + nHI(1 + KH/ne) - nH_tot = 0. Use the numerically-stable root
        # nHI = 2 nH_tot / (b + sqrt(b^2 + 8 nH_tot/Kd)) so that Kd->0 (cold) => nHI->0 (all
        # H2) instead of overflowing 2/Kd. (Kd underflowing to 0 gives 8nH_tot/Kd=inf ->nHI=0.)
        rH = KH/ne; b = 1.0 + rH
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            disc = np.sqrt(b*b + 8.0*nH_tot/Kd) if Kd > 0 else np.inf
            nHI = 2.0*nH_tot/(b + disc)
        if not np.isfinite(nHI): nHI = 0.0
        nHII = rH*nHI
        # He side
        r1 = K1/ne; r2 = K2/ne
        nHeI = nHe_tot/(1.0 + r1 + r1*r2)
        nHeII = r1*nHeI; nHeIII = r1*r2*nHeI
        ne_new = nHII + nHeII + 2.0*nHeIII
        return ne_new, (nHI, nHII, nHeI, nHeII, nHeIII)
    lo, hi = -40.0, np.log10(max(nH_tot+2*nHe_tot, 1e-30))+1.0
    for _ in range(200):
        mid = 0.5*(lo+hi); ne = 10.0**mid
        ne_new, _ = neutrality(ne)
        if ne_new > ne: lo = mid
        else: hi = mid
    ne = 10.0**(0.5*(lo+hi))
    _, (nHI, nHII, nHeI, nHeII, nHeIII) = neutrality(ne)
    nH2 = max((nH_tot - nHI - nHII)/2.0, 0.0)  # H-nucleus conservation (robust as Kd->0)
    return dict(H2=nH2, HI=nHI, HII=nHII, e=ne, HeI=nHeI, HeII=nHeII, HeIII=nHeIII)

def P_e_of_rho_T(rho, T):
    s = solve_saha(rho, T)
    n_tot = s['H2']+s['HI']+s['HII']+s['e']+s['HeI']+s['HeII']+s['HeIII']
    P = n_tot * k_B * T
    e = 1.5*k_B*T*n_tot
    e += s['H2']*(erot_per_H2(T) + evib_per_H2(T))
    e += chi_d*(s['HI']+s['HII'])/2.0
    e += chi_H*s['HII']
    e += chi_He1*(s['HeII']+s['HeIII']) + chi_He2*s['HeIII']
    return P, e   # erg/cm^3 (pressure), erg/cm^3 (internal energy density)

def build_table(path, nr=180, ne=220, nT=200,
                rho_phys_min=1e-20, rho_phys_max=1e0,
                T_min=8.0, T_max=3.0e5,
                esp_pad_pts=0, T_pad_pts=0, allow_esp_sat=False, T_pad_hi_pts=0,
                rho_pad_pts=0):
    """Generate the code-unit EOS table and write HDF5.
      hydro grid  (log10 rho_code, log10 esp_code) -> P_code, cs2_code, log10T
      RT grid     (log10 rho_code, log10 T)        -> esp_code
    esp = specific internal energy (erg/g -> code v0^2); P,e densities -> e_unit; cs2 -> v0^2.
    """
    import h5py
    v0sq = v0*v0
    rho_code = np.logspace(np.log10(rho_phys_min/rho0), np.log10(rho_phys_max/rho0), nr)
    # ---- DOWNWARD RHO EXTENSION (validation_tests/eos_tabulated E-F6, 2026-10-05) -----------
    # Below rho_phys_min the runtime used to clamp (eos_interp = p: P frozen) or continue P ~ rho at fixed esp
    # (eos_extrapolate) -- exact only where the composition does not depend on density; through H2 dissociation
    # (~1000 K) and H ionization (~3000-5000 K) the Saha fractions shift with rho and the continuation is off by
    # up to 20 % at 1e-24 g/cc (results/tables/EF56_validity.txt). rho_pad_pts PREPENDS rows at the SAME dlog rho,
    # so the original nr rows keep their exact nodes; the esp axis is derived from the ORIGINAL rows only (F-19:
    # deriving it from rows that include lower densities raises esp_max and rescales every esp node).
    # Default 0 -> bit-identical to every existing call.
    if rho_pad_pts > 0:
        lr0_ = np.log10(rho_code)
        dlr_ = lr0_[1] - lr0_[0]
        rho_code = np.concatenate([10.0**(lr0_[0] - dlr_*np.arange(rho_pad_pts, 0, -1)), rho_code])
    nr = rho_code.size
    rho_cgs  = rho_code*rho0

    # ---- DOWNWARD AXIS EXTENSION (2026-08-18) ------------------------------------------
    # WHY. The production run walks off the LOW esp edge of this table. Measured over 201
    # cycles of the production configuration at 256^3 (job 2531600): 41,650 edge-clamped EOS
    # lookups, 100 % of them `esp<` and zero on every other edge, rising from 21 per cycle to
    # 191 by cycle 200. From that run's own restart: 5 cells of 20,905,984 below the floor,
    # minimum esp = 1.0021 code against a floor of 1.1760, i.e. 0.069 decades under. Those
    # cells get P and c_s evaluated at the edge -- nearest-neighbour extrapolation, not
    # interpolation.
    #
    # The esp floor is not an independent knob: `esp_min` below is derived from
    # esp_rhoT.min(), which is set by T_min. At low T the gas is molecular and frozen out, so
    # esp = (3/2)kT/(mu m_H) exactly -- 8 K gives 4.25e8 erg/g = 1.176 in code units, which is
    # precisely the observed floor. Lowering the floor therefore means lowering T_min.
    #
    # HOW, so that this is an EXTENSION and not a new table. Naively lowering T_min with nT
    # fixed rescales the grid spacing, which moves EVERY node and changes every interpolated
    # value -- a result-changing change to physics that is already validated. Instead the
    # spacing is held fixed and points are PREPENDED: the original nT (and ne) nodes keep
    # their exact values and the new ones sit below them. The overlap is then reproducible to
    # round-off, which is the thing to verify rather than assert.
    #
    # esp_pad_pts / T_pad_pts default to 0, so every existing call is bit-identical.
    lT_base = np.log10(T_min); dlT = (np.log10(T_max) - lT_base)/(nT - 1)
    # AUDIT F-19 (2026-08-28) -- UPWARD EXTENSION AT FIXED SPACING.
    # Raising T_max with nT fixed RESCALES dlogT, which moves every node and changes every
    # interpolated value -- a result-changing rebuild of an already-validated table. That is
    # the same objection the downward T_pad_pts mechanism was written to answer, so the
    # upward extension works identically: hold dlT at its T_max value and APPEND nodes above
    # it. The original nT nodes then keep their exact values and the overlap with an
    # unextended build is reproducible to round-off.
    # Default 0 => bit-identical to every existing call.
    lT = np.linspace(lT_base, np.log10(T_max), nT)
    if T_pad_pts > 0:
        lT = np.concatenate([lT_base - dlT*np.arange(T_pad_pts, 0, -1), lT])
    if T_pad_hi_pts > 0:
        lT = np.concatenate([lT, np.log10(T_max) + dlT*np.arange(1, T_pad_hi_pts + 1)])
    Tgrid = 10.0**lT
    nT_tot = lT.size

    # --- (rho,T) tables: e_sp(rho,T) [code], P(rho,T) [code] ---
    esp_rhoT = np.zeros((nr, nT_tot)); P_rhoT = np.zeros((nr, nT_tot))
    for i, rc in enumerate(rho_cgs):
        for j, T in enumerate(Tgrid):
            P, ed = P_e_of_rho_T(rc, T)          # cgs: erg/cm^3
            esp_rhoT[i, j] = (ed/rc)/v0sq        # specific -> code
            P_rhoT[i, j]   = P/e_unit
    log10T = np.log10(Tgrid)

    # --- build hydro (rho, esp) grid by inverting e_sp(rho,T) per rho ---
    # The esp axis is derived from the UNPADDED temperature columns, so esp_min/esp_max --
    # and hence the base node values and the spacing dle -- are exactly what an unpadded
    # build produces. The pad is then prepended at that same spacing.
    # AUDIT F-19 (2026-08-28) -- THE ESP AXIS MUST BE EXTENDED, NOT RESCALED.
    #
    # The esp SPACING is set by the UNEXTENDED temperature window, exactly as before, so the
    # original ne nodes keep their exact values. The upward T extension raises esp_max (more
    # energy per gram at higher T), and that extra range is covered by APPENDING nodes at the
    # SAME dle -- never by stretching the original linspace over a wider interval, which would
    # move every node and silently change every interpolated value.
    #
    # Getting this wrong is easy and was caught in testing: deriving le_base from the
    # T-EXTENDED columns took log10(esp_max) from 5.79 to 7.55 at fixed ne, i.e. dle grew by
    # 31 % and the whole grid shifted. The overlap with an unextended build must be
    # reproducible to round-off; that is the property this block exists to preserve.
    hi0 = esp_rhoT.shape[1] - T_pad_hi_pts if T_pad_hi_pts > 0 else esp_rhoT.shape[1]
    base = esp_rhoT[rho_pad_pts:, T_pad_pts:hi0]   # original rows only (see the rho extension note)
    esp_min = base.min()*0.999; esp_max = base.max()*1.001
    le_base = np.linspace(np.log10(esp_min), np.log10(esp_max), ne)
    dle_base = le_base[1] - le_base[0]
    le_all = le_base
    if esp_pad_pts > 0:
        le_all = np.concatenate([le_base[0] - dle_base*np.arange(esp_pad_pts, 0, -1), le_all])
    if T_pad_hi_pts > 0:
        # cover esp up to the EXTENDED maximum, at the unchanged spacing
        esp_hi = esp_rhoT.max()*1.001
        n_up = int(np.ceil((np.log10(esp_hi) - le_base[-1]) / dle_base))
        if n_up > 0:
            le_all = np.concatenate([le_all, le_base[-1] + dle_base*np.arange(1, n_up + 1)])
    esp_code = 10.0**le_all
    ne_tot = le_all.size
    # The padded esp nodes must lie INSIDE the (rho,T) coverage, or np.interp below clamps
    # them to log10T[0] and the extension buys nothing. Fail loudly rather than ship a table
    # whose new region is a flat extrapolation.
    if esp_pad_pts > 0 and esp_code[0] < esp_rhoT.min():
        raise SystemExit(
            "T_pad_pts=%d is not enough for esp_pad_pts=%d: the lowest esp node %.4e erg/g is "
            "below the coldest tabulated esp %.4e erg/g, so it would be edge-clamped. Increase "
            "T_pad_pts." % (T_pad_pts, esp_pad_pts, esp_code[0]*v0sq, esp_rhoT.min()*v0sq))
    # AUDIT F-19/F-37 (2026-08-28) -- THE ESP-SATURATION REPORT.
    #
    # TWO WRONG VERSIONS OF THIS CHECK PRECEDED THIS ONE; both are recorded so neither is
    # reintroduced.
    #   (1) `esp_code[-1] > esp_rhoT[i].max()` -- true whenever even the SINGLE top node is
    #       clamped, i.e. nearly every row at every resolution (fired on 40 of 40 rows of a
    #       test build where the shipped table has 45 of 400). A single clamped node still
    #       sits above its neighbour, so P stays monotone and EspFromP is fine.
    #   (2) the same test as a hard SystemExit -- which can NEVER pass, for a reason that is
    #       geometric rather than numerical: the (rho, esp) grid is RECTANGULAR and the
    #       physical domain is not. At fixed rho, esp rises monotonically with T, so
    #       esp_max(rho) = esp(rho, T_max), and the esp AXIS must span the GLOBAL maximum,
    #       which occurs at the LOWEST rho (most ionized, most energy per gram). Every
    #       higher-rho row therefore has top esp nodes above its own esp(T_max), for ANY
    #       T_max. Those nodes simply mean "hotter than this table goes" -- the T ceiling
    #       seen through the esp axis, exactly as rho > rho_max is the density ceiling.
    #
    # So the artifact CANNOT be removed by any rectangular table; it can only be pushed to
    # higher temperature. What matters is therefore reported, not asserted:
    #   * WHERE the flat region starts (it always starts at T_max, by the argument above), and
    #   * HOW WIDE the flat runs are, since a wider flat interval makes EspFromP's bisection
    #     ambiguous over a larger range of target pressures.
    # Raising T_max shortens the runs AND moves them away from the physics: going 3.0e5 ->
    # 1.0e6 K took the widest run from 37 esp nodes to 10.
    #
    # The consumer-side alarm for this is already live and is the one that matters: the
    # per-cycle EOS clamp report (hydro.cpp PreStepMeshUserWorkInLoop) counts real CELL-STATE
    # lookups that leave the table, so a run that actually reaches the ceiling says so.
    nsat = np.zeros(nr, dtype=int)
    for i in range(nr):
        nsat[i] = int(np.count_nonzero(esp_code > esp_rhoT[i].max()))
    bad = np.where(nsat >= 2)[0]
    if len(bad):
        T_top = 10.0**lT[-1]
        msg = ("EOS esp-axis saturation: %d of %d density rows (rho = %.3e..%.3e g/cm^3) have "
               ">= 2 esp nodes above their own esp(T_top = %.4g K); widest flat run %d of %d "
               "esp nodes. P is FLAT in esp there, so EspFromP's bisection is ambiguous for a "
               "target pressure inside that interval. This is the T ceiling seen through a "
               "RECTANGULAR (rho,esp) grid and cannot be removed -- only raised. Raise T_max "
               "(via T_pad_hi_pts, which extends at fixed spacing) if the run approaches %.4g K."
               % (len(bad), nr, rho_cgs[bad[0]], rho_cgs[bad[-1]], T_top,
                  int(nsat.max()), ne_tot, T_top))
        if allow_esp_sat:
            print("### NOTE " + msg)
        else:
            print("### WARNING " + msg)
    P_re   = np.zeros((nr, ne_tot)); logT_re = np.zeros((nr, ne_tot))
    for i in range(nr):
        # monotone e_sp(T) at this rho -> interpolate T(esp), then P
        Tof = np.interp(np.log10(esp_code), np.log10(esp_rhoT[i]), log10T,
                        left=log10T[0], right=log10T[-1])
        logT_re[i] = Tof
        Pof = np.interp(Tof, log10T, P_rhoT[i])
        P_re[i] = Pof
    # adiabatic cs^2 = (dP/drho)_esp + (P/rho^2)(dP/desp)_rho  [all code units]
    # (standard identity with esp = SPECIFIC internal energy: from de = Tds + (P/rho^2)drho,
    #  an isentrope has desp = (P/rho^2) drho, so cs^2 = P_rho|esp + (P/rho^2) P_esp|rho.
    #  Ideal-gas check: P=(g-1)rho*esp -> P_rho=(g-1)esp, P_esp=(g-1)rho
    #  -> cs^2=(g-1)esp + (g-1)^2 esp = g(g-1)esp = gP/rho. An earlier version divided the
    #  second derivative by an extra rho, which is only correct at rho_code = 1.)
    lr = np.log10(rho_code); le = np.log10(esp_code)
    dP_dlr = np.gradient(P_re, lr, axis=0)      # dP/dlog10rho
    dP_dle = np.gradient(P_re, le, axis=1)
    RHO = rho_code[:, None]; ESP = esp_code[None, :]
    dP_drho_e = dP_dlr/(np.log(10)*RHO)          # dP/drho at fixed esp
    dP_desp   = dP_dle/(np.log(10)*ESP)          # dP/desp at fixed rho (esp SPECIFIC)
    P_over_rho2 = P_re/(RHO*RHO)
    cs2_re = dP_drho_e + P_over_rho2*dP_desp
    cs2_re = np.maximum(cs2_re, 1e-12)

    with h5py.File(path, "w") as f:
        f.attrs["rho0_cgs"]=rho0; f.attrs["v0_cgs"]=v0; f.attrs["e_unit_cgs"]=e_unit
        f.attrs["X"]=X; f.attrs["Y"]=Y
        f.create_dataset("log10_rho_code", data=lr)
        f.create_dataset("log10_esp_code", data=le)
        f.create_dataset("P_code",   data=P_re)
        f.create_dataset("cs2_code", data=cs2_re)
        f.create_dataset("log10T",   data=logT_re)
        f.create_dataset("log10_T_grid", data=log10T)
        f.create_dataset("esp_code_of_rhoT", data=esp_rhoT)
    # flat binary for the C++ device loader (no HDF5-C-API dependency).
    #
    # AUDIT N14 (2026-08-05). Format v2. The v1 file was headerless and stored everything in
    # CODE units built from this script's own rounded rho0/v0, which the loader had no way to
    # detect -- there was not one spare byte in the file for a unit. v2 stamps a magic +
    # version + flags header and stores everything in CGS; the loader converts with the
    # RUNNING simulation's units, so the file is correct for any normalization.
    #
    # header: int64 [MAGIC, VERSION, FLAGS, nr, ne, nT]
    #         double [lr0, dlr, le0, dle, lT0, dlT, gen_rho0, gen_v0]
    #           axes in CGS: log10(rho_cgs), log10(esp_cgs [erg/g]), log10(T[K])
    #           gen_* are provenance and are 0 for a cgs table -- it assumes no units
    # payload: row-major float64 P[nr,ne] (erg/cm^3), cs2[nr,ne] (cm^2/s^2),
    #          logT[nr,ne] (log10 K, unit-free), esp_rhoT[nr,nT] (erg/g).
    #
    # The physics is computed in code units exactly as before and converted only here, so a
    # v2 file is the v1 file times exact unit factors and nothing else.
    esp_unit = v0 * v0                 # erg/g
    binpath = path.replace(".h5", ".bin")
    with open(binpath, "wb") as fb:
        np.array([EOS_MAGIC, EOS_VERSION, EOS_FLAG_CGS, nr, ne_tot, nT_tot], dtype=np.int64).tofile(fb)
        np.array([lr[0] + np.log10(rho0), lr[1]-lr[0],
                  le[0] + np.log10(esp_unit), le[1]-le[0],
                  log10T[0], log10T[1]-log10T[0],
                  0.0, 0.0], dtype=np.float64).tofile(fb)
        (P_re * e_unit).astype(np.float64).tofile(fb)
        (cs2_re * esp_unit).astype(np.float64).tofile(fb)
        logT_re.astype(np.float64).tofile(fb)          # log10 T[K]: unit-free
        (esp_rhoT * esp_unit).astype(np.float64).tofile(fb)
    print("wrote %s (%d bytes)" % (binpath, __import__('os').path.getsize(binpath)))
    print("wrote %s  hydro grid %dx%d  RT grid %dx%d  (esp_pad=%d, T_pad=%d)"
          % (path, nr, ne_tot, nr, nT_tot, esp_pad_pts, T_pad_pts))
    print("  log10 rho_code in [%.2f, %.2f]  log10 esp_code in [%.2f, %.2f]" %
          (lr[0], lr[-1], le[0], le[-1]))
    print("  cs2_code range [%.3e, %.3e]  (any nonpos: %d)" %
          (cs2_re.min(), cs2_re.max(), int(np.sum(cs2_re<=0))))
    return path

if __name__ == "__main__":
    import os, sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # `build`                       -> the original low-resolution table (180x220x200) as
        #   src/eos/eos_table.h5/.bin. That table is NOT shipped any more; the shipped tables and their exact build
        #   commands are in TABLE_PROVENANCE.md. Do not change these defaults (they define that legacy table).
        # `build <nr> <ne> <nT> [out.h5]` -> a higher-resolution table at an explicit path.
        #   Use this for the Phase-3 kink-fidelity table (e.g. 400 1000 920 -> ~0.9% worst at
        #   the H2-dissociation/H-ionization cusps vs ~6.9% at shipped res). Swapping it into
        #   production is a result-changing, runtime-loaded change -> user-gated.
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eos_table.h5")
        # `build <nr> <ne> <nT> [out.h5] [esp_pad_pts] [T_pad_pts]`
        #   The two pads PREPEND nodes below the existing axes at the SAME spacing, so the
        #   original nr/ne/nT nodes keep their exact values and the overlap with an unpadded
        #   build is reproducible to round-off. Both default to 0 -> bit-identical output.
        if len(sys.argv) >= 5:
            nr, ne, nT = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
            if len(sys.argv) >= 6:
                out = sys.argv[5]
            epad = int(sys.argv[6]) if len(sys.argv) >= 7 else 0
            tpad = int(sys.argv[7]) if len(sys.argv) >= 8 else 0
            # AUDIT F-19/F-37 (2026-08-28): T_max and the saturation override are now CLI
            # arguments. T_max was hardcoded at 3.0e5 K, which saturates the forward grid on
            # the high-density rows (the flat-P region that breaks EspFromP's monotone
            # inversion). Raising it is the fix; --allow-sat ships a saturated table anyway.
            tmax = float(sys.argv[8]) if len(sys.argv) >= 9 else 3.0e5
            allow = (len(sys.argv) >= 10 and sys.argv[9] in ("1", "true", "allow"))
            thipad = int(sys.argv[10]) if len(sys.argv) >= 11 else 0
            rpad = int(sys.argv[11]) if len(sys.argv) >= 12 else 0
            build_table(out, nr=nr, ne=ne, nT=nT, esp_pad_pts=epad, T_pad_pts=tpad,
                        T_max=tmax, allow_esp_sat=allow, T_pad_hi_pts=thipad, rho_pad_pts=rpad)
        else:
            build_table(out)
        sys.exit(0)
    # quick validation: gamma_eff(T) at fixed rho showing the two softening dips
    print("# code units: esp_unit(v0^2)=%.4e erg/g, e_unit=%.4e, rho0=%.3e" %
          (esp_unit, e_unit, rho0))
    for rho in [1e-10, 1e-6, 1e-2]:
        print("\n# rho=%.0e g/cm^3 : T[K]  x_H2  x_HII  gamma_eff  cs[cm/s]" % rho)
        for T in [50,100,300,1000,2000,3000,6000,1e4,2e4,5e4,1e5]:
            P1,e1 = P_e_of_rho_T(rho, T*0.999)
            P2,e2 = P_e_of_rho_T(rho, T*1.001)
            # gamma_eff = 1 + P/e_int ; and dlnP/dlnrho|_s ~ use (dP/de)_rho*(de) ... report 1+P/e
            P,e = P_e_of_rho_T(rho, T)
            s = solve_saha(rho,T)
            nHt = rho*X/m_H
            xH2 = 2*s['H2']/nHt; xHII = s['HII']/nHt
            gam = 1.0 + P/e
            cs = np.sqrt(P/rho*gam)  # rough
            print("  %8.0f  %.3f  %.3e  %.4f  %.3e" % (T, xH2, xHII, gam, cs))
