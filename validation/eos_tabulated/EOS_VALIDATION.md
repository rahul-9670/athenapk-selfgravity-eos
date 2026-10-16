# Tabulated equation of state in AthenaPK: implementation guide and validation report (paper I)

Version 2026-10-16 (public edition). Folder `validation/eos_tabulated/`. This document is the authoritative report for
the EOS module; `README.md` is the index of decks, scripts and results.

*Public edition.* The validation was carried out in a larger private research code from which this repository was
extracted (self-gravity + tabulated EOS only). Passages about the private development history (commit and binary
hashes, the adoption into the production runs), about physics modules that are not part of this repository, and
about compute systems have been removed or generalised; every test result, table and figure of the EOS tests is
unchanged. Hashes that remain identify the binaries recorded in each run's `cmd.txt`.

**How this document is organised.** It reads from the physics to the evidence:

* **Part A — the EOS from physics to code** (added 2026-10-16): why the solver needs an EOS (A1), the chain from the
  offline generator to the device lookups (A2), the Saha microphysics with derivations (A3), how the table is built
  and stored (A4), units (A5), the runtime execution path (A6), every lookup function (A7), how the answers enter
  ConsToPrim, the floors and the Riemann solver (A8), the numerical error budget (A9), all configuration keys (A10),
  three worked examples computed by the real code (A11) and a one-paragraph summary (A12).
* **Part B — validation** (Sec. 1–14, unchanged in substance; the section numbers and finding IDs E-F1 … E-F11 that
  other documents cite are kept): the tests E1–E10, their results, the findings and the reproduction map.
* **Part C — reading the validation** (added 2026-10-16): what each test establishes and what it does not (C1),
  limitations and untested regimes (C2), implementation observations O-1 … O-7 found while writing Part A and fixed
  (C3), proposed tests that have not been run (C4), and the source reference index with permalinks (C5).

Readers who want the results first can read Sec. 0 and then Part B; Part A explains what those results are results
*of*.

---

## 0. Summary

The tabulated protostellar EOS (`eos = hydrogen`, table `eos_table_hires_v5_cold.bin`) was validated in the
configuration used by the production collapse runs, from the table file itself (provenance, thermodynamic
consistency, inversions) through its coupling to the MHD solver (sound waves, general-EOS Riemann problems against an
exact solver, adiabatic compression through H2 dissociation and H ionization) to the GPU path and the behaviour at
the table edges and floors.

| group | tests | verdict |
|---|---|---|
| A. The table | E1 provenance, E2 thermodynamic gates, E3 asymptotic limits, E5 inversions | PASS. Byte-identical rebuild; Saha consistency 0.16 % median; P strictly increasing in e below the 3e5 K cap; P -> e inversion error = its analytic bound 8.08e-6; T -> e -> T round trip 2e-5 in the production footprint, up to 1.3 % across dissociation/ionization (E-F7) |
| B. EOS + hydro | E6 acoustic waves, E7 Riemann problems (4 problems x 6 resolutions x 2 interpolation modes), E8 compression legs (5 legs, 10 K -> 12 500 K) | PASS. Waves travel at the interpolant's sound speed (E-F1, fixed); shocks converge to the exact general-EOS solution (R1, R3: plateaus to 1e-7); rarefactions through recombination converge slowly because no conservative scheme keeps a moving contact between EOS regimes in pressure equilibrium (Sec. 6.3); the opt-in double flux restores ~1e-6 plateaus at a measured energy cost (Sec. 6.4); compression follows the exact adiabat to <= 4e-7 after removing a known geometric term |
| C. GPU | E9 table lookups, waves, Riemann, compression, floor probe on H100 | PASS. Lookups = CPU to <= 3.3e-15; smooth flows to 1e-14 .. 5e-10; Riemann L1 differences 3-4 orders below the discretisation error |
| D. Edges and floors | E10 footprint of the production runs, floor-visibility probe | Table covers the production footprint. The pressure floor was inert and floored cells invisible (E-F5) and `eos_interp = p` mis-extrapolated below the density edge (E-F6): both FIXED (`hydro/eos_extrapolate`, default on), with a physical temperature floor (`hydro/Tfloor`, counted) and a table extended to 1e-26 g/cc (`eos_table_hires_v5_cold_rho26.bin`); a legacy build keeps an existing run bit-continuous (E-F9) |

Findings: **E-F1** P interpolation sound-speed sawtooth (FIXED, default); **E-F2** the shipped consistency
test could not read v2 tables (fixed in the copy); **E-F3** `dedner_extended` segfaults in 1D; **E-F4**
`hydro/cons_diag` segfaults in 1D; **E-F5** pressure floor inert, floored cells uncounted (FIXED); **E-F6** density
extrapolation below the table with `eos_interp = p` (FIXED); **E-F7** T -> e -> T round trip up to 1.3 %; **E-F8** 1D
meshblocks of 400 cells exceed the CUDA team-scratch limit; **E-F9** a runtime key cannot keep an existing run
bit-continuous when the inline lookups change (resolved with a compile-time legacy build); **E-F10** the slow R2/R4 convergence is
the spurious pressure of a conservative update at a moving contact between EOS regimes (Abgrall 1996), not the
table; fixed WITH PLM by an opt-in double-flux energy update (`hydro/double_flux`; contacts pressure-exact in
1D/2D/3D, CPU/GPU, refined meshes; plateaus 1e-6; energy error converging; opt-in).
Test-infrastructure traps in Sec. 12.

---

# Part A — The EOS from physics to code

Part A is a self-contained guide to the equation of state as it is implemented in this repository: the physics
that the table encodes, how the table is built, how the running code reads it, and how its answers enter the MHD
solver. Every implementation statement links to the source at commit `5649acd`, which contains the fixes of the
observations O-1 … O-7 (Part C3); where the earlier behaviour (commit `a40eff5`) differs, the text says so. Statements are labelled as follows when the label
matters: **[code]** read directly from the source; **[measured]** a number produced by a program, with the file
that holds it; **[derived]** a hand derivation; **[theory]** standard textbook physics that the code does not
itself establish; **[inferred]** a conclusion drawn from the code that no test isolates.

Notation used throughout (all symbols are re-defined where they first matter):

| symbol | meaning | unit (code / cgs) |
|---|---|---|
| $\rho$ | mass density | $\rho_u$ / g cm$^{-3}$ |
| $\mathbf v$, $\mathbf m=\rho\mathbf v$ | velocity, momentum density | $v_u$, $\rho_u v_u$ |
| $\mathbf B$ | magnetic field, Heaviside-Lorentz ($P_{\rm mag}=B^2/2$) | $\sqrt{\rho_u}\,v_u$ (HL) / $\sqrt{4\pi\rho_u}\,v_u$ G |
| $E$ | total energy density $=\rho\varepsilon+\tfrac12\rho v^2+\tfrac12B^2$ | $\rho_u v_u^2$ / erg cm$^{-3}$ |
| $\varepsilon$ (`esp` in the code) | **specific** internal energy | $v_u^2$ / erg g$^{-1}$ |
| $e_{\rm int}=\rho\varepsilon$ (`eint`) | internal-energy **density** | $\rho_u v_u^2$ |
| $P$ | gas pressure | $\rho_u v_u^2$ / erg cm$^{-3}$ |
| $T$ | temperature | **kelvin** in the table; $T_{\rm code}=T/T_u$ elsewhere |
| $c_s^2=(\partial P/\partial\rho)_s$ (`cs2`) | adiabatic sound speed squared | $v_u^2$ |
| $\Gamma_1=\rho c_s^2/P$ | first adiabatic index | — |
| $\mu$ | mean mass per free particle in units of $m_H$: $\rho=\mu m_H n_{\rm tot}$ | — |
| $\rho_u, v_u, T_u$ | code units: $5.467\times10^{-19}$ g cm$^{-3}$, $1.9\times10^4$ cm s$^{-1}$, 10.015 K (Sec. A5) | |

## A1. Why the simulation needs an equation of state

AthenaPK evolves the conserved variables of ideal (or non-ideal) MHD in each cell,
$U=(\rho,\ \mathbf m,\ E,\ \mathbf B,\ \psi)$, with $\psi$ the GLM divergence-cleaning scalar. These equations are
not closed: the flux of momentum contains the gas pressure $P$, the flux of energy contains $(E+P+B^2/2)\mathbf v$,
and the speed at which information travels — which sets both the Riemann-solver wave speeds and the time step —
contains the sound speed. The conserved variables only provide the internal energy, as the difference

```math
e_{\rm int} \;=\; E-\tfrac12\frac{|\mathbf m|^2}{\rho}-\tfrac12|\mathbf B|^2 ,\qquad \varepsilon=e_{\rm int}/\rho .
```

The equation of state (EOS) is the closure that turns $(\rho,\varepsilon)$ into $P$ and $c_s$, and — because the
solver reconstructs **primitive** variables $(\rho,\mathbf v,P,\mathbf B)$ at cell faces — also the inverse map
$(\rho,P)\mapsto\varepsilon$. Other physics needs the temperature: the radiation, dust and non-ideal-MHD packages,
the Jeans refinement criterion and the problem generators that set initial conditions from a temperature.

So the EOS must answer five questions, each of which appears as a function in the code:

| question | function ([code]) | main consumers |
|---|---|---|
| $P(\rho,e_{\rm int})$ | [`EosTable::PresFromRhoEint`][tab-iface] → [`Pof`][tab-pof] | [ConsToPrim][c2p-p] (every cell, every stage) |
| $e_{\rm int}(\rho,P)$ | [`EintFromRhoPres`][tab-iface] → [`EspFromP`][tab-espfromp] | [HLLD face energies][hlld-e], [pressure floor][c2p-pfl], double flux |
| $c_s^2(\rho,P)$ | [`AsqFromRhoPres`][tab-iface] → `EspFromP` + [`Cs2of`][tab-pof] | [fast magnetosonic speed][eos-cf] → HLLD wave speeds, [time step][dt-hyp], [Jeans refinement][jeans] |
| $T(\rho,e_{\rm int})$, $T(\rho,P)$ | [`TemperatureK`][tab-iface], [`TemperatureKFromPres`][tab-tfromp] | radiation, dust, non-ideal MHD coefficients |
| $e_{\rm int}(\rho,T)$ | [`EintFromRhoTk`][tab-eintT] (+ host twin) | initial conditions, `hydro/Tfloor`, the `collapse_be` barotropic step, radiation matter coupling |

Two EOS are implemented behind one interface, selected by `<hydro> eos` ([code][hydro-eos]):

* `eos = adiabatic` — the ideal gas $P=(\gamma-1)\rho\varepsilon$ with constant $\gamma$ (`hydro/gamma`);
* `eos = hydrogen` — the **tabulated protostellar EOS**: a hydrogen–helium mixture in local thermodynamic and
  chemical equilibrium (Saha) with H$_2$ rotation/vibration, H$_2$ dissociation, H and He ionization. This is the EOS
  this document is about.

Why a table? A collapsing protostellar core passes from cold molecular gas ($\Gamma_1=5/3$ below ~50 K, where
H$_2$ rotation is frozen; ~7/5 when it is excited) through H$_2$ dissociation near 2000 K, where most of the
compressional work goes into breaking molecules ($\Gamma_1$ falls to ~1.1, which is what triggers the second
collapse), and through H ionization near $10^4$ K. Solving the coupled Saha equations in every cell at every stage
would be far too expensive, so the equilibrium state is computed once, offline, on a grid, and the running code
interpolates.

## A2. The chain of computations

The EOS is not one function but a chain that starts long before the simulation does:

```
 OFFLINE (python)                         LOAD (host, once)                   RUN (device kernels, every stage)
 src/eos/gen_eos_table.py                 src/eos/eos_table.hpp: Load()       src/eos/eos_table.hpp lookups
 ───────────────────────                  ──────────────────────────          ───────────────────────────────
 Saha at (rho,T) grid ──► P, e(rho,T)     read header (v1 / v2 cgs)           Pof, Cs2of, LogTof   (rho,esp) grid
 invert e(T) per rho ──► T, P on           convert cgs → code units            EspFromP (bisection)
   a (rho, esp) grid                        with THIS run's units              EintFromRhoTk        (rho,T) grid
 cs2 = P_rho|e + P/rho^2 P_e|rho           optional P → P/rho_node             ideal-gas continuation outside
 write .bin (cgs) ───────────────────────► allocate Views + host mirrors       clamp counters
                                                     │                                │
 src/hydro/hydro.cpp Initialize(): read keys ────────┘                                ▼
   → AdiabaticGLMMHDEOS(floors, gamma, use_h2diss, EosTable, ...)   ConsToPrim · HLLD · time step · floors ·
     stored as the Hydro package Param "eos"                         collapse_be · radiation · dust · diffusion
```

Each box is one of the sections below: A3 the physics of the generator, A4 how the table is built and stored, A5
the units, A6 the runtime execution path, A7 the lookups function by function, A8 how the answers are used by
the solver, A9 the numerics and their error budget, A10 the configuration keys, A11 worked examples computed by
the real code.

## A3. The microphysics model (what the table contains)

All of this section is implemented in [`src/eos/gen_eos_table.py`][gen] and runs only when a table is built.

### A3.1 Assumptions

The generator models the gas as follows ([code][gen-head]; the limits of validity are discussed in A3.7):

1. **Composition.** Hydrogen mass fraction $X=0.716$, helium $Y=1-X=0.284$, no metals ([code][gen-const]). For fully
   molecular gas $1/\mu = X/2+Y/4$, i.e. $\mu=2.331$.
2. **Species.** H$_2$, H, H$^+$, e$^-$, He, He$^+$, He$^{2+}$. No H$^-$, H$_2^+$, H$_3^+$, molecules other than H$_2$, or
   dust.
3. **Equilibrium.** Chemical (Saha) equilibrium at the local $(\rho,T)$ — no reaction rates, no memory.
4. **Ideal, non-degenerate particles.** Each species is a classical ideal gas; no Coulomb (plasma) corrections, no
   electron degeneracy, no pressure ionization, no radiation pressure.
5. **Internal structure.** H$_2$: rigid-rotor rotation ($\theta_{\rm rot}=85.4$ K, levels $J<60$) and harmonic
   vibration ($\theta_{\rm vib}=5987$ K). Atoms: ground state only (statistical weights, no excited levels).
6. **Ortho/para H$_2$.** For the *energy*, a frozen 3:1 ortho:para mixture (ortho–para conversion is slow in the
   ISM); for the *dissociation equilibrium*, the nuclear-spin-weighted equilibrium partition function (A3.3, A3.5).

### A3.2 Pressure

For a mixture of ideal gases the partial pressures add (Dalton), and each species contributes $n_ik_BT$:

```math
P = n_{\rm tot}\,k_B T,\qquad n_{\rm tot}=n_{\rm H_2}+n_{\rm H}+n_{\rm H^+}+n_e+n_{\rm He}+n_{\rm He^+}+n_{\rm He^{2+}} .
```

Defining $\mu$ by $\rho=\mu m_H n_{\rm tot}$, this is $P=\rho k_BT/(\mu m_H)$: the familiar ideal-gas law, except that
$\mu$ depends on $(\rho,T)$ through the composition — 2.331 molecular, ~1.27 atomic, ~0.6 ionized. That is why the
code never uses $T_{\rm code}=P/\rho$ under `eos = hydrogen` (it calls [`TemperatureKFromPres`][tab-tfromp]
instead; see the comment there). Implemented in [`P_e_of_rho_T`][gen-pe].

### A3.3 Chemical equilibrium: the Saha equations

**Derivation [theory].** For a reaction $A\rightleftharpoons B+C$ among ideal gases in equilibrium, the chemical
potentials balance, $\mu_A=\mu_B+\mu_C$. For a classical ideal gas
$\mu_i = -k_BT\ln\!\big[(z_i/n_i)\,(2\pi m_ik_BT/h^2)^{3/2}\big]+E_{0,i}$, where $z_i$ is the internal partition
function and $E_{0,i}$ the energy of the internal ground state. Exponentiating the balance gives the law of mass action

```math
\frac{n_Bn_C}{n_A}=\Big(\frac{2\pi\,m_Bm_C}{m_A}\,\frac{k_BT}{h^2}\Big)^{3/2}\,\frac{z_Bz_C}{z_A}\,
\exp\!\Big(-\frac{\chi}{k_BT}\Big),\qquad \chi=E_{0,B}+E_{0,C}-E_{0,A}.
```

The generator uses four such laws ([code][gen-saha]; constants $\chi_d=4.4781$ eV, $\chi_{\rm H}=13.598$ eV,
$\chi_{\rm He}=24.587$ eV, $\chi_{\rm He^+}=54.418$ eV [code][gen-const]):

* **H$_2$ dissociation**, H$_2\rightleftharpoons$ 2H. The reduced mass factor is $m_H m_H/(2m_H)=m_H/2$, so the
  translational factor is $(\pi m_Hk_BT/h^2)^{3/2}$. Each H atom has $z_{\rm H}=4$ (electron spin 2 × nuclear spin 2);
  the molecule has electronic weight 1 and its nuclear-spin statistics inside the rotational sum:

```math
K_d \equiv \frac{n_{\rm H}^2}{n_{\rm H_2}} = \Big(\frac{\pi m_Hk_BT}{h^2}\Big)^{3/2}
\frac{16}{Z_{\rm rot}^{\rm ns}(T)\,Z_{\rm vib}(T)}\,e^{-\chi_d/k_BT},
\quad Z_{\rm rot}^{\rm ns}=\sum_{J}w_J(2J+1)e^{-J(J+1)\theta_{\rm rot}/T},\ w_J=\begin{cases}1&J\ \text{even}\\3&J\ \text{odd}\end{cases},
\quad Z_{\rm vib}=\frac{1}{1-e^{-\theta_{\rm vib}/T}} .
```

$\chi_d=D_0$ is measured from the $v=0$, $J=0$ level, consistent with both sums starting at zero energy
([`Zrot_ns`][gen-zrot], [`Zvib`][gen-zvib]). The code comment records a check against the textbook ~8 %
dissociation of H$_2$ at 3000 K and 1 atm.

* **H ionization**, H$\rightleftharpoons$ H$^+$ + e$^-$: $m_Hm_e/m_H\to m_e$, weights $z_{\rm H^+}z_e/z_{\rm H}=1\cdot2/2=1$:
  $K_{\rm H}=n_{\rm H^+}n_e/n_{\rm H}=(2\pi m_ek_BT/h^2)^{3/2}e^{-\chi_{\rm H}/k_BT}$.
* **He ionization**: $K_1=4\,(2\pi m_ek_BT/h^2)^{3/2}e^{-\chi_{\rm He}/k_BT}$ (weights $2\cdot2/1$) and
  $K_2=(2\pi m_ek_BT/h^2)^{3/2}e^{-\chi_{\rm He^+}/k_BT}$ (weights $1\cdot2/2$).

**Closure.** Two conservation laws and charge neutrality close the system:

```math
2n_{\rm H_2}+n_{\rm H}+n_{\rm H^+}=n_{\rm H,tot}=\frac{X\rho}{m_H},\qquad
n_{\rm He}+n_{\rm He^+}+n_{\rm He^{2+}}=\frac{Y\rho}{4m_H},\qquad
n_e=n_{\rm H^+}+n_{\rm He^+}+2n_{\rm He^{2+}} .
```

### A3.4 How the generator solves the equilibrium

For a trial electron density $n_e$, every other density follows in closed form ([`neutrality`][gen-neut]):

* Hydrogen. With $n_{\rm H_2}=n_{\rm H}^2/K_d$ and $n_{\rm H^+}=n_{\rm H}K_{\rm H}/n_e$, H conservation becomes the
  quadratic $2n_{\rm H}^2/K_d+n_{\rm H}(1+K_{\rm H}/n_e)-n_{\rm H,tot}=0$. Its positive root is evaluated in the
  rationalised form

```math
n_{\rm H}=\frac{2\,n_{\rm H,tot}}{b+\sqrt{b^2+8n_{\rm H,tot}/K_d}},\qquad b=1+K_{\rm H}/n_e ,
```

which is the textbook root multiplied by $(b+\sqrt{\cdot})/(b+\sqrt{\cdot})$. It avoids the cancellation of
$-b+\sqrt{b^2+\dots}$ and has the right cold limit: as $K_d\to0$ the square root diverges and $n_{\rm H}\to0$
(all H in H$_2$) instead of overflowing $2/K_d$. $n_{\rm H_2}$ is then taken from nucleus conservation,
$n_{\rm H_2}=\max\big((n_{\rm H,tot}-n_{\rm H}-n_{\rm H^+})/2,0\big)$ ([code][gen-nh2]).

* Helium. $n_{\rm He}=n_{\rm He,tot}/(1+r_1+r_1r_2)$ with $r_{1,2}=K_{1,2}/n_e$.

These give a new electron density $n_e'=n_{\rm H^+}+n_{\rm He^+}+2n_{\rm He^{2+}}$. The equilibrium is the fixed
point $n_e'=n_e$; because $n_e'(n_e)$ decreases as $n_e$ grows, the residual $n_e'-n_e$ changes sign once and the
generator brackets it by **bisection in $\log_{10}n_e$** over $[-40,\ \log_{10}(n_{\rm H,tot}+2n_{\rm He,tot})+1]$
with a fixed 200 halvings ([code][gen-bisect]). 200 halvings of a bracket of a few tens of decades are far below double
precision, so the result is converged to round-off. One consequence of the bracket: a fully neutral state returns
$n_e=10^{-40}$ cm$^{-3}$ with zero ions — an absolute residual of $10^{-40}$, harmless ([measured]: W1 in
`results/tables/W_worked_saha.txt`, $|n_e-\sum Z n_{\rm ion}|/n_{\rm H}=4\times10^{-46}$).

### A3.5 Internal energy

The internal energy is measured from a **reference state** of cold, ground-state H$_2$ and neutral He. Every free
particle carries $\tfrac32k_BT$ of translational energy; each H$_2$ carries its rotational and vibrational energy;
each H nucleus that is no longer in a molecule has paid half the dissociation energy; each ion has paid its
ionization energy. Hence ([code][gen-pe])

```math
e_{\rm int}=\tfrac32 n_{\rm tot}k_BT+n_{\rm H_2}\big[\epsilon_{\rm rot}(T)+\epsilon_{\rm vib}(T)\big]
+\tfrac12\chi_d\,(n_{\rm H}+n_{\rm H^+})+\chi_{\rm H}\,n_{\rm H^+}
+\chi_{\rm He}\,(n_{\rm He^+}+n_{\rm He^{2+}})+\chi_{\rm He^+}\,n_{\rm He^{2+}} .
```

* Vibration: the harmonic-oscillator mean energy above the ground level,
  $\epsilon_{\rm vib}=k_B\theta_{\rm vib}/(e^{\theta_{\rm vib}/T}-1)$ ([code][gen-evib]). Below ~4 K the exponential
  overflows to $+\infty$ and the expression returns exactly 0, the correct frozen limit (the overflow warning is
  suppressed rather than the formula rewritten, to keep shipped tables bit-reproducible; comment in the code).
* Rotation: $\epsilon_{\rm rot}=\tfrac14\langle E\rangle_{\rm para}+\tfrac34\langle E\rangle_{\rm ortho}$, each species'
  Boltzmann average over its own levels (even / odd $J$) and **measured from its own ground state** (para $J=0$,
  ortho $J=1$), so $\epsilon_{\rm rot}\to0$ as $T\to0$ ([code][gen-erot]).

**A modelling inconsistency (measured by T-4).** The rotational energy uses a *frozen* 3:1 mixture with
ortho measured from $J=1$, while the dissociation constant $K_d$ uses the *equilibrium* spin-weighted sum measured
from para $J=0$. The energy is therefore not $-\partial\ln Z/\partial\beta$ of the partition function that sets the
chemical equilibrium, i.e. the table is not derived from one Helmholtz free energy where H$_2$ dissociates. Where the
composition does not change with density (no dissociation), $P=n_{\rm tot}k_BT$ does not depend on the internal
levels and the Maxwell identity of E2 holds regardless. Where H$_2$ dissociates, the energy charged per dissociated
molecule differs between the two models by the mean ortho zero point $\tfrac34\cdot2k_B\theta_{\rm rot}=128$ K$\cdot k_B$
= 0.011 eV (0.25 % of $\chi_d$) [derived]: at $T\gtrsim1000$ K the equilibrium ortho:para ratio is already 3:1, so this
offset is the only difference left. The choice of a frozen mixture itself is physical (slow ortho–para conversion)
and deliberate (code comment).

**T-4 (2026-10-16, `scripts/t4_orthopara.py`, `results/tables/T4_orthopara.txt`).** The Maxwell identity of E2 on its
154 states, with the generator's own Saha solver, for four rotational-energy models and three finite-difference
steps $h$ (worst relative residual):

| model | $h=2\times10^{-3}$ | $10^{-3}$ | $5\times10^{-4}$ | where |
|---|---|---|---|---|
| shipped (frozen 3:1, ortho from $J=1$) | 2.56e-3 | **2.46e-3** | 2.43e-3 | $10^{-12}$ g cm$^{-3}$, 1848 K |
| equilibrium, $u_{\rm rot}=k_BT^2\,d\ln Z_{\rm rot}^{\rm ns}/dT$ (consistent with $K_d$) | 1.96e-4 | 4.9e-5 | 1.2e-5 | $10^{-13}$, 1848 K |
| frozen 3:1 with ortho measured from para $J=0$ | 1.96e-4 | 4.9e-5 | 1.2e-5 | $10^{-13}$, 1848 K |
| shipped energy, dissociation charged $\chi_d-\tfrac34\cdot2k_B\theta_{\rm rot}$ | 1.96e-4 | 4.9e-5 | 1.2e-5 | $10^{-13}$, 1848 K |

The shipped residual does not change with $h$, so it is **not** finite-difference error (as E2 assumed) but the
model inconsistency; every consistent variant falls as $h^2$ (factor 4 per halving), i.e. its residual is pure
truncation. All of it is the ortho zero-point offset: the last two rows, which change nothing but that offset, are as
consistent as the full equilibrium model. Below 1000 K all four agree to $<10^{-9}$ (no composition change).

What a fix would change: the minimal one (last row) leaves the EOS below 1000 K bit for bit unchanged and lowers $u$ in
the dissociation band by 0.12 % (2000 K, $10^{-9}$ g cm$^{-3}$), 0.20 % (3000 K, $10^{-8}$), 0.19 % (5000 K, $10^{-6}$);
$\Gamma_1$ there moves by $\le2.4\times10^{-4}$ (e.g. 1.16718 → 1.16742). The equilibrium model instead changes the
molecular regime itself — $\Gamma_1$ at 50 K from 1.665 to 1.310 — which is a different physical assumption (equilibrium
instead of normal H$_2$), not a consistency fix. Neither is applied: either requires regenerating the tables, which
changes the table EOS above ~1000 K at the 0.1–0.2 % level and every validated result there.

### A3.6 Derived thermodynamic quantities

* **Adiabatic sound speed.** With $\varepsilon$ the specific internal energy, the first law reads
  $d\varepsilon=T\,ds+(P/\rho^2)\,d\rho$. Along an adiabat ($ds=0$) $d\varepsilon=(P/\rho^2)\,d\rho$, and with
  $P=P(\rho,\varepsilon)$:

```math
c_s^2\equiv\Big(\frac{\partial P}{\partial\rho}\Big)_s=\Big(\frac{\partial P}{\partial\rho}\Big)_\varepsilon
+\frac{P}{\rho^2}\Big(\frac{\partial P}{\partial\varepsilon}\Big)_\rho .
```

Ideal-gas check: $P=(\gamma-1)\rho\varepsilon$ gives $c_s^2=(\gamma-1)\varepsilon+(\gamma-1)^2\varepsilon=\gamma P/\rho$.
This identity — not a fit — is what the generator tabulates ([code][gen-cs2]; the comment there records an
earlier version that divided by an extra $\rho$).

* **First adiabatic index** $\Gamma_1=\rho c_s^2/P$, the logarithmic stiffness of the gas. Collapse is resisted
  only while $\Gamma_1>4/3$; the dissociation minimum $\Gamma_1\approx1.1$ (E3, W2) is the physical trigger of
  second collapse.
* **$\gamma_{\rm eff}=1+P/e_{\rm int}$** is a different quantity (the ratio of pressure to energy density, printed
  by the generator's self-check). It equals $\Gamma_1$ only for a constant-$\gamma$ gas: at 2000 K and
  $10^{-9}$ g cm$^{-3}$, $1+P/e_{\rm int}=1.233$ but $\Gamma_1=1.101$ ([measured], W2).
* **Heat capacity** $c_v=(\partial\varepsilon/\partial T)_\rho$ is not tabulated. The one consumer that needs it
  (radiation's Fleck factor) differentiates $\varepsilon(\rho,T)$ numerically with a relative step $10^{-3}T$
  ([code][rad-dtde]).

### A3.7 Range of validity of the model

* **Covered by the physics:** cold molecular gas, rotational excitation, H$_2$ dissociation, H and He
  ionization in the dilute limit.
* **Not in the model [theory, with numbers from the generator's own composition]:** non-ideal effects grow towards
  the dense, hot corner of the table. At $\rho=1$ g cm$^{-3}$ and $3\times10^5$ K the Saha solution is only 42 %
  ionized, the electron degeneracy parameter $n_e\lambda_e^3/2$ is 0.27 and the ion Coulomb coupling
  $\Gamma=e^2/(ak_BT)$ is 0.53; at $10^{-1}$ g cm$^{-3}$, $3\times10^4$ K, $\Gamma\approx1$. Pressure ionization
  and Coulomb corrections, which a dense-plasma EOS includes, are absent, so table values above ~$10^{-2}$ g cm$^{-3}$
  in the ionization band are model-limited, not interpolation-limited. (Estimates computed with
  `gen_eos_table.solve_saha`; they are not a validation test.) The production footprint measured in E10
  ($\rho\le1.5\times10^{-11}$ g cm$^{-3}$, $T\le133$ K) is far from this corner.
* **Above $3\times10^5$ K** the table ends (A4.3).

## A4. From the physics to the table file

### A4.1 Two grids

`build_table` ([code][gen-build]) computes two 2D tables, both on evenly spaced $\log_{10}$ axes:

1. the **$(\rho,T)$ grid**: $\varepsilon(\rho,T)$ and $P(\rho,T)$ from A3, evaluated at every node ([code][gen-rhoT]);
   only $\varepsilon(\rho,T)$ is written (as `espT`);
2. the **$(\rho,\varepsilon)$ grid**: $P$, $c_s^2$ and $\log_{10}T$ as functions of the variables the hydro solver
   holds. It is obtained by inverting, row by row, the monotone map $T\mapsto\varepsilon$ at fixed $\rho$
   ([code][gen-invert]): $\log_{10}T(\varepsilon)$ is the piecewise-linear interpolant of the pairs
   $(\log_{10}\varepsilon_{ij},\log_{10}T_j)$ (`np.interp`, clamped at the ends), and $P$ is then interpolated
   linearly in $\log_{10}T$ along the row. $c_s^2$ is formed from the identity of A3.6 with `np.gradient`
   derivatives of this $P$ on the $(\log\rho,\log\varepsilon)$ grid (second-order central differences inside,
   one-sided at the edges) and floored at $10^{-12}$ ([code][gen-cs2]).

Consequence: the table's $c_s^2$ is the derivative **of the tabulated $P$**, not an independent physical value, and it
inherits a finite-difference bias that is computed exactly in A9.2.

### A4.2 Axes of the shipped tables

`gen_eos_table.py build 400 1000 920 <out.h5> 399 450` (and `... 3.0e5 0 0 120` for rho26;
[provenance][prov]) builds:

| axis | base | pads (prepended, same spacing) | final ([measured], `W_worked_examples.txt` header) |
|---|---|---|---|
| $\log_{10}\rho$ [g cm$^{-3}$] | 400 nodes over $[-20,0]$ | rho26: 120 rows below | 520 nodes, $[-26.015,\,0]$, step 0.050125 |
| $\log_{10}T$ [K] | 920 nodes over $[\log 8,\ \log3\times10^5]$ | 450 below | 1370 nodes, $[-1.3366,\,5.4771]$ (0.046 K–$3\times10^5$ K), step 0.004977 |
| $\log_{10}\varepsilon$ [erg g$^{-1}$] | 1000 nodes over the base $(\rho,T)$ range ×[0.999, 1.001] | 399 below | 1399 nodes, $[6.5267,\,13.8888]$, step 0.005266 |

Padding (prepending nodes at the existing spacing) rather than re-spanning the axes keeps every validated node
**bit-identical** when an axis is extended; the generator refuses a pad that would leave the lowest $\varepsilon$ nodes
outside the $(\rho,T)$ coverage ([code][gen-pad]). The $\varepsilon$ axis is derived from the original rows and
columns only, so the rho26 extension did not move it (AUDIT F-19 comment, [code][gen-eaxis]).

### A4.3 Why the top of the table is flat at high density

At fixed $\rho$, $\varepsilon$ rises monotonically with $T$, so $\varepsilon_{\max}(\rho)=\varepsilon(\rho,T_{\max})$,
and the global $\varepsilon$ axis must reach the largest value, which occurs at the **lowest** density (most
dissociated and ionized per gram). Every denser row therefore has top $\varepsilon$ nodes above its own
$\varepsilon(T_{\max})$; `np.interp` clamps $T$ there and $P$ is flat in $\varepsilon$. This cannot be removed by any
rectangular grid, only pushed to higher $T$; the generator reports it as a warning ([code][gen-sat]). E5 measured
529 of 559 200 node pairs with equal (not decreasing) $P$, all at the $3\times10^5$ K cap for
$\rho>6\times10^{-3}$ g cm$^{-3}$.

### A4.4 The file format

[`eos_table_format.hpp`][fmt] (header parser) and the writer at the end of `build_table` ([code][gen-write]) agree
on version 2: `int64 [MAGIC "EOSTABL1", VERSION 2, FLAGS (bit 0 = cgs), nr, ne, nT]`,
`double [lr0, dlr, le0, dle, lT0, dlT, gen_rho0, gen_v0]`, then four row-major float64 arrays
$P[n_r,n_e]$ (erg cm$^{-3}$), $c_s^2[n_r,n_e]$ (cm$^2$ s$^{-2}$), $\log_{10}T[n_r,n_e]$ (K), $\varepsilon[n_r,n_T]$ (erg g$^{-1}$).
All in **cgs**; the generator computes in its own code units internally and converts only when writing, so its
rounded $\rho_0,v_0$ never reach the file (AUDIT N14). Legacy v1 files (headerless, code units) still load
unchanged and trigger a warning that quantifies their unit bias ([code][hydro-n14]). Shipped tables and their md5
are listed in [`TABLE_PROVENANCE.md`][prov]; E1 verified a byte-identical rebuild.

## A5. Units and dimensional consistency

### A5.1 Code units

The repository runs in the normalisation of the Bonnor–Ebert collapse problem ([`physical_units.hpp`][units]): base
scales $\rho_u$, $v_u$, $l_u$ with $4\pi G=1$; derived $t_u=l_u/v_u$, pressure / energy-density unit $\rho_uv_u^2$,
specific-energy unit $v_u^2$, magnetic unit $\sqrt{4\pi\rho_u}\,v_u$ (Heaviside–Lorentz), and the temperature unit
([code][units-T])

```math
T_u=\frac{\mu_{\rm th}\,m_H\,v_u^2}{k_B},\qquad \mu_{\rm th}=2.29 .
```

($\mu_{\rm th}$ is the key `units/mu_thermal`.)

With the defaults ($\rho_u=5.467\times10^{-19}$ g cm$^{-3}$, $v_u=1.9\times10^4$ cm s$^{-1}$) $T_u=10.015144$ K,
$v_u^2=3.61\times10^8$ erg g$^{-1}$, $\rho_uv_u^2=1.973587\times10^{-10}$ erg cm$^{-3}$ ([measured], W header). For a
`collapse_be` run the base scales are derived from the Bonnor–Ebert IC (mass, temperature, `f`) and a `<units>`
override is refused ([code][units-build]).

### A5.2 Where units change

| interface | unit convention | conversion ([code]) |
|---|---|---|
| table file | cgs, $T$ in K | — |
| `EosTable::Load` | file → code | $\log\rho_{\rm code}=\log\rho_{\rm cgs}-\log\rho_u$ and $\log\varepsilon_{\rm code}=\log\varepsilon_{\rm cgs}-\log v_u^2$ (shift of the axis **origin**; the spacing is invariant), $P/(\rho_uv_u^2)$, $c_s^2/v_u^2$, $\varepsilon/v_u^2$; $\log T$ untouched ([Load][tab-load]) |
| all lookups | $\rho$, $\varepsilon$, $e_{\rm int}$, $P$, $c_s^2$ in **code** units; $T$ in **kelvin** | — |
| callers wanting code temperature | $T_{\rm code}=T/T_u$ | radiation, non-ideal MHD (`T_unit`) |
| `hydro/Tfloor`, `hydro/Tceil` under `eos = hydrogen` | **code** temperature | $T=T_{\rm floor}\,T_u$, $T_{\rm ceil}\,T_u$ in K, handed to the EOS; the energy is $e(\rho,T)$ from the table at each cell's density ([code][hydro-tfloor], [code][hydro-tceil], [code][eos-tlim]). Before `5649acd` `Tceil` was in kelvin with the ideal-gas conversion (O-3) and the floor energy was taken at $\rho_{\rm code}=1$ (O-5) |
| `hydro/Tfloor`, `hydro/Tceil` under `eos = adiabatic` | **kelvin**, via $\bar m/k_B$ of the `Units` class | $\varepsilon=T/(\bar m/k_B)/(\gamma-1)$ ([code][hydro-tceil]) |

The load conversion is exact because the axes are logarithmic: a change of unit multiplies $\rho$ by a constant,
which shifts $\log\rho$ by a constant. Check with the rho26 table: the file's lowest density
$\log\rho_{\rm cgs}=-26.0150$ becomes $-26.0150-\log_{10}(5.467\times10^{-19})=-7.7528$ in code units, as printed by
the harness ($-7.752787$).

### A5.3 Dimensional checks of the implemented formulas

| expression (where) | dimensions | check |
|---|---|---|
| $P=n_{\rm tot}k_BT$ (generator) | cm$^{-3}$ · erg K$^{-1}$ · K = erg cm$^{-3}$ | pressure |
| $c_s^2=(\partial P/\partial\rho)_\varepsilon+(P/\rho^2)(\partial P/\partial\varepsilon)_\rho$ | (erg cm$^{-3}$)/(g cm$^{-3}$) = cm$^2$ s$^{-2}$; (erg cm$^{-3}$)/(g$^2$cm$^{-6}$) · (erg cm$^{-3}$)/(erg g$^{-1}$) = cm$^2$ s$^{-2}$ | both terms are velocity$^2$ |
| `Pof`: $P=\rho\cdot\widehat{(P/\rho)}$ (p_over_rho) | $\rho_u\cdot v_u^2$ | the stored array is $P/\rho_{\rm node}$ in $v_u^2$ |
| `EintFromRhoTk`: $e_{\rm int}=\rho\cdot\hat\varepsilon(\rho,T)$ | $\rho_u\cdot v_u^2=\rho_uv_u^2$ | energy density |
| fast speed $c_f^2=\frac{1}{2\rho}\big[B^2+\rho c_s^2+\sqrt{(B^2-\rho c_s^2)^2+4\rho c_s^2B_t^2}\big]$ ([code][eos-cf]) | $B^2$ is an energy density in Heaviside–Lorentz units, as is $\rho c_s^2$ | velocity$^2$ |
| $T_u=\mu_{\rm th}m_Hv_u^2/k_B$ | g · cm$^2$s$^{-2}$ / (erg K$^{-1}$) = K | temperature |

### A5.4 Two "molecular weights" — a consequence worth knowing

$\mu_{\rm th}=2.29$ is a **calibration** that makes the isothermal sound speed exactly $v_u$ at $T_u$ for an
ideal gas; the table's composition gives $\mu=2.331$ for molecular gas ([measured]: W1, $\mu=2.330921$ from the
table, 2.331002 from the generator). At the temperature $T_u$ the table gas therefore has
$P/\rho=\mu_{\rm th}/\mu=0.98244$ in code units, not 1 ([measured], W1): with the table, "$T_{\rm code}=1$" and
"isothermal sound speed 1" are two different states, 1.8 % apart in pressure.

This matters for `collapse_be`, whose Bonnor–Ebert density profile assumes $p=\rho$. The IC sets
$e_{\rm int}=p\cdot\varepsilon(\rho,T_0)$ with $p=\rho$ ([code][be-ic], [code][be-p]) and the barotropic step
holds the isothermal phase at $T_0$. Up to commit `a40eff5`, $T_0=T_u$, so the table returned $P=0.9824\,\rho$ and
the sphere started 1.76 % below its equilibrium (observation O-4). Since `5649acd` the default
(`problem/collapse_be/table_isothermal = pressure`) takes $T_0$ as the table temperature at which $P=\rho$: a
fixed-point iteration $T\leftarrow T\,\rho/P(\rho,\varepsilon(\rho,T))$ at the central density, evaluated once at
package initialisation ([code][be-t0]); 10.19 K for the default units. Measured: $P/\rho=1.00000000$ in every cell at
$t=0$ (`O_fixes_check_2026-10-16.txt`). `table_isothermal = temperature` restores $T_0=T_u$ and reproduces the
earlier runs bitwise. Which of the two is wanted is a modelling choice: the first keeps the dynamics of the
Bonnor–Ebert sphere, the second its nominal temperature. It is not a defect of the EOS.

## A6. Execution path at runtime

### A6.1 Initialization (once per run, host)

1. **Keys and table** — `Hydro::Initialize` ([code][hydro-eos]):
   1. `hydro/eos` is read with `GetString` (required); `adiabatic` or `hydrogen`, otherwise the run aborts.
   2. `hydro/gamma` is read with `GetReal` (required for **both** EOS; uses under `hydrogen` in A10).
   3. If `hydrogen`: the table path `hydro/eos_table_file` must be given unless `hydro/eos_table_allow_default` is
      true; `PhysUnits::BuildPhysicalUnits` provides $\rho_u,v_u$; `eos_interp` and `eos_extrapolate` are set
      **before** `Load` (both change what `Load` stores or how lookups branch); `Load` runs; then `eos_bisections`
      is read and range-checked ([10, 60]); a legacy table prints the N14 warning; `hydro/riemann` must be `hlld`
      ("HLLE's Roe average bakes in a constant gamma", [code][hydro-hlld]).
2. **Floors and ceilings** ([code][hydro-tfloor]): `dfloor`, `pfloor` (code units, default −1 = off); `Tfloor`
   (default −1) is converted to a specific-energy floor $\varepsilon_{\rm fl}$ — under `hydrogen` by the host lookup
   `EintFromRhoTkHost(1.0, Tfloor·T_u)/1.0`, which only switches the floor on; the floor temperature in kelvin is
   handed to the EOS (`SetTableTemperatureLimits`, [code][eos-tlim]) and `ConsToPrim` takes the floor energy from
   the table at each cell's density. `Tceil` (default $\infty$) likewise under `hydrogen` (a code temperature);
   under `adiabatic` both are kelvin with the ideal-gas formula ([code][hydro-tceil]).
3. **The EOS object** — `AdiabaticGLMMHDEOS eos(pfloor, dfloor, efloor, vceil, eceil, gamma, use_h2diss, eos_tab,
   dual-energy keys…)` is constructed and stored as the Hydro package Param `"eos"` ([code][hydro-ctor]); the
   `EosTable` is held **by value** inside it ([code][eos-class]). `FillDerivedMesh` is set to `ConsToPrim`
   (or `ConsToPrimDoubleFlux` with the double-flux fill) and `EstimateTimestepMesh` to the hyperbolic estimate.
4. The **radiation** package, when enabled, loads its **own** copy of the same table with the same keys
   ([code][rad-load]); the clamp report sums both copies' counters.

`EosTable` is small: Kokkos View handles plus grid scalars. Kernels capture it by value, so the handles travel to
the device. The two host mirrors are `std::shared_ptr` so that this per-launch copy is a reference-count increment
(E-F11: as `std::vector` they were deep-copied at every kernel launch, 10–23× slower steps, identical results).

### A6.2 Every stage (device)

The order below is the order in which the EOS is called within one stage of the VL2 integrator for an
`eos = hydrogen` MHD run (functions in Sec. A7/A8):

1. **Time step** (once per cycle): `EstimateHyperbolicTimestep` computes $|v|+c_f$ per cell and direction with
   `FastMagnetosonicSpeed` → `AsqFromRhoPres` → `EspFromP` + `Cs2of` ([code][dt-hyp]).
2. **Reconstruction**: PLM (or another scheme) reconstructs **primitive** variables, including $P$, to the faces.
   No EOS call.
3. **Riemann solver** (HLLD): per face, two `InternalEnergyFromPres` (one bisection each) to build $E_L,E_R$ and
   two `FastMagnetosonicSpeed` (one bisection + `Cs2of` each) for the outer wave speeds ([code][hlld-e]).
4. **Conservative update** of $U$ with the fluxes. No EOS call.
5. **Source terms** that set the energy from a temperature: `collapse_be`'s barotropic step calls
   `EintFromRhoTk` per cell ([code][be-baro]); radiation matter coupling calls `TemperatureK` and `EintFromRhoTk`.
6. **Boundary exchange**, then **`ConservedToPrimitive`** over the entire block including ghosts ([code][c2p-mesh]):
   one `Pof` per cell, plus two lookups for the floor energy and pressure when `Tfloor` is set, plus the floor inversion
   when a cell is floored ([code][c2p]).
7. Consumers of the new primitives: Jeans refinement (`SoundSpeed`), non-ideal coefficients
   (`TemperatureKFromPres`), dust (`TemperatureK`), diagnostics.

Once per cycle, on the host, `PreStepMeshUserWorkInLoop` drains the out-of-table counters, sums them over MPI ranks
and prints them ([code][clamp-report]).

## A7. The table lookups, function by function

All functions in this section are in [`src/eos/eos_table.hpp`][tab]. The code exists in two compile-time variants:
the default build, and a verbatim copy of the pre-2026-10-05 lookups selected by the CMake option
`ATHENAPK_EOS_LEGACY_LOOKUPS` ([code][tab-legacy]; E-F9 explains why this is a compile-time switch). This section
describes the default build.

### A7.1 `bilin`: the interpolation kernel

**Purpose.** Every lookup is a bilinear interpolation on an evenly spaced grid in two logarithmic coordinates
([code][tab-bilin]). For an array $A$ with axes $x_0+i\,\Delta x$, $y_0+j\,\Delta y$ and a query $(x,y)$:

```math
f_i=\frac{x-x_0}{\Delta x},\quad i=\mathrm{clamp}(\lfloor f_i\rfloor_{\to0},\,0,\,n_1-2),\quad t_i=\mathrm{clamp}(f_i-i,\,0,\,1),
\qquad (\text{same for } j,\,t_j)
```
```math
\hat A(x,y)=(1-t_i)(1-t_j)A_{i,j}+t_i(1-t_j)A_{i+1,j}+(1-t_i)t_jA_{i,j+1}+t_it_jA_{i+1,j+1}.
```

**Numerical interpretation.** The index is found by arithmetic, not search: $O(1)$, branch-light, identical cost
for every cell — the right shape for GPUs. `static_cast<int>` truncates toward zero, so for $f_i\in(-1,0)$ the index
is 0 and only the clamp of $t_i$ to 0 keeps the result at the edge node; outside the grid **both** the index and the
weight are clamped, so the result is the nearest edge value (nearest-neighbour extrapolation), never a linear
extrapolation.

**Safeguard (observability).** Each excursion is counted in a 12-slot device View `clamp_ctr_`
([code][tab-ctr]): slots 0–5 count real cell states (ρ below/above, second axis below/above for the ε grid, T
below/above for the T grid), slots 6–11 the same for root-finder **probe** evaluations (radiation's Newton
iterations), so benign probes do not mask a genuine clamp. The low-side test uses $f_i<0$, not the truncated index,
so excursions of less than one cell are counted (code-review fix noted in the source). The counter is write-only
from the kernels: it cannot change a result. The upper count fires only for $f_i>n-1$, i.e. beyond the top node;
up to `a40eff5` the test was $i>n-2$, so a query exactly on the top node (an exact table value) was counted as a
clamp (O-7; [measured], W3(f): now 0 counts at the top node).

### A7.2 `Pof`, `Cs2of`, `LogTof`: the $(\rho,\varepsilon)$ grid

```math
P(\rho,\varepsilon)=\phi(\rho)\;\widehat{A_P}(\log_{10}\rho,\log_{10}\varepsilon),\qquad
c_s^2=\widehat{A_{c^2}}(\cdot,\cdot),\qquad T=10^{\widehat{A_{\log T}}(\cdot,\cdot)}
```

([code][tab-pof]; $T$ is returned by `TemperatureK` as $10^{\rm LogTof}$). The factor $\phi$ and the stored array
depend on `hydro/eos_interp`:

* `p_over_rho` (default since E-F1): at load the stored array is divided by the node density,
  $A_P(i,j)=P_{ij}/\rho_i$ ([code][tab-prho]), and $\phi(\rho)=\rho$ (the *actual* density).
* `p` (legacy): $A_P=P_{ij}$, $\phi=1$.

**Why `p_over_rho` [derived].** At fixed $\varepsilon$ and fixed composition the gas obeys $P\propto\rho$ exactly
(A3.2 with $\mu$ and $T(\varepsilon)$ independent of $\rho$). Interpolating $P$ linearly in $\log\rho$ between nodes
$\rho_i$ and $\rho_i10^{\Delta}$ ($\Delta=0.050125$) returns, at fractional position $t$,
$[(1-t)+t\,10^{\Delta}]/10^{t\Delta}$ times the true value: at mid-cell $\cosh(\tfrac12\Delta\ln10)=1.0016656$, i.e.
+0.167 %, and the slope $\partial P/\partial\rho|_\varepsilon$ seen by the interpolant varies across each cell by the
factor $(10^\Delta-1)/(\Delta\ln10)\cdot\rho_i/\rho$, from +5.9 % at the lower node to −5.6 % at the upper node. That
slope enters the effective sound speed of the scheme and produces the sawtooth of E-F1 (E6 measured phase-speed
errors of −1.40 % … +0.55 % with `p`). Storing $P/\rho_i$ makes the interpolated quantity **constant** along $\rho$
where $P\propto\rho$, so the interpolation is exact there; the remaining error comes only from the composition's
density dependence. Verified on the real code: the mid-cell ratio of the `p` to the `p_over_rho` pressure is 1.00166562, equal to
the prediction to all printed digits ([measured], W3(e)).

**Below the energy axis** ($\varepsilon<\varepsilon_{\min}$, i.e. $T<0.063$ K; default `eos_extrapolate = true`)
the gas is continued as the ideal gas it is there ([code][tab-x]):

```math
P(\rho,\varepsilon)=P(\rho,\varepsilon_{\min})\,\frac{\varepsilon}{\varepsilon_{\min}},\qquad
c_s^2=c_s^2(\rho,\varepsilon_{\min})\,\frac{\varepsilon}{\varepsilon_{\min}},\qquad
T=T(\rho,\varepsilon_{\min})\,\frac{\varepsilon}{\varepsilon_{\min}} .
```

Derivation: with constant composition and frozen internal degrees of freedom, $P=(\gamma-1)\rho\varepsilon$,
$c_s^2=\gamma(\gamma-1)\varepsilon$ and $T\propto\varepsilon$ with $\gamma=5/3$, so all three are linear in
$\varepsilon$ at fixed $\rho$. It is exact here because the table's own bottom row is ideal: $P/(\rho\varepsilon)$ on
that row is 0.666675 at all densities (§9.4). Below the density axis with `eos_interp = p` the pressure is continued
as $P=(\rho/\rho_{\min})P(\rho_{\min},\varepsilon)$ (E-F6), which is what `p_over_rho` does by construction. $c_s^2$
and $T$ are intensive and stay clamped in $\rho$. All these lookups are still counted (they are outside the table).

### A7.3 `EspFromP`: the inverse $\varepsilon(\rho,P)$

**Purpose.** The face states of the Riemann solver carry $P$; the solver needs $E=\rho\varepsilon+\dots$. The
inverse of `Pof` at fixed $\rho$ is computed by **bisection on $\log_{10}\varepsilon$** over the whole axis
$[\ell_0,\ \ell_0+(n_e-1)\Delta\ell]$ with a fixed number of steps $n_b$ (`hydro/eos_bisections`, default 20)
([code][tab-espfromp]):

```math
\ell_m=\tfrac12(\ell_{\rm lo}+\ell_{\rm hi});\quad
\text{if } \phi(\rho)\widehat{A_P}(\log\rho,\ell_m)<P:\ \ell_{\rm lo}=\ell_m\ \text{else}\ \ell_{\rm hi}=\ell_m;\qquad
\varepsilon=10^{(\ell_{\rm lo}+\ell_{\rm hi})/2}.
```

**Requirements.** Bisection needs $P$ increasing in $\varepsilon$ at fixed $\rho$; E5 verified it at every node pair
(strictly increasing below $3\times10^5$ K, equal on 529 pairs at the cap, A4.3).

**Error bound [derived].** After $n_b$ halvings the bracket has width $W2^{-n_b}$ in $\log_{10}\varepsilon$ with
$W=(n_e-1)\Delta\ell=7.362$ dex, and the midpoint is within half of it, so

```math
\Big|\frac{\delta\varepsilon}{\varepsilon}\Big|\le \ln 10\cdot W\cdot 2^{-(n_b+1)} = 8.08\times10^{-6}\quad(n_b=20).
```

E5 measured exactly this maximum; W1/W2 show $-5.5\times10^{-6}$ and $-6.4\times10^{-6}$ at 20 steps,
$\le3\times10^{-9}$ at 30, round-off at 60 ([measured]). The resulting **pressure** error is
$(\partial\ln P/\partial\ln\varepsilon)\,\delta\varepsilon/\varepsilon$, up to $9.2\times10^{-5}$ in the dissociation
and ionization bands where that logarithmic derivative is large (E5).

**Why a fixed count, not a tolerance.** On a GPU all lanes of a warp execute until the slowest one finishes; a fixed
count is divergence-free and predictable (code comment). The price is an error that is a *piecewise-constant*
function of $P$ — non-smooth, unlike the interpolation error — and it enters every HLLD face energy. The double-flux
reset, which needs pressure equilibrium to round-off, therefore refines the bisection result with three Newton steps
on the interpolant ([code][dflux-newton]).

**Edges.** With the continuation on, a target below $P(\rho,\varepsilon_{\min})$ is inverted exactly,
$\varepsilon=\varepsilon_{\min}P/P_{\min}$, and counted as `esp<` ([code][tab-x]). Only the $\rho$ axis is counted once
per call, because the loop itself runs with counting off (else every one of the $n_b$ probes would count a $\rho$
clamp). A target **above** $P(\rho,\varepsilon_{\max})$ cannot be represented: the bisection converges to the top of
the axis and returns $\varepsilon\approx\varepsilon_{\max}$ ([measured], W3(f): target $10\,P_{\max}$ returns
$0.999992\,\varepsilon_{\max}$, whose pressure is 10× lower than the target). Since `5649acd` this is counted as
`esp>`: when the bisection never lowered its upper bound, `XCountEspHighP` compares the target with
$P(\rho,\varepsilon_{\max})$ ([code][tab-xhigh]), one extra lookup only in that case. Up to `a40eff5` it was silent
(O-1).

### A7.4 The compound interface and the $(\rho,T)$ grid

([code][tab-iface])

| function | definition | note |
|---|---|---|
| `PresFromRhoEint(rho, eint)` | `Pof(rho, eint/rho)` | $e_{\rm int}$ is an energy **density** |
| `EintFromRhoPres(rho, P)` | `EspFromP(rho, P) * rho` | one bisection |
| `AsqFromRhoPres(rho, P)` | `Cs2of(rho, EspFromP(rho, P))` | one bisection + one lookup |
| `TemperatureK(rho, eint)` | `10^LogTof(rho, eint/rho)` | kelvin |
| `TemperatureKFromPres(rho, P)` | `10^LogT(rho, EspFromP(rho,P))`, ρ axis counted only by `EspFromP` | [code][tab-tfromp] |
| `EintFromRhoTk(rho, Tk, probe)` | $\rho\cdot\widehat{A_\varepsilon}(\log\rho,\log T)$ on the $(\rho,T)$ grid; below $T_{\min}$: $\rho\,\varepsilon(\rho,T_{\min})\,T/T_{\min}$ | `probe = true` tallies in the probe bank ([code][tab-eintT]) |

**Forward–inverse consistency.** $\varepsilon(\rho,T)$ (the $(\rho,T)$ grid) and $T(\rho,\varepsilon)$ (the
$(\rho,\varepsilon)$ grid) are **two independent bilinear interpolants** of the same physics; their composition is
not the identity (E-F7): $2\times10^{-5}$ in the production footprint, up to 1.3 % across dissociation and ionization
(E5); W1: $1.6\times10^{-5}$, W2: $1.0\times10^{-4}$ ([measured]).

**Host twins.** `EintFromRhoTkHost` and `TemperatureKHost` ([code][tab-host]) are line-for-line copies operating on
the host mirrors, for setup code (`hydro/Tfloor`, problem generators): dereferencing a device View from host code is
illegal and traps on some architectures (source comment). They do not count clamps.

### A7.5 Loading: `EosTable::Load`

([code][tab-load]) In order: open the file; parse the header (v1 or v2); reject a grid with fewer than two nodes on
an axis or a non-positive spacing; shift the axis origins and set the payload scale factors (A5.2); compute
$\rho_{\min},\varepsilon_{\min},T_{\min}=10^{\text{origin}}$; read the four arrays into device Views through host
mirrors, scaling on the way, and (p_over_rho) divide $P$ by $\rho_i$; throw if the file is truncated; copy the
$\varepsilon(\rho,T)$ and $\log T$ arrays back from the device into the shared host mirrors (so the host twins read
exactly what the kernels read); allocate the zero-initialised counter View. `p_over_rho_` is applied inside `Load`
(the division by $\rho_i$) and `extrapolate_` by every lookup; the source requires both to be set before `Load`, and
both load sites (hydro, radiation) do so.

## A8. How the EOS enters the solver

### A8.1 `AdiabaticGLMMHDEOS`: one interface for both EOS

[`src/eos/adiabatic_glmmhd.hpp`][eos-class] holds $\gamma$, the floors and ceilings, `use_h2diss_` (true for
`eos = hydrogen`) and the `EosTable`. Each method branches once on `use_h2diss_`, and with `use_h2diss_ = false` it is
the ideal gas:

| method | `eos = adiabatic` | `eos = hydrogen` |
|---|---|---|
| `InternalEnergyFromPres(rho, P)` | $P/(\gamma-1)$ | `EintFromRhoPres` |
| `SoundSpeed(prim)` | $\sqrt{\gamma P/\rho}$ | square root of `AsqFromRhoPres` |
| `FastMagnetosonicSpeed(d, p, bx, by, bz)` | with $\rho a^2=\gamma P$ | with $\rho a^2=\rho\,c_s^2$, $c_s^2$ from `AsqFromRhoPres` |
| pressure in `ConsToPrim` | $(\gamma-1)e_{\rm int}$ | `PresFromRhoEint` if $e_{\rm int}>0$, else $(\gamma-1)e_{\rm int}$ |

**Fast magnetosonic speed [derived].** The MHD fast speed is
$c_f^2=\tfrac12\big[a^2+b^2+\sqrt{(a^2+b^2)^2-4a^2b_x^2}\big]$ with $a^2=c_s^2$ and $b^2=B^2/\rho$ (Heaviside–Lorentz,
no $4\pi$). With $Q\equiv\rho a^2$ and $B^2=B_x^2+B_t^2$, $(Q+B^2)^2-4QB_x^2=(B^2-Q)^2+4QB_t^2$, which is the form the
code evaluates ([code][eos-cf]): `asq` $=Q$, `qsq` $=B^2+Q$, `tmp` $=B^2-Q$,
$c_f=\sqrt{\tfrac12\big(B^2+Q+\sqrt{(B^2-Q)^2+4QB_t^2}\big)/\rho}$. The rewritten radicand is a sum of
non-negative terms, so it cannot go negative through cancellation. The general EOS enters only through $a^2$.

### A8.2 `ConsToPrim`: from conserved to primitive variables, with floors

([code][c2p]; called per cell by `ConservedToPrimitive` over the **entire** block including ghosts, [code][c2p-mesh])

1. **Density floor**: abort if $\rho\le0$ and no floor is set; else $\rho\leftarrow\max(\rho,\rho_{\rm fl})$
   (momentum and energy untouched) ([code][c2p-d]).
2. Velocities $\mathbf v=\mathbf m/\rho$; $\mathbf B$, $\psi$ copied.
3. $e_{\rm int}=E-|\mathbf m|^2/(2\rho)-B^2/2$ ([code][c2p-eint]).
4. **Dual energy** (only with `hydro/dual_energy`; see the dual-energy report): may replace $e_{\rm int}$ by the
   separately advected internal energy where the difference above is ill-conditioned.
5. **Pressure** ([code][c2p-p]): `PresFromRhoEint(rho, eint)` if $e_{\rm int}>0$; otherwise $(\gamma-1)e_{\rm int}$.
   *Why the branch:* the table is indexed by $\log_{10}(e_{\rm int}/\rho)$; for $e_{\rm int}\le0$ that is NaN, and
   NaN compares false against every floor test below, so the cell would be silently poisoned. The ideal-gas
   expression instead gives a negative pressure that the floor logic (or the `PARTHENON_REQUIRE`) catches. Inside
   `ConsToPrim` this is where `hydro/gamma` enters a table run, and only through the sign of the result.
6. Velocity ceiling (rescales $\mathbf m$ and $E$; not EOS).
7. **Pressure floor** ([code][c2p-pfl]): if $P<P_{\rm fl}$, set $e_{\rm fl}=$ `InternalEnergyFromPres`$(\rho,P_{\rm fl})$
   and $E=e_{\rm fl}+$KE$+$ME. The stored pressure is $P_{\rm fl}$, or — with `hydro/eos_floor_consistent = true` —
   the table pressure of the energy actually stored, `PresFromRhoEint(rho, e_fl)`. The two differ when the
   inversion cannot represent $P_{\rm fl}$; without the continuation that was the case for every realistic floor
   (E-F5: $P(e_{\rm fl})=6.2\times10^5P_{\rm fl}$ at $\rho_{\rm code}=1$, W3(b) legacy). With the default continuation
   the inversion is exact ($P(e_{\rm fl})=P_{\rm fl}\times1.000000000000$, W3(b)) and the floored gas sits at
   $T=10^{-7}$ K — which is why a temperature floor is the physical choice.
8. **Temperature floor** ([code][c2p-tfl]): the floor energy is $e_{\rm fl}(\rho)=$`EintFloor`$(\rho)$, which is
   `EintFromRhoTk`$(\rho,T_{\rm floor})$ under the table and $\rho\varepsilon_{\rm fl}$ for the ideal gas
   ([code][eos-tlim]); the floor pressure is $P_{T,\rm fl}=$`PresFromRhoEint`$(\rho,e_{\rm fl})$ under the table (two
   extra lookups per cell when `Tfloor` is set), $(\gamma-1)\rho\varepsilon_{\rm fl}$ otherwise; if $P<P_{T,\rm fl}$ the
   energy is set to $e_{\rm fl}+$KE$+$ME. A cell held by this floor is counted in the history column `cons-nTfloor`,
   with the same per-density floor ([code][diag-tfloor]). Up to `a40eff5` the table floor energy was
   $\rho\,\varepsilon(\rho_{\rm code}=1,T_{\rm floor})$, exact only where $\varepsilon(T)$ does not depend on density
   (O-5: a 2000 K floor held gas at 2176 K at $10^{-12}$ g cm$^{-3}$ and 2779 K at $10^{-9}$; now 1999.9 and 2000.2 K).
9. **Temperature ceiling** ([code][c2p-tceil]): analogous with `EintCeil`, guarded so that the default
   $\varepsilon_{\rm ceil}=\infty$ never reaches the table ($\log_{10}\infty$ would clamp to the edge and trigger the
   ceiling spuriously).
10. Dual-energy re-sync; passive scalars $=\,$cons$/\rho$.

Every floor **injects** energy (non-conservatively, by design) and records it only through the history counters.
Floors and ceilings that rewrite $E$ keep the dual energy consistent with the new $E$.

### A8.3 HLLD and the other Riemann solvers

HLLD (Miyoshi & Kusano 2005) builds the left/right conserved states from the reconstructed primitives
([code][hlld-e]):

```math
E_{L,R}=e_{\rm int}(\rho_{L,R},P_{L,R})+\tfrac12\rho_{L,R}v_{L,R}^2+\tfrac12B_{L,R}^2 ,
```

with $e_{\rm int}$ from `InternalEnergyFromPres`, and estimates the outer wave speeds
$S_{L}=\min(v_{x,L}-c_{f,L},v_{x,R}-c_{f,R})$, $S_R=\max(\dots)$ with the general-EOS $c_f$. The intermediate states
follow from Rankine–Hugoniot conditions and continuity of the total pressure across the contact and Alfvén
waves, which do not involve the EOS (code comment), so these two places are the solver's entire EOS dependence.

`eos = hydrogen` requires `riemann = hlld`. Two other solvers still contain the ideal-gas energy:

* `glmmhd_hlle` takes $\gamma$ from the EOS and is refused under the table ([code][hydro-hlld]).
* `glmmhd_dc_llf` is used by `hydro/first_order_flux_correct` (FOFC, `FirstOrderFluxCorrect`, [code][fofc]), which
  is allowed under `eos = hydrogen` (refused only together with `hydro/double_flux`). Since `5649acd` it takes the
  face internal energies from the active EOS (`InternalEnergyFromPres`, [code][llf-e]); up to `a40eff5` it used
  $P/(\gamma-1)$ with `igm1` also under the table (O-2). Measured in a low-β vacuum-forming flow where FOFC corrects
  32 cells: the ideal-gas energy changed the pressure by up to 13 % in single cells after 200 cycles
  (`O_fixes_check_2026-10-16.txt`).

### A8.4 Problem generators and source terms that set energy from temperature

* `eos_test` (all EOS validation decks) sets every state through the table on the device:
  $e_{\rm int}=$`EintFromRhoTk`$(\rho,T)$, or directly from a specific energy (`espL_code/espR_code`) to plant
  sub-table or negative energies ([code][eostest]). The acoustic mode perturbs $e_{\rm int}$ linearly,
  $\delta(\rho\varepsilon)=(\varepsilon_0+P_0/\rho_0)\,\delta\rho$ (the specific enthalpy times $\delta\rho$), to avoid the
  bisection error ($\sim10^{-5}$) at a $10^{-6}$ amplitude.
* `collapse_be` with `eos = hydrogen`: the IC uses $\varepsilon(\rho,T_0)$ (A5.4; $T_0$ = `collapse_be_T0K`), and the
  barotropic step prescribes $T=T_0\sqrt{1+(\rho/\rho_{\rm crit})^{2(\gamma-1)}}$ and sets $E=$`EintFromRhoTk`$(\rho,T)+$KE$+$ME
  ([code][be-baro]); there `hydro/gamma` is only the barotropic index. Skipped when the radiation package owns the
  energy.
* Radiation: temperature from `TemperatureK`; the Fleck factor needs $dT/d\varepsilon$, obtained as
  $2\Delta T/[\varepsilon(T+\Delta T)-\varepsilon(T-\Delta T)]$ with $\Delta T=10^{-3}T$ and set to 0 (explicit
  scheme) if the difference is not positive ([code][rad-dtde]).
* Non-ideal MHD: the ionization model's temperature is `TemperatureKFromPres`$/T_u$, never $P/\rho$
  ([code][diff-T]); dust: `TemperatureK` with a floor ([code][dust-T]).
* Double flux (E-F10, Sec. 6.4): freezes a stiffened-gas fit through the cell state,
  $g_*=\rho c_s^2/P$ (clamped to $[1.0001,\,5]$) and $\varepsilon_{0*}=(e_{\rm int}-P/(g_*-1))/\rho$
  ([code][dflux-freeze]), and resets $E$ from the table at the frozen-EOS pressure with `EspFromPRefined`
  (bisection + 3 Newton steps, [code][dflux-newton]).

## A9. Numerical accuracy and stability

### A9.1 The error budget of one lookup

| source | size | where measured / derived |
|---|---|---|
| Saha physics vs the real gas | model-limited (A3.7) | not tested |
| generator: Saha solve | round-off (bisection 200×) | E2: neutrality $5\times10^{-15}$, element conservation $2\times10^{-16}$ |
| table vs direct Saha at off-node points ($P$, $c_s^2$) | median $1.6\times10^{-3}$, max $5.5\times10^{-3}$ ($P$); max $3.3\times10^{-2}$ ($c_s^2$, at a kink) | E2 |
| bilinear $\varepsilon(\rho,T)$ at an off-node point | W1 $1.6\times10^{-5}$, W2 $8.8\times10^{-4}$ (table 914.15 vs Saha 913.35 code) | W, `W_worked_saha.txt` vs `W_worked_examples.txt` |
| $c_s^2$ finite-difference bias | +0.133 % in $\Gamma_1$ for molecular gas (derived below) | W1 |
| `p` interpolation (legacy) | +0.167 % in $P$ at mid-cell, slope sawtooth ±6 % | A7.2, E6, E-F1 |
| P → ε bisection | $\le8.08\times10^{-6}$ in $\varepsilon$, $\le9.2\times10^{-5}$ in $P$ | A7.3, E5 |
| T → ε → T round trip | $2\times10^{-5}$ (footprint) … 1.3 % | E-F7, E5 |

### A9.2 The bias of the tabulated sound speed [derived, verified]

For molecular gas at 10 K, $P=\tfrac23\rho\varepsilon$ exactly (frozen rotation; W1: $e_{\rm int}/P=1.499971$ in the
table, 1.5 from the generator), so $c_s^2=P/\rho+(P/\rho^2)(\tfrac23\rho)=\tfrac53P/\rho$, of which the
$(\partial P/\partial\rho)_\varepsilon$ term is the fraction $3/5$. The generator evaluates that derivative with a
central difference on the $\log_{10}\rho$ grid: for $P=A\rho$,

```math
\frac{P_{i+1}-P_{i-1}}{2\Delta}=A\rho_i\,\frac{10^{\Delta}-10^{-\Delta}}{2\Delta}
=\ln10\,A\rho_i\,\frac{\sinh x}{x},\qquad x=\Delta\ln10=0.115417,
```

i.e. the exact derivative times $\sinh x/x=1+x^2/6+\dots=1.0022217$. The same factor along $\log\varepsilon$
($y=0.005266\ln10$) is $1+2.45\times10^{-5}$. Weighted by $3/5$ and $2/5$: predicted
$\Gamma_1^{\rm table}/\Gamma_1=1.0013428$. Measured on the real code: $1.668887/1.666667=1.0013322$ (W1,
[measured]); the 1e-5 remainder comes from the interpolations in the generator and at run time, which this estimate
ignores. So the table's sound speed in cold molecular gas is 0.067 % fast, purely from the grid spacing in density;
E3 reported $\Gamma_1=1.668$ at 10 K for the same reason. The effect depends only on $\Delta\log\rho$; halving it
would reduce it 4×.

### A9.3 Precision and cancellation

* $e_{\rm int}=E-\mathrm{KE}-\mathrm{ME}$ loses relative precision as $(\mathrm{KE}+\mathrm{ME})/e_{\rm int}$ grows (low
  plasma beta, hypersonic flow). That is the dual-energy module's subject, not the table's; the EOS's part is to
  receive a possibly non-positive $e_{\rm int}$ safely (A8.2 step 5).
* All table arithmetic is double precision. GCC may contract $a\cdot b+c$ into fused multiply-adds differently
  depending on inlining, so any change to the inline lookup bodies changes the last bit of results in callers
  (E-F9); bitwise reproduction of an old run therefore needs the legacy build.
* The lookups call `log10` and `pow`; the GPU's versions may differ from the CPU's in the last ulp (E9: lookups
  agree to $\le3.3\times10^{-15}$).

### A9.4 Robustness: what happens to bad inputs

| input | path | result ([code] / [measured]) |
|---|---|---|
| $e_{\rm int}\le0$ in ConsToPrim | ideal-gas branch | negative $P$ → pressure/temperature floor or abort |
| $0<\varepsilon<\varepsilon_{\min}$ | continuation | $P\propto\varepsilon$, counted `esp<` (W3(a): ratio 0.500000, 5 counts) |
| $\rho<\rho_{\min}$ | `p_over_rho`: exact $P\propto\rho$; `p`: continuation (legacy: frozen, ×2 at $0.5\rho_{\min}$) | counted `rho<` (W3(d)) |
| $\varepsilon>\varepsilon_{\max}$, $T>T_{\max}$, $\rho>\rho_{\max}$ | edge clamp | counted `esp>`/`T>`/`rho>` (W3(f)) |
| $P>P(\rho,\varepsilon_{\max})$ in `EspFromP` | bisection pinned at the top | $\varepsilon\approx\varepsilon_{\max}$, counted `esp>` (W3(f); silent up to `a40eff5`, O-1) |
| $P<P(\rho,\varepsilon_{\min})$ in `EspFromP` | exact continuation inverse | counted `esp<` |
| $c_s^2$ at a flat top segment | `Cs2of` returns the tabulated (floored $10^{-12}$) value | — |
| radiation $d\varepsilon\le0$ | Fleck factor disabled | explicit exchange |

A clamp counter that fires on **cell states** means the table does not cover the run; the per-cycle report says so
and names the edge ([code][clamp-report]). E10 found the production runs inside the table, with only transient
`esp<` lookups.

## A10. Configuration parameters

Defaults are the values passed to `GetOrAdd*` in the source; "required" means the key is read with `Get*` and the
run aborts without it.

| parameter | meaning | default | units | accepted | effect | source |
|---|---|---|---|---|---|---|
| `hydro/eos` | EOS choice | required | — | `adiabatic`, `hydrogen` | selects the ideal gas or the table | [hydro.cpp][hydro-eos] |
| `hydro/gamma` | ratio of specific heats | required | — | > 1 | ideal gas: the EOS. Table: sign of $P$ for $e_{\rm int}\le0$ and the `collapse_be` barotropic index (up to `a40eff5` also the `Tceil` conversion and the FOFC energy, O-2/O-3) | [hydro.cpp][hydro-eos] |
| `hydro/eos_table_file` | table path | none (`eos_table_hires_v5_cold.bin` in the run directory only if `eos_table_allow_default`) | — | v1 or v2 file | the table | [hydro.cpp][hydro-eos] |
| `hydro/eos_table_allow_default` | permit the default file name | false | — | bool | — | [hydro.cpp][hydro-eos] |
| `hydro/eos_interp` | what is interpolated along ρ | `p_over_rho` | — | `p`, `p_over_rho` | A7.2, E-F1 | [hydro.cpp][hydro-eos] |
| `hydro/eos_extrapolate` | ideal-gas continuation outside the table | true (false and refused in the legacy build) | — | bool | A7.2–A7.4, E-F5/E-F6 | [hydro.cpp][hydro-eos] |
| `hydro/eos_bisections` | P → ε bisection steps | 20 | — | integer in [10, 60] | A7.3 | [hydro.cpp][hydro-eos] |
| `hydro/riemann` | Riemann solver | required | — | `hlld` with the table | A8.3 | [hydro.cpp][hydro-hlld] |
| `hydro/eos_floor_consistent` | floored cell stores the pressure of its energy | false | — | bool | A8.2 step 7 | [hydro.cpp][hydro-ctor] |
| `hydro/eos_clamp_report` | per-cycle out-of-table report | true | — | bool | printing only | [hydro.cpp][hydro-ctor] |
| `hydro/dfloor` | density floor | −1 (off) | code density | > 0 to enable | A8.2 step 1 | [hydro.cpp][hydro-tfloor] |
| `hydro/pfloor` | pressure floor | −1 (off) | code pressure | > 0 to enable | A8.2 step 7 | [hydro.cpp][hydro-tfloor] |
| `hydro/Tfloor` | temperature floor | −1 (off) | **code T** (`hydrogen`) / K (`adiabatic`, needs `<units>` + `He_mass_fraction`) | > 0 to enable | floor energy $e(\rho,T_{\rm floor})$ at each density (table); A8.2 step 8 | [hydro.cpp][hydro-tfloor] |
| `hydro/Tceil` | temperature ceiling | $\infty$ | **code T** (`hydrogen`) / K (`adiabatic`, needs `<units>` + `He_mass_fraction`) | finite and > 0 to enable | ceiling energy $e(\rho,T_{\rm ceil})$ at each density (table); A8.2 step 9. Up to `a40eff5`: kelvin and ideal-gas conversion also under the table (O-3) | [hydro.cpp][hydro-tceil] |
| `hydro/He_mass_fraction` | composition of the **ideal-gas** $\bar m$ | none | — | — | `mbar_over_kb` for the ideal `Tfloor`/`Tceil` only; the table's $X=0.716$ is fixed in the generator | [hydro.cpp][hydro-he] |
| `units/rho_unit_cgs`, `v_unit_cgs`, `length_unit_cgs` | base scales | 5.467e-19, 1.9e4, 2.81e16 | cgs | refused for `collapse_be` | table conversion at load | [physical_units.hpp][units-build] |
| `units/mu_thermal` | temperature calibration | 2.29 | — | — | $T_u$ (A5.4) | [physical_units.hpp][units-build] |
| `problem/collapse_be/eint_over_p_override` | constant $e_{\rm int}/p$ for the IC | −1 (use the table) | — | > 0 to enable | A5.4 | [collapse_be.cpp][be-ic] |
| `problem/collapse_be/table_isothermal` | what the isothermal $T_0$ of a table run keeps | `pressure` | — | `pressure`, `temperature` | `pressure`: $P=\rho$ (BE equilibrium, 10.19 K); `temperature`: $T_0=T_u$ (behaviour up to `a40eff5`); A5.4, O-4 | [collapse_be.cpp][be-t0] |
| `hydro/double_flux`, `double_flux_kappa`, `double_flux_ghost_fill`, `double_flux_fill_sensor` | contact fix (opt-in) | false, 0.2, true, true | — | requires `eos = hydrogen`, HLLD, VL2 | Sec. 6.4–6.6 | [hydro.cpp][hydro-dflux] |
| CMake `ATHENAPK_EOS_LEGACY_LOOKUPS` | verbatim pre-fix lookups | OFF | — | ON/OFF | E-F9; refuses `eos_extrapolate = true` | [CMakeLists.txt][cmake] |

The radiation package reads the same `hydro/eos_*` keys for its own table copy ([code][rad-load]); Parthenon
requires one default per key, so the two load sites use identical defaults.

## A11. Worked examples, computed by the real code

All three examples were **executed**: the table side by
[`host/worked_examples.cpp`](host/worked_examples.cpp), which compiles the real `src/eos/eos_table.hpp` on the host
exactly as the E5 harness does (run: `host/run_worked.sh`, output
[`results/tables/W_worked_examples.txt`](results/tables/W_worked_examples.txt)); the physics side by
[`scripts/worked_saha.py`](scripts/worked_saha.py), which calls the generator's own `solve_saha` and `P_e_of_rho_T`
(output [`results/tables/W_worked_saha.txt`](results/tables/W_worked_saha.txt)). Configuration: the shipped
`eos_table_hires_v5_cold_rho26.bin` (md5 `cb68bd0f`), default keys (`p_over_rho`, `eos_extrapolate = true`, 20
bisections), default units. Hand calculations below are labelled; every other number is copied from those files.

### W1. An ordinary state: the reference gas of the code units

**Input.** $\rho_{\rm code}=1$ ($5.467\times10^{-19}$ g cm$^{-3}$), $T=T_u=10.015144$ K.

**Physics (generator).** $n_{\rm H,tot}=X\rho/m_H=0.716\times5.467\times10^{-19}/1.6726\times10^{-24}=2.340\times10^5$ cm$^{-3}$
[hand]; at 10 K $K_d$ underflows, so all H is in H$_2$: $n_{\rm H_2}=1.1701\times10^5$, $n_{\rm He}=2.3206\times10^4$,
$n_{\rm tot}=1.402196\times10^5$ cm$^{-3}$, $\mu=\rho/(m_Hn_{\rm tot})=2.331002$. Rotation and vibration are frozen
($n_{\rm H_2}\epsilon_{\rm rot}=6\times10^{-31}$ erg cm$^{-3}$ against a total $2.9\times10^{-10}$), so
$e_{\rm int}=\tfrac32n_{\rm tot}k_BT=2.908307\times10^{-10}$ erg cm$^{-3}$ and $P=n_{\rm tot}k_BT=1.938872\times10^{-10}$
erg cm$^{-3}$: $e_{\rm int}/P=1.5$, $\Gamma_1=5/3$. In code units $\varepsilon=e_{\rm int}/(\rho v_u^2)=1.473615$ and
$P=0.982410$.

**The lookup `EintFromRhoTk(1, 10.015144)`.** $\log_{10}\rho_{\rm code}=0$, so
$f_i=(0-(-7.752787))/0.050125=154.668$ → $i=154$, $t_i=0.668$; $f_j=(\log_{10}10.015144+1.336642)/0.004977=469.603$
→ $j=469$, $t_j=0.603$. The four nodes are $1.4634683$ (both $\rho$ nodes at $j$) and $1.4803367$ (both at $j+1$): at
10 K the specific energy does not depend on density. Bilinear:
$\varepsilon=1.4634683+0.602894\times(1.4803367-1.4634683)=1.4736382$ [hand; the code prints 1.4736381505], i.e. $e_{\rm int}=1.4736382$ (code) and $5.3198\times10^8$ erg g$^{-1}$.
Table vs generator: $+1.6\times10^{-5}$.

**`Pof(1, 1.4736382)`** (p_over_rho): stencil $j=417$, $t_j=0.611$; nodes $P/\rho_i=0.97517798$ and $0.98707328$; the
product with $\rho=1$ gives $P=0.98244421$ code $=1.938939\times10^{-10}$ erg cm$^{-3}$, $P/\rho=0.98244$ — the
$\mu_{\rm th}/\mu$ of A5.4. **`Cs2of`**: $c_s^2=1.6395886$ (code), $c_s=2.4329\times10^4$ cm s$^{-1}$,
$\Gamma_1=1.668887$ (A9.2: +0.133 % from the $\rho$ spacing). `LogTof` returns 10.015301 K ($+1.6\times10^{-5}$,
E-F7). **`EspFromP(1, P)`**: $1.473630061$ with 20 bisections ($-5.5\times10^{-6}$, inside the $8.08\times10^{-6}$
bound), $-3.0\times10^{-9}$ with 30, $-1.1\times10^{-16}$ with 60. No counter fired.

**Reproduce.** `cd validation/eos_tabulated/host && REPO=<clone> ./run_worked.sh`.

### W2. A state in H$_2$ dissociation: the softest point of the EOS

**Input.** $\rho=10^{-9}$ g cm$^{-3}$ ($\rho_{\rm code}=1.829157\times10^9$), $T=2000$ K.

**Physics.** 10.2 % of H nuclei are atomic, $2n_{\rm H_2}/n_{\rm H,tot}=0.8977$, ionization is negligible
($n_{\rm H^+}/n_{\rm H,tot}=1.7\times10^{-15}$), $\mu=2.1476$. The internal energy is now dominated by chemistry:

| term | erg cm$^{-3}$ | share |
|---|---|---|
| translational $\tfrac32n_{\rm tot}k_BT$ | 115.31 | 35.0 % |
| H$_2$ rotation | 48.90 | 14.8 % |
| H$_2$ vibration | 8.38 | 2.5 % |
| dissociation $\tfrac12\chi_d(n_{\rm H}+n_{\rm H^+})$ | 157.14 | 47.7 % |

$e_{\rm int}=329.72$, $P=76.870$ erg cm$^{-3}$; $e_{\rm int}/P=4.29$ and $1+P/e_{\rm int}=1.233$, while the adiabatic index
from the exact Saha EOS is $\Gamma_1=1.100951$: compressing the gas mostly dissociates more H$_2$ instead of heating
it.

**Table.** $\varepsilon=914.15$ code (Saha 913.35: $+8.8\times10^{-4}$, the bilinear error where $\varepsilon(T)$ is
strongly curved); $P=3.89623\times10^{11}$ code (Saha $3.89496\times10^{11}$); $\Gamma_1=1.102987$ (+0.18 %); T round
trip $+1.0\times10^{-4}$; `EspFromP` $-6.4\times10^{-6}$ (20 steps). W2b ($10^{-8}$ g cm$^{-3}$, 3000 K, 89 % atomic):
dissociation holds 82.6 % of $e_{\rm int}$, $\mu=1.335$, $\Gamma_1=1.1672$ (Saha) / 1.1699 (table).

### W3. Edges, floors and safeguards (real branches)

At $\rho_{\rm code}=1$: $\varepsilon_{\min}=9.3147\times10^{-3}$ code ($T=0.0633$ K), $P(\rho,\varepsilon_{\min})=6.2098\times10^{-3}$.

* **(a) Sub-edge energy**, $\varepsilon=\tfrac12\varepsilon_{\min}$: the continuation gives
  $P=3.1049\times10^{-3}$ (ratio 0.500000), the legacy clamp $6.2098\times10^{-3}$ — twice the pressure the energy
  holds (E-F5). $T=0.0317$ K. Counters: `esp<` 5.
* **(b) Pressure floor** $P_{\rm fl}=10^{-8}$ through `EintFromRhoPres` (the call `InternalEnergyFromPres` makes):
  hand [continuation inverse] $\varepsilon=\varepsilon_{\min}P_{\rm fl}/P_{\min}=9.3147\times10^{-3}\times10^{-8}/6.2098\times10^{-3}=1.4999811\times10^{-8}$;
  code $1.4999812\times10^{-8}$, and $P(\varepsilon)=P_{\rm fl}$ to 12 digits, at $T=1.0\times10^{-7}$ K. Legacy:
  $\varepsilon=\varepsilon_{\min}(1+8.1\times10^{-6})$ (the bisection, pinned at the bottom of the axis, returns the
  midpoint of its last bracket) with $P=6.2\times10^5P_{\rm fl}$.
* **(c) Temperature floor** `hydro/Tfloor = 0.2726` (code) $=2.730128$ K: `EintFromRhoTkHost(1, T)` $=0.40171153$ code;
  the floor pressure $P(\rho,\rho\varepsilon_{\rm fl})/\rho=0.26781559$ and $T=2.730156$ K at $\rho_{\rm code}=10^{-6}$,
  1 and $10^6$ alike: at 2.7 K $\varepsilon$ does not depend on $\rho$, so the floor energy taken at
  $\rho_{\rm code}=1$ (the behaviour up to `a40eff5`) and the per-density floor of `5649acd` agree; in runs they differ
  in the last bit (1.8e-16). The E-F5 probe measured $P/\rho=0.26782$ for the same floor in a run.
* **(d) Density below the table**, $\rho=\tfrac12\rho_{\min}$, $\varepsilon$ of 10 K: $P/\rho=0.98096$ with
  `p_over_rho` and with `p` + continuation; `p` legacy clamp 1.96192 (×2, E-F6). Counted `rho<`.
* **(e) The E-F1 bias**: A7.2.
* **(f) Upper edges**: $\varepsilon_{\max}=2.1445\times10^5$ code ($T=3\times10^5$ K); `Pof` at $2\varepsilon_{\max}$
  returns $P(\varepsilon_{\max})$ (counted `esp>`); `EintFromRhoTk` at $10^6$ K returns $\varepsilon(T_{\max})$ (counted
  `T>`); `EspFromP` at $10P(\varepsilon_{\max})$ returns $0.999992\varepsilon_{\max}$ and is counted `esp>` (silent up to
  `a40eff5`, O-1); a target $0.999\,P(\varepsilon_{\max})$ is inside the table and not counted. The lookup exactly at
  $\varepsilon_{\max}$ is no longer counted (O-7).
* **(g) A temperature floor where $\varepsilon(T)$ depends on density** (O-5): for a 2000 K floor, the energy taken at
  $\rho_{\rm code}=1$ and scaled with $\rho$ corresponds to 1997.8 / 2175.7 / 2779.5 / 4005.7 K at
  $10^{-16}$ / $10^{-12}$ / $10^{-9}$ / $10^{-6}$ g cm$^{-3}$; $e(\rho,T_{\rm floor})$ at each density to
  1997.7 / 1999.9 / 2000.2 / 2000.0 K (the residual is the E-F7 round trip).

## A12. The EOS in one paragraph

A cell's conserved state gives $\rho$ and, by subtraction, $e_{\rm int}$. `ConsToPrim` turns $(\rho,e_{\rm int}/\rho)$
into $P$ with one bilinear lookup of $P/\rho_i$ on a logarithmic grid that was precomputed from a Saha equilibrium of
H$_2$/H/H$^+$/He/He$^+$/He$^{2+}$/e$^-$ with H$_2$ rotation, vibration and dissociation energies; floors repair
non-physical cells. The reconstruction carries $P$ to the faces; HLLD converts it back to energy with a 20-step
bisection of the same lookup and uses the tabulated $c_s^2$ (a finite-difference derivative of the tabulated $P$) in
the fast magnetosonic speed for its wave speeds and for the time step. Temperatures, when another package needs
them, come from a second lookup on the same grid ($T$ in kelvin), and energies from a temperature from a third
table on a $(\rho,T)$ grid. Everything outside the table is either continued as the ideal gas it is there (cold, dilute
side) or clamped and counted (hot, dense side).

---

# Part B — Validation

## 1. Scope

What is validated: `src/eos/eos_table.hpp` (lookup, inversion, clamp instrumentation), the table
`eos_table_hires_v5_cold.bin` and its generator `src/eos/gen_eos_table.py`, and the coupling through
`AdiabaticGLMMHDEOS` (`src/eos/adiabatic_glmmhd.hpp`: ConsToPrim, floors, `eos_floor_consistent`) to the
glmmhd/HLLD/PLM/VL2 solver. Not in scope: any other physics of the production runs.

## 2. Code under test

### 2.1 Production configuration
`eos = hydrogen`, `eos_table_file = eos_table_hires_v5_cold.bin` (md5 `f712d6a8`),
`riemann = hlld`, `reconstruction = plm`, `integrator = vl2`, `eos_floor_consistent = true`, `eos_bisections` 20
(default), `pfloor = 1e-8`, `dfloor = 1e-6` (code units), units rho0 = 5.467e-19 g/cc, v0 = 1.9e4 cm/s.
`hydro/eos_interp`: default `p_over_rho` (E-F1); `p` reproduces runs made before the fix.

Table: 400 log-rho nodes over [1e-20, 1] g/cc (0.0501 dex), 1399 log-e nodes over 10^6.527..10^13.889 erg/g
(0.0053 dex) for P, cs2, log T; 1370 log-T nodes over 0.046 K..3e5 K (0.0050 dex) for e(rho, T).
Lookups are bilinear in (log rho, log e) / (log rho, log T), edge-clamped and counted (`eos_clamp_report`).
P -> e is a fixed 20-step bisection on the log-e axis.

### 2.2 Source and binaries
The tests ran on CPU (Kokkos OpenMP back end, one thread per MPI rank) and on one H100 GPU, with binaries built from
the private research tree whose EOS code is the one shipped here: `eos/eos_table.hpp` (incl. the below-table
continuation `hydro/eos_extrapolate` and, under the CMake switch `ATHENAPK_EOS_LEGACY_LOOKUPS`, the verbatim legacy
lookups), `eos/adiabatic_glmmhd.*`, `hydro/hydro.cpp` (key reads; `hydro/Tfloor` for the tabulated EOS),
`diagnostics/cons_diag.*` (`cons-nTfloor`), `hydro/double_flux.{hpp,cpp}` with its hooks in `hydro_driver.cpp` and
the two null-default Parthenon hooks of Sec. 6.6, and the `eos_test` problem generator (`uL/uR`,
`espL_code/espR_code`, modes wave/riemann/compress/slab, the `eos_homologous` x1 BC). The binary hashes quoted below
identify the binary of each run (recorded in its `cmd.txt`; run directories are not shipped). The final validated
CPU binary is `85b625bf`.

**Extraction check (2026-10-07).** This repository's code was compared with the validated binary (`85b625bf`) on the
same decks and MPI ranks: 47 paired runs covering the self-gravity final-configuration legs (S1, D1, D2, D4, D7), three
`collapse_be` runs (deep static AMR solve, BE stability, magnetized rotating collapse to 300 cycles), the EOS 1D legs
(E6, E7e with and without double flux, E8, E10), the refined-mesh double-flux + fill legs (E7r, E7x) and 2D/3D slabs
(E7q). 12 are bitwise identical as built (incl. S1, D1, D2 and the deep-AMR solve); the rest differ at round-off
level (max |diff| / max |field| <= 1.6e-5 at moving contacts of the standard scheme, <= 2e-10 elsewhere; L1 <= 3e-6),
the same GCC FMA-contraction effect as E-F9 in the EOS report: with both trees compiled with `-ffp-contract=off`, the
28 legs with the largest differences (E7e, E7r, E8, D4, D7, L10) are BITWISE identical (the E10 probes, <= 3e-18, and
the E7q slabs, <= 8e-14, were not rerun). The history files differ only by one diagnostic column that is not part of
this code.

## 3. References and tools

| tool | what | checked |
|---|---|---|
| `scripts/eostab.py` | the v2 table in code units; bilinear exactly as the runtime | E6 predictions = measurements to 1e-4 |
| `host/e5_inversion.cpp` (+ `shims/`, `run_e5.sh`) | compiles the REAL `eos_table.hpp` on the host (KOKKOS_INLINE_FUNCTION -> inline), so E5 runs the kernels' arithmetic | — |
| `scripts/riemann_exact.py` | exact Riemann solver for any P(rho, e) (Colella & Glaz 1985): rarefactions = RK4 along the isentrope de = P/rho^2 drho, du = c/rho drho, c^2 = P_rho + P P_e/rho^2 of the interpolant; shocks = Rankine-Hugoniot root; vectorised fan sampler | Toro (2009) tests 1-5 to his printed digits; Sod fan vs analytic 2.3e-10; step refinement 2e-4 -> 5e-5 moves P* by <= 2.8e-6 on R2/R4 |
| `scripts/e7_hugoniot_check.py` | Hugoniot single-valuedness (all roots in rho at each P) and Liu's chord condition with exact shock states | Sec. 6.1 |
| `scripts/analyze_E7.py` / `analyze_E8.py` / `analyze_E9.py` / `analyze_E10.py`, `e7f_eval.py` | analyses (Sec. 13) | — |

The reference for the hydro tests is always the RUNTIME INTERPOLANT (mode `p` or `p_over_rho`), because a
conservative scheme converges to the weak solution of the equations it actually solves; the interpolant's
deviation from the physical EOS is measured separately (E2, E-F1).

---

## 4. Group A — the table

**E1 provenance.** `gen_eos_table.py build 400 1000 920 <out.h5> 399 450` rebuilds the production table
byte-identically (6.6 min, `runs/E1_rebuild`).

**E2 thermodynamic gates** (`scripts/test_eos_consistency.py`, `results/tables/E2_consistency_v5cold.txt`):
charge neutrality 5.05e-15, element conservation 2.2e-16, Maxwell identity -rho^2 (du/drho)_T = T (dP/dT)_rho - P
to 2.46e-3 (tol 3e-3, finite-difference limited — corrected by T-4, Part A Sec. A3.5: the residual does not depend on
the step and is the ortho/para modelling inconsistency of the generator), table P vs direct Saha median 1.56e-3 / max 5.5e-3 (7454 K,
1e-14 g/cc), cs2 vs Saha median 1.55e-3 / max 3.3e-2 (9047 K, 1e-13: a kink).

**E3 limits.** Gamma1 = 1.668 (10 K), 1.597 (100 K), 1.428 (500 K), 1.101 (2000 K, 1e-9 g/cc), 1.145 (1e4 K,
1e-8 g/cc).

**E5 inversions** (host harness on the real header, `results/tables/E5_inversion.txt`; identical in both modes
except where stated):

| check | result |
|---|---|
| P(rho, e) increasing in e (needed by the bisection) | strictly increasing everywhere below 3e5 K; 529 of 559 200 node pairs are EQUAL (not decreasing): the generator's T cap (3e5 K) for rho > 6e-3 g/cc. log T(rho, e) has the same plateau. e(rho, T) strictly increasing (0 violations) |
| P -> e bisection (20 steps) | max relative e error 8.08e-6 = the analytic bound 0.5 x 7.362 dex x 2^-20; resulting P error <= 9.2e-5 (dissociation/ionization, where dlnP/dlne is large); 30 steps 7.9e-9, 40 7.7e-12, 60 5e-15 |
| T -> e -> T round trip (797 x 1201 off-node points, 0.1 K-2.9e5 K) | median 1.2e-5..7.9e-5 by regime; max 1.6e-5 (< 100 K), 1.7e-4 (100-1e3 K), 1.3e-2 (dissociation, worst at 1e-19 g/cc, 1377 K), 1.2e-2 (H ionization), 7e-4 (> 3e4 K). Production footprint (rho <= 1e-10, 1-200 K): max 2.0e-5 -> E-F7 |
| clamp counters over all interior sweeps | 0 |
| rho below the table, T = 10 K | `p`: P frozen at the edge value (x2 too high at 0.5 rho_edge, x1e4 at 1e-4 rho_edge); `p_over_rho`: exactly the ideal-gas continuation rho (P/rho)_edge (ratio 1.000000); counted as `rho<` -> E-F6 |
| P below P(rho, e_min) (EspFromP) | returns e_min (1 + 8.1e-6), i.e. T = 0.063 K, and is NOT counted (the bisection runs with counting off; only the rho axis is tallied) -> E-F5 |
| 0 < e < e_min | P = P(rho, e_min) whatever e is (1e-9 e_min gives the same P); counted as `esp<` |
| production pfloor 1e-8 through InternalEnergyFromPres | e_fl = e_min (1 + 8.1e-6) at every density; P(e_fl) = 1.4e4 x pfloor at 1.3e-20 g/cc up to 1.1e19 x pfloor at 1e-5 g/cc; 0 clamps -> E-F5 |

---

## 5. Group B1 — acoustic waves (E6)

Planar eigenmode (amplitude 1e-6) on five states, one period, N = 16-256 (`runs/E6_wave`,
`results/tables/E6_waves*.txt`). Amplitude damping converges at >= 3rd order. With the legacy `p` interpolation the
phase speed is off by -1.40 / -0.85 / -0.46 / +0.16 / +0.55 % (10 K / 100 K / 500 K / 2000 K / 1e4 K) independent of N,
and equal to the interpolant's own sound speed to 1e-4 (E-F1). With `p_over_rho` at N = 256: -0.059 / -0.015 /
-0.12 / -0.16 / -0.04 %.

## 6. Group B2 — general-EOS Riemann problems (E7)

### 6.1 Problems and their exact solutions
States (rho [g/cc], T [K], u [km/s]), 1D, outflow, x0 = 0.5, t such that the fastest wave travels 0.4:

| problem | left | right | exact (p_over_rho): P* [code], u* [code], waves, T* L/R |
|---|---|---|---|
| R1 molecular (rotational excitation) | 1e-14, 300, 0 | 1e-15, 30, 0 | 1.0842e5, 6.3948; rarefaction/shock; 182 / 166 K |
| R2 dissociation (driver partly dissociated) | 1e-9, 4000, 0 | 1e-10, 1000, 0 | 2.4962e11, 32.806; rarefaction through recombination / shock; 2581 / 1947 K |
| R3 dissociating shock (colliding streams) | 1e-10, 100, +16 | 1e-10, 100, -16 | 1.3701e12, 0; two shocks, compression 19.2, 2528 K |
| R4 ionization (ionized driver) | 1e-8, 3e4, 0 | 1e-9, 6000, 0 | 3.4927e13, 125.97; rarefaction through H recombination / shock; 17 405 / 10 076 K |

Admissibility (`results/tables/E7_hugoniot_check.txt`, both modes): every shock branch is single-valued in P (one
rho root at each of 400 pressures up to 3 P*), Liu's chord condition holds strictly (one sign over 150 exact
states), every rarefaction is convex along its path (fundamental derivative G_min = 1.137 / 0.999 / 1.117 for
R1/R2/R4 with `p_over_rho`, 0.77 / 0.61 / 0.70 with `p` — the E-F1 sawtooth). R3's Hugoniot is non-monotone in
rho only ABOVE P* (957-1500 P0, where dissociation completes), outside the solution. At +-8 km/s the shocked gas
only reached 1995 K, hence +-16 km/s.

### 6.2 Results (`results/tables/E7e_riemann.txt`, figure `results/figures/E7e_riemann.png`)
L1 errors normalised by max|q|, exact cell averages from 25 600 sub-cells; plateau = median over the middle half
of each star region; N = 100 ... 3200.

| problem (p_over_rho) | L1(rho) N = 400 / 3200, order | plateau P* error N = 400 / 1600 / 3200 | shock position |
|---|---|---|---|
| R1 | 1.43e-3 / 2.04e-4, 0.87-1.00 | +3.9e-5 / +1.3e-6 / -2.0e-7 | +0.53 cells, constant |
| R2 | 4.21e-3 / 7.60e-4, 0.81-0.85 | -5.2e-4 / -2.6e-4 / -1.7e-4 | +0.10 -> -0.05 cells |
| R3 | 1.83e-3 / 2.28e-4, 0.95-1.01 | -5.3e-5 / +2.2e-6 / +6.5e-7 | -0.03 / +0.03 cells, constant |
| R4 | 5.44e-3 / 1.07e-3, 0.66-0.80 | -1.8e-3 / -8.7e-4 / -5.4e-4 | -0.22 / -0.52 / -0.78 cells |

![E7](results/figures/E7e_riemann.png)

The `p` legs behave the same against their own exact solutions (R1 P* +1.3e-5, R3 +5.6e-7 at N = 3200); their
L1 orders are lower at moderate N (0.6-0.9) because the sawtooth sound speed perturbs the fans.

### 6.3 Slow convergence through recombination
R1 and R3 (molecular rarefaction; shocks) converge to the exact plateaus at the 1e-7 level. R2 and R4, whose
rarefactions pass through H2 resp. H recombination, converge at order ~0.5-0.7 in the plateau and their shock
positions drift by a slowly shrinking fraction of the box. The reference is not the cause (P* moves by <= 2.8e-6
when the isentrope step is refined 4x). Experiment E7f (`runs/E7f`, `results/tables/E7f_bisect.txt`, p_over_rho):

| | N = 400 | 800 | 1600 | 3200 | 6400 | 12800 |
|---|---|---|---|---|---|---|
| R4 plateau P* error | -1.79e-3 | -1.11e-3 | -8.72e-4 | -5.35e-4 | -3.40e-4 | -2.19e-4 |
| R2 plateau P* error | -5.20e-4 | -3.77e-4 | -2.59e-4 | -1.72e-4 | -1.09e-4 | -6.8e-5 |

Monotone convergence at order ~0.65 out to N = 12800 (ratios 1.56-1.61 per doubling at the top): the scheme
converges to the exact solution, slowly. One change at a time on R4, N = 1600 (reference -8.72e-4): 60 bisections
-8.75e-4, CFL 0.1 -8.39e-4, RK2 integrator -8.70e-4 — not the bisection, the time step or the integrator.

**Is it the table? No (2026-10-05, four falsification tests).** The error profiles (`results/figures/
E7g_error_profiles.png`) show spikes at grid lines the fan crosses, and the interpolant's slope d lnP/d ln e JUMPS at
e nodes by up to 1.16 (R2) and 0.42 (R4) where P(e) is strongly curved through dissociation/ionization (R1: 0.010;
`results/tables/E7g_kinks2.txt`; the tabulated node values themselves are smooth). But removing those kinks does not
change the plateau error (R4, P* error at N = 3200 / 6400):

| interpolant | R4 | R2 |
|---|---|---|
| production table, bilinear | -5.35e-4 / -3.40e-4 | -1.72e-4 / -1.09e-4 |
| C1 along log rho (Catmull-Rom; E7c) | -5.34e-4 / -3.45e-4 | -1.72e-4 / -1.09e-4 |
| C1 along log rho AND log e (monotone Hermite; E7d) | -5.78e-4 / -3.62e-4 | -1.68e-4 / -1.07e-4 |
| table rebuilt with 4x the rho nodes (E7g r4) | -5.37e-4 / -3.47e-4 | -1.71e-4 / -1.09e-4 |
| table rebuilt with 2x the nodes on every axis (E7g x2) | -5.55e-4 / -3.59e-4 | -1.75e-4 / -1.09e-4 |

(each against the exact solution of its own interpolant/table). The C1 interpolant was verified on the real header
(P strictly increasing in e at 39.9 M points; slope jumps 9.8 -> 2.4e-4 along e, 2.5e-2 -> 1e-6 along rho) and then
REMOVED from the code: it is not the remedy.

**It is the reconstruction (E7h/E7i, `results/tables/E7h_scheme.txt`, `E7i_limo3.txt`).** R4, P* error at
N = 800 / 1600 / 3200 / 6400 and the L1(rho) order:

| scheme | P* error | L1 order |
|---|---|---|
| PLM, VL2 + PLM predictor (production) | -1.1e-3 / -8.7e-4 / -5.4e-4 / -3.4e-4 | ~0.75 |
| PLM, VL2 + donor-cell predictor | -6.7e-4 / -5.0e-4 / -3.3e-4 / -2.0e-4 | ~0.85 |
| PLM, RK3 | -1.1e-3 / -8.5e-4 / -5.1e-4 / -3.4e-4 | ~0.8 |
| PPM | -9.4e-4 / -6.5e-4 / -4.4e-4 / -3.6e-4 | ~0.7 |
| WENO-Z | -5.7e-4 / -2.7e-4 / -2.3e-4 / -1.8e-4 | ~0.85 |
| donor cell | -1.4e-2 / -8.7e-3 / -5.8e-3 / -4.0e-3 | ~0.5 |
| **LimO3** (Cada & Torrilhon 2009, nghost 3) | **-4.0e-5 / -4.7e-5 / -8.4e-5 / -3.9e-5** | **0.93-1.0** |

LimO3 on the other problems: R2 -8.6e-5 / -5.4e-5 / -3.6e-5 at 1600 / 3200 / 6400 (PLM -2.6e-4 / -1.7e-4 / -1.1e-4),
L1 order 0.85-0.92; R1 and R3 unchanged (plateaus ~1e-6, L1 ~10 % lower than PLM).

**The mechanism: spurious pressure at the moving contact (E7j/E7k).** The fan itself is accurate. In the left star
region the specific entropy and the Riemann invariant J are those of the exact solution to a median of ~1e-6
(R4 PLM, N = 1600 / 3200 / 6400: dJ/|u*| = -5.9e-6 / -3.0e-6 / -5.6e-7; `E7j_entropy.txt`, `E7j_invariant.txt`), so
the error does not come from the rarefaction. E7k isolates the contact: a pure contact between the two star states
of each problem (equal P and u, density and composition jump), advected (`mov`) or at rest
(`results/tables/E7k_contact.txt`; L1 / max of |P/P* - 1|):

| contact | N = 400 | 800 | 1600 | at rest (any N) |
|---|---|---|---|---|
| R4 (ionization), PLM | 2.8e-3 / 4.2e-2 | 1.8e-3 / 3.7e-2 | 1.1e-3 / 3.2e-2 | 4e-15 / 1e-14 |
| R4, LimO3 | 2.0e-3 / 3.9e-2 | 1.1e-3 / 3.4e-2 | 6.3e-4 / 3.0e-2 | 4e-15 / 1e-14 |
| R2 (dissociation), PLM | 7.7e-4 / 1.1e-2 | 4.8e-4 / 9.6e-3 | 3.0e-4 / 8.3e-3 | 5e-16 / 7e-15 |
| R1 (molecular), PLM | 5.1e-6 / 3.4e-5 | 3.5e-6 / 3.2e-5 | 3.2e-6 / 2.2e-5 | 7e-16 / 5e-15 |

At rest the contact is exact; moving, the cells that straddle it carry up to 3-4 % (R4) and 1 % (R2) spurious
pressure, and L1 converges as ~N^-0.7, the plateau's order. This is the known failure of conservative schemes at
interfaces with a nonlinear EOS (Abgrall 1996): the update mixes (rho, rho e) LINEARLY, and where Gamma varies
(dissociation, ionization) a linear mix of two states at the same P is not at that P. The pressure blips launch
acoustic waves into the plateaus. Molecular gas (R1, nearly ideal) shows 2e-5. LimO3 only sharpens the contact
(1.5-1.7x lower L1, the same maximum); it does not remove the mechanism. Not a defect of the EOS or the table.
LimO3 (`hydro/reconstruction = limo3` + `parthenon/mesh/nghost = 3`, 1.18x cells per 32^3 block) stays a
possible production change; the fix WITH PLM is Sec. 6.4. Finding E-F10.

### 6.4 Double flux: pressure-exact contacts with PLM (opt-in, `hydro/double_flux`)

Goal: a fix that keeps PLM. Chosen: the double-flux method (Abgrall & Karni 2001,
JCP 169, 594; Billet & Abgrall 2003). Implementation `src/hydro/double_flux.{hpp,cpp}` (separate TU) and a
two-task hook in `hydro_driver.cpp`:

1. At step start every cell freezes a stiffened-gas fit of the table through its own state,
   rho e = P/(g* - 1) + rho e0*, with g* = rho c^2/P (table c^2) and e0* from the cell's conserved internal energy.
   With a frozen EOS, rho e is linear in rho at fixed P, so a linear mix of two states at equal P stays at that P.
2. Each stage: the face states are reconstructed exactly as the flux kernel does; HLLD's energy flux is affine in
   the two face energies (F_E = F0 + aL E_L + aR E_R, coefficients replicated from the solver's wave speeds and
   region), so the energy flux "seen" by a cell with its own frozen EOS is the standard flux plus
   aL (E_L^frozen - E_L^table) + aR (E_R^frozen - E_R^table). Mass, momentum and B fluxes are untouched.
3. In a double-flux cell, after the standard update: E += beta dt sum (own-EOS increments), then P is taken from the
   frozen EOS and E is rebuilt from the TABLE at that P (EspFromP + 3 Newton steps on the interpolant). Total
   energy is therefore not conserved exactly where g*, e0* vary; measured below.
4. Contact sensor, PER CELL: a cell is a double-flux cell iff every face has dP/P <= kappa drho/rho
   (`hydro/double_flux_kappa`, default 0.2), or both jumps are at round-off. Shocks and adiabatic flow have
   dP/P ~ Gamma1 drho/rho with Gamma1 >= 1.1; a contact has a density jump without a pressure jump. All other cells
   are exactly the standard conservative scheme.

Scope, refused at startup otherwise: `eos = hydrogen`, HLLD, PLM or donor cell, VL2, uniform mesh (the
increments are not AMR flux-corrected; lifted in Sec. 6.5-6.6), no first-order flux correction.

How the design got there (each version measured, `results/tables/E7m_dflux.txt`, `E7n_dflux_sensor.txt`,
`E7o_dflux_contact.txt`, `E7p_dflux_cell.txt`):

| version | contacts | R1 plateau dP* (N = 3200) | R3 / R2 / R4 | verdict |
|---|---|---|---|---|
| v1: everywhere (E7m, 04db22ca) | exact (1e-14) | -1.59e-3 at every N (400-1600; not run at 3200) | R4 +4.5e-3 / +4.2e-3 / +4.1e-3 at 400 / 800 / 1600: not converging | the frozen tangent EOS is wrong across shocks |
| v2: faces with dP/P < 0.1 (E7n, 783a5a10) | exact | -1.12e-3 at every N | R3 dE/E +7.5e-4 at every N | still acts in shock feet and rarefactions (INFERRED, not isolated: table c^2 is not exactly the table's own dP/drho at fixed s, so each frozen-EOS reset in smooth flow drifts by a step-independent amount) |
| v3: per-FACE contact sensor (E7o, 4f39c532) | exact | -1.9e-6 | R1/R3 = standard; R4 -1.8e-2 (1600), -2.9e-2 (3200): GROWS with N | own-EOS increments applied without the reset in cells with one contact face; faces flip on/off on the plateaus |
| **v4: per-CELL contact sensor (E7p, 54b1958b)** | **exact** | **-1.3e-6 (standard -2.0e-7)** | **converge, below** | **adopted** |

Results of v4 (E7p; standard PLM in brackets; dE/E = (E_dflux - E_std)/E_std of the whole domain at t_final):

| problem | N = 400 | 800 | 1600 | 3200 | 6400 |
|---|---|---|---|---|---|
| R4 plateau dP* | +1.8e-4 [-1.8e-3] | +1.6e-4 [-1.1e-3] | +1.3e-6 [-8.7e-4] | -3.2e-6 [-5.4e-4] | -1.2e-6 [-3.4e-4] |
| R4 L1(rho) | 5.4e-3 [5.4e-3] | 3.1e-3 [3.1e-3] | 1.78e-3 [1.82e-3] | 1.02e-3 [1.07e-3] | 5.9e-4 [6.3e-4] |
| R4 dE/E | +9.0e-4 | +6.2e-4 | +4.3e-4 | +2.9e-4 | +1.9e-4 |
| R2 plateau dP* | +2.6e-4 [-5.2e-4] | +2.7e-5 [-3.8e-4] | -5.5e-6 [-2.6e-4] | -2.6e-6 [-1.7e-4] | -3.7e-6 [-1.1e-4] |
| R2 L1(rho) | 4.2e-3 [4.2e-3] | 2.3e-3 [2.4e-3] | 1.32e-3 [1.35e-3] | 7.4e-4 [7.6e-4] | 4.5e-4 [4.6e-4] |
| R2 dE/E | +5.3e-4 | +3.4e-4 | +2.1e-4 | +1.4e-4 | +8.6e-5 |
| R1 plateau dP* | +2.6e-5 [+4.0e-5] | -2.2e-6 [+4.7e-6] | -2.0e-6 [+1.3e-6] | -1.3e-6 [-2.0e-7] | +4.1e-7 |
| R1 dE/E | -2.7e-6 | -1.9e-6 | -1.1e-6 | -5.8e-7 | — |
| R3 plateau dP* | -5.3e-5 [-5.3e-5] | -8.8e-6 [-8.9e-6] | +2.1e-6 [+2.2e-6] | +5.9e-7 [+6.5e-7] | — |
| R3 dE/E | +2.6e-6 | +1.3e-6 | +6.5e-7 | +3.3e-7 | — |
| pure contact R4, max \|P/P*-1\| | 3.2e-14 [4.2e-2] | 3.8e-14 [3.7e-2] | 4.8e-14 [3.2e-2] | 8.9e-14 [2.8e-2] | 1.4e-13 [2.3e-2] |
| pure contact R4 dE/E | +2.4e-3 | +1.5e-3 | +8.9e-4 | +5.5e-4 | +3.4e-4 |
| pure contact R2, max \|P/P*-1\| | 1.8e-14 [1.1e-2] | 4.9e-14 [9.6e-3] | 5.3e-14 [8.3e-3] | 9.4e-14 [7.2e-3] | 2.0e-13 [6.1e-3] |

So with PLM: moving contacts are pressure-exact to round-off; the R2/R4 plateaus reach the 1e-6 level (the level of
R1/R3 and of the reference solver) from N = 1600 on, 65-280x below standard PLM at N = 3200-6400; L1(rho), which
includes the shock positions, is equal to or up to 7 % below standard. The price is the energy: +1.9e-4 (R4) and
+8.6e-5 (R2) of the domain's total energy at N = 6400, converging at order ~0.6 (the width of the smeared contact),
always positive here; R1/R3 (no regime change at the contact) 1e-6.

Falsification and non-regression:
* Smooth acoustic waves must not trigger the sensor: E6 states S1-S5, N = 16-256, double flux on vs off with the
  same binary (`runs/E6_dflux`): all 25 final states BITWISE identical (the key was active: fields allocated, wall
  time higher). The sensor never fired.
* Key off: binary 54b1958b reproduces the previous default binary a15514d6 BITWISE on all 20 1D legs
  (`runs/EF56_regate`: E10 probes, E7e n400 x 8, E6 S1-S5, E8_leg2 61 outputs).
* Legacy build of the same source (`-DATHENAPK_EOS_LEGACY_LOOKUPS=ON`): on a 3D production restart (64 ranks, full
  production configuration, 3 cycles) BIT-IDENTICAL to the pre-fix binary a6870124.
* Wall time (1D, 50 concurrent runs on one node, noisy): 1.04-1.5x. Not a 3D cost measurement.

### 6.5 Double flux in 2D/3D, with mesh refinement and on the GPU (2026-10-06)

The items deferred in Sec. 6.4, in the order they were done. Final source: CPU
`37c90051` (bitwise = `fe0caa0f` on four double-flux runs, `runs/E7u_bitid37`), GPU `84aab51e` (same source).

**v6 design change (for refinement).** For a frozen EOS (g, e0) the own-EOS increment of a face energy flux is
X/(g-1) + e0 Y - Z with X = aL PL + aR PR, Y = aL rhoL + aR rhoR, Z = aL E_tab,L + aR E_tab,R. X, Y, Z are stored as
the fluxes of a field `dflux_xyz` ({Cell, Independent, WithFluxes, OperatorSplit}: flux-corrected with the hydro
fluxes, excluded from every {WithFluxes, Cell} pack and from the lean registers), so a coarse cell at a fine/coarse
face gets the increment of exactly the restricted flux it receives. The coefficient task now runs before the flux
correction is sent. v6 reproduces v4 in 1D (`results/tables/E7t_dflux_v6.txt`: contacts <= 1e-13; R2/R4 plateaus
within a few 1e-6 of v4; energies to 3 digits; R3 identical).

**Two bugs found and fixed on the way.**
* Scratch overrun (in the double-flux code itself): the coefficient kernel sized its reconstruction scratch rows by
  the hydro count, but `Reconstruct` writes every prim component; with any extra advected component (e.g. a
  passive scalar) it wrote past `wl` into the next array. Seen as 2.8e-4 differences in R2/R4 between runs with and
  without such a component; after sizing from `prim.GetDim(4)` the two are BITWISE identical. Production runs carry
  passive scalars, so this would have hit the first production use.
* nvcc rejects a device lambda that first captures a variable inside `if constexpr` (GPU build failed; CPU fine).

**2D and 3D, uniform mesh (E7q, `results/tables/E7q_slab.txt`).** New `eos_test` mode `slab`: periodic oblique slab of
pure contacts, wave vector (mx, my, mz) in box units, velocity along k (R4 star states). Exact solution: translation
at uniform P*.

| case | double flux max / L1 \|P/P*-1\| | standard | dE/E over the run (double flux) |
|---|---|---|---|
| 2D k = (1,1), N = 64 / 128 / 256 / 512 | 1.1e-14 / 2.0e-14 / 2.4e-14 / 1.1e-12 | 9.3e-2 / 6.9e-2 / 5.3e-2 / 4.4e-2 | +1.8e-2 / +1.1e-2 / +6.6e-3 / +4.1e-3 |
| 3D k = (1,1,1), N = 32 / 64 / 128 | 7e-15 / 2.0e-14 / 1.4e-14 | 8.4e-2 / 6.3e-2 / 5.2e-2 | +3.1e-2 / +1.9e-2 / +1.2e-2 |

Invariance (bitwise, key on and off): 2D N = 128 in 32^2 blocks on 4 ranks = one block; 3D N = 64 in 16^3 blocks on 8
ranks = one block; k = (1,0) on a 2D grid = the 1D run row by row (the x2 path adds nothing). The energy price is
larger than in the Riemann problems because half of the box is a slab bounded by two regime-changing contacts; it
converges at order ~0.7.

**Mesh refinement (E7r, `results/tables/E7r_amr.txt`).** R4 moving contact in 1D through a 2-level static patch
(smr) and with `maxdensity` adaptive refinement whose fine/coarse boundary rides on the contact (amr); oblique 2D slab
with a static patch and adaptive refinement. Max |P/P*-1| over 8 snapshots:

| case | standard | v5 (coarse cells at fine/coarse faces standard) | v6 (flux-corrected X, Y, Z) | v6 + piecewise-constant prolongation (E7v) |
|---|---|---|---|---|
| 1D smr | 6.4e-2 | 8.1e-3 | 1.35e-3 | **1.2e-13** |
| 1D amr | 5.3e-2 | 9.7e-4 | 5.5e-4 | 4.3e-4 |
| 2D smr | 1.2e-1 | 8.9e-2 | 5.1e-3 | 4.5e-4 |
| 2D amr | 8.5e-2 | 4.1e-2 | 4.0e-3 | 3.0e-5 |
| (1D uniform, same coarse N) | 6.4e-2 | 2e-14 | 2.8e-14 | — |

The residual on refined meshes is born where the contact crosses a fine/coarse boundary (error located at the
level jumps, then radiated as sound). E7v isolates it (validation-only key `hydro/cons_prolong_pc`): with
piecewise-constant prolongation of `cons` the 1D static case becomes exact, so the source is the minmod prolongation of
CONSERVED variables into ghost cells -- rho and E interpolated separately mix rho e linearly, the same Abgrall effect,
in cells that never get the double-flux reset. What remains with piecewise-constant prolongation comes from
restriction (2D) and from regridding block interiors (amr). OPEN: an EOS-consistent ghost fill (prolong P, set E from
the table at the prolongated P) would remove the prolongation part; restriction/regrid need the same idea. Not
implemented. `hydro/cons_prolong_pc` is a diagnostic, not a remedy (first-order ghosts).

**GPU (E9g, `results/tables/E9g_dflux_gpu.txt`, H100).** Pressure-exact on the device: 1D contact 4.2e-14, 2D slab
1.7e-14, 3D slab 1.0e-14 (CPU 3.1e-14 / 2.0e-14 / 2.0e-14). GPU vs CPU: contacts <= 2e-13; R4 Riemann L1(rho) 2.4e-6
(discretisation error 5.4e-3); 1D adaptive refinement L1 3.8e-7 with identical cycle counts.

**Non-regression with the key off (fe0caa0f / a5cd4e41).** 20 1D legs bitwise vs a15514d6; E6 waves with the key on
bitwise = standard (the sensor never fires); E7r decks with minmod prolongation reproduced bitwise by a5cd4e41.
Legacy build of the final source (incl. `hydro/cons_prolong_pc`): BIT-IDENTICAL to a6870124 on the production
restart.

Still open at this point: EOS-consistent ghost prolongation/restriction for refined meshes (done in Sec. 6.6); a
production-like 3D run (done in Sec. 6.6); first-order flux correction stays refused. Relevance: the production runs
stay <= 133 K so far (R1 regime, 2e-5 at contacts); the effect matters past ~1500 K.

---

### 6.6 EOS-consistent fill for refined meshes (2026-10-06)

**Final design (binary 85b625bf; `hydro/double_flux_ghost_fill`, default true with double flux on a refined mesh;
`hydro/double_flux_fill_sensor`, default true).** Prolongation, restriction and regridding interpolate rho and E
separately, which mixes rho e linearly -- the Abgrall effect again, in cells the double-flux reset never sees. The
PRESSURE is carried as its own field `dflux_p` ({Cell, Independent, FillGhost, OperatorSplit}, the same refinement
operators as cons). `DoubleFluxRefreshP` sets it to P(cons) in block interiors before every boundary send and after
every ConsToPrim, so Parthenon restricts it (average of the fine P), prolongates it (the limited stencil applied to P)
and carries it through regrids. Where the transported P and P(cons) disagree (> 1e-12) AND the cell passes the
double-flux contact test on the transported pressure, E is rebuilt from the table at the transported P (rho, momenta,
B as interpolated): in ghost regions next to a block on another level (`DoubleFluxGhostFill`, run before ConsToPrim
by `ConsToPrimDoubleFlux`, which replaces Hydro's FillDerived only when the fill is on) and in regridded interiors
(`DoubleFluxPostRegrid`). Two optional hooks were added to the vendored Parthenon (external/parthenon_local, 13
lines, null by default): `Mesh::PreSendBoundBufsHook` (SendBoundBufs, before the send-side restriction) and
`Mesh::PostRegridHook` (after the regrid prolongation/restriction, before the first exchange); AthenaPK installs them
in main.cpp only when the fill is on.

**How it got there (each step measured; `results/tables/E7w_ghostfill.txt`, `E7w2_pfield.txt`, `E7w3_sensor.txt`,
`E7w4_sensor.txt`; max |P/P*-1| over 8 snapshots of the E7r decks).**

| version | 1D smr | 1D amr | 2D smr | 2D amr |
|---|---|---|---|---|
| standard | 6.4e-2 | 5.3e-2 | 1.2e-1 | 8.5e-2 |
| v6 (flux-corrected X, Y, Z only) | 1.35e-3 | 5.5e-4 | 5.1e-3 | 4.0e-3 |
| v7a: prolongated ghosts from the coarse buffer | 4.0e-14 | 4.7e-4 | 4.8e-4 | 4.0e-3 |
| v8: transported pressure, unconditional | 5.8e-14 | 2.2e-13 | 1.6e-13 | 2.5e-14 |
| v8 + contact sensor, post-regrid sensor on ghosts | 5.8e-14 | 4.5e-4 | 1.6e-13 | 2.7e-14 |
| **final: + post-regrid sensor on interior neighbours** | **5.8e-14** | **2.2e-13** | **1.6e-13** | **2.7e-14** |

* v7a read the coarse buffer through `MeshData::PackVariables({"cons"}, /*coarse=*/true)` first. The MeshData pack
  cache is keyed WITHOUT the coarse flag, so that call returns the FINE pack once it exists; a built-in self-check
  (prolongate the coarse density the same way, compare with Parthenon's) showed 45-170 % deviations, then exactly 0
  after reading `Variable::coarse_s` per block. v7a fixed only prolongated ghosts.
* v8 covers restricted ghosts (fine-P average) and regrids. Unconditional, it was exact on the tests but in a
  production restart (below) it rewrote non-contact ghosts at level jumps: e_int changed by up to 44 % in 3 cycles, 89 % of
  those cells on block faces -- interpolating P instead of E is a different ghost scheme in steep fronts, not a
  contact fix. Hence the contact sensor.
* The post-regrid hook runs before the post-regrid exchange, so its sensor must ignore (not yet valid) ghosts.

**Checks of the final version.** Fill off: the four E7r runs reproduced bitwise. Key off (standard): reproduced
bitwise. Acoustic wave S3 through a static patch (E7x decks, N = 32/64/128): BITWISE identical to standard (the
sensor never fires in sound waves; the unconditional v8 differed by 1e-7..6e-9 of the amplitude). Production restart (E7z: 3D collapse at 1.1 t0, 64 ranks, the full production configuration incl. self-gravity and
10-level Jeans AMR, 3 cycles): double flux + fill vs key off, e_int max 1.9e-5,
99.99 % 8.4e-6, rho max 2.0e-10, 0 cells above 1e-3 -- inside the noise floor of the same binary on 48 vs 64 ranks
(e_int max 2.8e-5). The production state at 1.1 t0 (<= 133 K) has no contact where the method changes anything,
as R1 predicted. Wall time 623 s vs 589 s (+6 %).
Key off: 20 1D legs bitwise vs a15514d6 (`runs/EF56_regate_v8`). Legacy build of this source incl. the
Parthenon hooks: BIT-IDENTICAL to a6870124 on the production restart. GPU (E9i, H100, GPU 26b44e57 built from the identical source; `results/tables/E9i_gf_gpu.txt`): all four
refined-mesh contacts pressure-exact on the device (max |P/P*-1| 2.2e-13 / 9.9e-14 / 1.6e-13 / 2.4e-14 for 1D amr / 1D
smr / 2D smr / 2D amr), same cycle counts as the CPU, GPU-CPU max rel diff <= 4.1e-12 (rho) and <= 1.1e-13 (P).

---

## 7. Group B3 — homologous compression along the adiabat (E8)

Uniform gas compressed by the planar homologous flow v = -alpha (x - xc) (`eos_test compress`), 33 cells (the
centre cell at rest), ghosts = exact homologous continuation (`eos_homologous`). Five legs of 100x compression
start on the table adiabat through (1e-13 g/cc, 10 K): 1e-13 -> 1e-11 (10 -> 181 K), 1e-11 -> 1e-9 (-> 1309 K),
1e-9 -> 1e-7 (-> 2515 K, into dissociation), 1e-7 -> 1e-5 (-> 3888 K), 1e-5 -> 1e-3 (-> 12 476 K, into H
ionization). Reference: the exact adiabat de/dln rho = P/rho of the same interpolant from the leg's actual initial
state. A total-energy scheme assigns the cell-averaged kinetic energy of the sub-cell velocity gradient to the
internal energy at (alpha dx C)^2/24 (C = compression): alpha was chosen to keep that at ~1e-6 of e, and a leg
at alpha/2 measures it.

| leg | e/e_ad - 1 at 100x: alpha / alpha/2 / CFL 0.15 / 60 bisections | Richardson alpha -> 0 | box uniformity |
|---|---|---|---|
| 0 (10 -> 181 K) | +1.26e-6 / +3.4e-7 / +1.9e-6 / +1.23e-6 | +3e-8 | 3e-9 |
| 1 (-> 1309 K) | +2.58e-6 / +4.8e-7 / +4.0e-6 / +2.56e-6 | -2e-7 | 2e-8 |
| 2 (-> 2515 K) | +5.42e-6 / +1.26e-6 / +7.3e-6 / +5.65e-6 | -1.3e-7 | 6e-8 |
| 3 (-> 3888 K) | +1.04e-5 / +2.35e-6 / +1.36e-5 / +1.08e-5 | -3e-7 | 4e-7 |
| 4 (-> 12 476 K) | +1.07e-5 / +2.94e-6 / +1.54e-5 / +1.13e-5 | +3.5e-7 | 1.4e-6 |

The residual scales as alpha^2 (ratio 3.6-4.6 for alpha/2), i.e. it is the geometric kinetic-energy term; with it
removed the scheme follows the adiabat to <= 4e-7 through dissociation and into ionization. A smaller time step
does not reduce it (CFL 0.15 is 1.3-1.5x larger: more steps, more of the same sub-cell term), and 60 bisections
change nothing. The `p` interpolant has its own adiabat, 0.24 / 0.20 / 0.02 / 0.03 / 0.23 % hotter in T per leg
than `p_over_rho` (E-F1 in compressional heating); the scheme follows it equally well.

![E8](results/figures/E8_adiabat.png)

Adiabat landmarks (`results/tables/E8_adiabat_path.txt`, figure `results/figures/E8_adiabat.png`): T = 100 K at
3.3e-12 g/cc, 2000 K at 7.7e-9 g/cc; Gamma1 = 1.666 below 1e-12, 1.43 plateau (1e-10), minimum 1.118 at
3.8e-8 g/cc (H2 dissociation), recovery above 1e-4 and the H-ionization dip near 1e-2. Literature
(Masunaga & Inutsuka 2000, ApJ 531, 350): the regimes and their order are those of their central evolution
(adiabatic first core, second collapse once H2 dissociates near 2000 K); a QUANTITATIVE overlay needs their
central T(rho) data, which are not in the abstract and were not available offline — not done. Their path is not
adiabatic (radiative losses), so only the regimes are comparable anyway.

---

## 8. Group C — GPU (E9)

One H100 (GPU binaries 56149b55 / 4d8a6115) vs one CPU rank (890d7f56 / a6870124), same source and decks
(`results/tables/E9_gpu_cpu.txt`, `E9_riemann_L1.txt`, `E9_decomposition.txt`). The t = 0 dump compares the
LOOKUPS (IC energy through EintFromRhoTk on the device, P = Pof in the first ConsToPrim).

| test | t = 0 (lookups) | end of run |
|---|---|---|
| E6 waves, 5 states, N = 128 (427 cycles) | max rel 2.3e-15 in P, rho bitwise | rho, P within 1.2e-14 .. 8.1e-14 |
| E7e Riemann, 4 problems x 2 modes, N = 400 as 4 x 100 blocks (E-F8) | P <= 3.3e-15, 4 of 8 bitwise | same cycle count and final time; L1(GPU - CPU)/max 3e-10 .. 6.5e-6, 3-4 orders below the L1 error vs exact (1.3e-3 .. 6.0e-3), which agrees to all 4 printed digits |
| E8 compression legs 2 and 3 (12 427 / 7970 cycles) | bitwise | rho, P within 5e-10 |
| E10 floor probe, 6 legs | P <= 2e-16 | P <= 1.5e-15; the same flooring / clamp behaviour |

The pointwise maxima at discontinuities (up to 3e-4 in E7e) are a round-off-level shift of a jump within a cell,
hence the L1 metric. CPU decomposition / binary checks: 4 x 100 vs 1 x 400 blocks and binary a6870124 vs 890d7f56
are BITWISE identical on all 8 E7e N = 400 legs. **GPU path: PASS.**

**With the E-F5/E-F6 fix (E9f, GPU a967dca0 = source of CPU a15514d6, `runs/E9_gpu_final`,
`results/tables/E9_gpu_fix.txt`).** The extrapolating lookups run through out-of-line helpers on CUDA, so they were
re-checked: the six floor probes (incl. negative-energy and sub-edge cells, i.e. the new below-table paths) agree
with the CPU to <= 1.3e-15; E8_leg2 to 5.2e-10 after 12 427 cycles; R4 N = 400 L1(rho) difference 5.1e-6 against a
discretisation error of 5.4e-3. Same cycle counts everywhere. Double flux and Tfloor were NOT run on the GPU.

---

## 9. Group D — production footprint and floor probe (E10)

### 9.1 What production occupies
Restarts of a production collapse run at two epochs (1520 blocks, levels 0-7, 49.8 M cells): rho = 1.8e-20 ...
1.5e-11 g/cc, T = 0.67 ... 133 K (0.05-dex bins; 99.9 % of cells per density bin between 9.4 K and 133 K). Inside
the table everywhere — but the AMBIENT sits only 0.25 dex above the table's lower density edge (1e-20 g/cc). No cell
has e below the table (0 of 49.8 M), none has e <= 0.

### 9.2 Clamp counters in the logs
Across three production runs (726 to 1872 cycles each) every cell-state clamp was `esp<` = a lookup with
0 < e < e_min (T < 0.063 K): transient stage/ghost states (none survive into the restarts); one run also walked off
the LOW-DENSITY edge 29 times (E-F6).

### 9.3 Floor-visibility probe (`runs/E10`, `results/tables/E10_probe.txt`)
Thin 3D box (32 x 4 x 4; cons_diag needs 3D, E-F4), 1e-17 g/cc, left half 10 K, right half planted with
e = 0.5 e_min (sub-edge) or e = -1e-3 code (negative), production pfloor/dfloor, `cons_diag = true`,
`eos_floor_consistent` true/false, 3 cycles.

| leg | EOS clamp lines | cons-npfloor | right-half P/rho | total energy |
|---|---|---|---|---|
| control | 0 | 0 | 0.981 (10 K) | 26.9144 |
| sub-edge, fc true / false | 3 (2304, 2176, 2176 `esp<` lookups per cycle) | 0 / 0 | 6.2098e-3 = P(e_min)/rho | 13.4998 = unchanged (e kept at 0.5 e_min) |
| negative, fc true / false | 0 | 0 / 0 | 6.2099e-3 | 13.5424 = 13.4572 + 0.0852 injected (e raised to e_min) |

A cell with e <= 0 is floored to the table edge, energy is injected, and NOTHING records it — not the clamp
counter (the lookup takes the gm1 branch, the bisection is uncounted), not cons-npfloor in either
`eos_floor_consistent` setting (after the flooring pass the cell's P is P(e_min) >> pfloor). A sub-edge positive
e is counted but not floored, and its pressure is that of e_min, not of its energy (here x2). -> E-F5.

With the fix (default build a15514d6, `runs/EF56_final`, `results/tables/E10_probe_fix_final.txt`):

| leg | cons-npfloor (cycles 0-3) | right-half P/rho | total energy |
|---|---|---|---|
| sub-edge | 0 (not floored: P = 0.5 P(e_min) > pfloor) | 3.1049e-3 = 0.5 x 6.2098e-3 | 13.4998 (unchanged) |
| negative | 256, 224, 208, 208 | 5.467e-10 = pfloor/rho | 13.4572 (no visible injection) |

With `hydro/eos_extrapolate = false` all six legs reproduce the old E10 runs bitwise.

### 9.4 Physical closure of E-F5 and E-F6

**Where is the continuation physically right? (`scripts/ef56_validity.py`, `results/tables/EF56_validity.txt`).**
Below the energy edge (T < 0.063 K) the table itself is an ideal gas: P/(rho e) on the bottom e row is 0.666675 at
all 400 density nodes (2/3 = 0.666667), so the gamma = 5/3 continuation of E-F5 is exact physics, not a guess.
Below the density edge the continuation (P = rho (P/rho)_edge at fixed e) is compared with a direct Saha
calculation at 1e-21, 1e-22, 1e-24 g/cc: <= 3e-5 in P and T for T <= 300 K, but 4-20 % off in the H2-dissociation
(~1000 K) and H-ionization (3000-5000 K) bands, where composition depends on density. So the continuation is
realistic for cold gas only.

**Extended table (shipped as `src/eos/eos_table_hires_v5_cold_rho26.bin`).** `src/eos/gen_eos_table.py` gains `rho_pad_pts`
(argv[11]): prepend rows below the original density edge at the same d log rho, keeping the e and T axes from the
original rows. `runs/EF6_table/v5cold_rho26.bin` (md5 cb68bd0f): 520 rows down to 1e-26 g/cc; identical e/T axes;
the original rows' P, log T and e(T) BIT-IDENTICAL; cs2 differs only on the former edge row (row 0), by up to 6.1 %,
because its one-sided derivative becomes central (the old edge value was the less accurate one). New rows vs direct
Saha at off-node densities: max 2.0e-2, typically 1e-3 (the v5_cold table's continuation at the same points:
up to 3.2e-1; `results/tables/EF6_table_check.txt`). E2 thermodynamic gates pass on the extended table
(`E2_consistency_rho26.txt`). Regenerating with `rho_pad_pts = 0` is byte-identical to
`eos_table_hires_v5_cold.bin` (f712d6a8). Switching tables changes results only in cells near 1e-20 g/cc (cs2 of
row 0) and below.

**Physical floor (`hydro/Tfloor`).** The extrapolating EOS makes `pfloor = 1e-8` bind, which drives a floored cell
to T ~ 0. A temperature floor is the physical choice. `hydro/Tfloor` previously required the ideal-gas `mbar_over_kb`
and so could not be used with the tabulated EOS; it now works with `eos = hydrogen` (code temperature units,
1 = 10.015 K) and floored cells are counted in a new `cons-nTfloor` history column. Probe with `Tfloor = 0.2726`
(2.73 K, CMB; binary 04db22ca, `runs/EF5_tfloor`): negative-energy and sub-edge legs end at P/rho = 0.26782 (code),
i.e. T = 2.729 K with mu = 2.33; `cons-nTfloor` 256, 224, 192, 192 as the left gas sweeps over the probe;
`cons-npfloor` 0 (pfloor never reached); control leg 0 floored cells. `hydro/Tfloor` is set per deck.

---

## 10. Findings

**E-F1 — sound-speed sawtooth from interpolating P on a log-rho axis (FIXED; `hydro/eos_interp = p_over_rho` is the
default).** Interpolating P bilinearly in (log rho, log e) is linear in P on a 0.05-dex log-rho axis; interpolating
P/rho_node and multiplying by rho removes it. E6 + E8 quantify it (phase speed up to 1.4 %, adiabat 0.2 % in T per
100x compression, rarefaction convexity G_min 0.6 instead of 1.0).

**E-F2 — the shipped consistency test could not read v2 tables** (fixed in the suite's copy only).

**E-F3 — `glmmhd_source = dedner_extended` segfaults in 1D** (indexes y/z neighbours). Validation-only.

**E-F4 — `hydro/cons_diag = true` segfaults in 1D** (VERIFIED, gdb: `ConsDiagReduce` evaluates the X2/X3
boundary-face fluxes, which a 1D mesh does not allocate; null dereference at the first history write).
Production is 3D; validation-only. Workaround: thin 3D boxes.

**E-F5 — the pressure floor was inert under the tabulated EOS and floored cells were invisible (VERIFIED; FIXED
2026-10-05, key `hydro/eos_extrapolate`, default true).** The lowest pressure the table can return, P(rho, e_min) =
6.21e-3 rho (code; T = 0.063 K), exceeds the production `pfloor = 1e-8` by 4 to 19 decades over the table's density
range, so (a) a cell with 0 < e < e_min was never floored -- it was counted (`esp<`) but got P(e_min), more
pressure than its energy holds; (b) a cell with e <= 0 was floored to e_min (InternalEnergyFromPres clamps silently
at the table edge) with energy injected, and no diagnostic saw it: the clamp counter was not touched and
`cons-npfloor` (P <= pfloor) stayed 0 with `eos_floor_consistent` on AND off. FIX: below the e axis the gas is
continued as the ideal gas it is (cold H2, rotation frozen, gamma = 5/3): P, cs2 and T scale with e/e_min at fixed rho,
EspFromP inverts that exactly, T below the T axis gives e ~ T. Verified (`host/e5_inversion.cpp`,
`results/tables/E5_inversion_final.txt`; probe `runs/EF56_final`, `results/tables/E10_probe_fix_final.txt`, binary
a15514d6): the floor now binds at P = pfloor exactly, a negative-energy cell is floored to P = pfloor with negligible
energy injection (tot-E 13.4572 vs 13.5424 before), and `cons-npfloor` counts it (256 -> 224 -> 208 cells as the
left gas sweeps over the probe); a sub-edge cell carries the pressure of its own energy (0.5 P(e_min) for
e = 0.5 e_min); every such lookup is counted as `esp<` (EspFromP's below-table exit is now counted too). In-table
lookups are bitwise the legacy ones (0 of 1 431 582 differ; 14 in-table 1D legs bitwise). An earlier ConsToPrim
floor-event counter was dropped (see E-F9). The production pfloor = 1e-8 now means T -> 0 in a floored cell; the
physical closure is `hydro/Tfloor` (now usable with the tabulated EOS and counted, Sec. 9.4); setting it is a deck
decision. Physical validity of the gamma = 5/3 continuation: the table's bottom row IS ideal (P/(rho e) = 0.666675).

**E-F6 — with `eos_interp = p` the pressure was frozen below the table's lower density edge (VERIFIED; FIXED by the
same key).** At rho < 1e-20 g/cc the legacy mode returned P(rho_edge, e): x2 too high at 0.5 rho_edge, x10 at 0.1.
With the key on, `p` continues as P = rho (P/rho)_edge, exactly what `p_over_rho` already does (verified: ratio
1.000000 at 0.5-1e-4 rho_edge). The production ambient sits 0.25 dex above the edge; one production run walked off
it 29 times. The continuation is physically right only below ~300 K (<= 3e-5 vs Saha; 4-20 %
off in the dissociation/ionization bands); the realistic closure is the extended table to 1e-26 g/cc (Sec. 9.4,
<= 2 % vs Saha, original rows bit-identical except row-0 cs2), shipped as `eos_table_hires_v5_cold_rho26.bin`.

**E-F7 — T -> e -> T is not an identity (VERIFIED, characterisation).** e(rho, T) and T(rho, e) are two
independently bilinear tables; their composition errs by up to 1.3 % in T across H2 dissociation and H
ionization (2e-5 in the production footprint). Every code path that goes T -> e -> T inherits it (e.g. an IC set
from T). It bit this suite: the first E7 decks passed T and the pgen re-derived e, which shifted the initial
pressures by up to 4.6e-4 and was exactly the plateau error those runs stalled at; any T-based coupling above
1000 K is where it would matter.

**E-F8 — 1D meshblocks of 400 cells abort on the GPU** (`Kokkos::Impl::ParallelFor<Cuda, TeamPolicy> could not
find a valid execution configuration`: per-team scratch for a 400-cell pencil exceeds the CUDA limit; 128 runs).
Production meshblocks are 32^3; affects only long-pencil test decks.

**E-F9 — a runtime key cannot keep an existing run bit-continuous when the inline lookups change (VERIFIED,
2026-10-05).** Gates on a 3D production restart (64 ranks, full production configuration, 3 cycles): every version
of the fix with the key OFF differed from the pre-fix binary a6870124 (cons max rel 1.1e-4), although the 1D tests and
the host harness were bitwise. Bisection: the same source with only the new MEMBERS is BIT-IDENTICAL; the changed
inline bodies alter GCC's inlining decisions (`nm`: `Pof` went out of line in the ConsToPrim TU, ...) and with them
the FMA contraction in the callers. Five restructurings (verbatim legacy branch, select style, out-of-line helpers,
private bilin copy, forced inlining) moved the fingerprint but never removed it. Scale: the SAME pre-fix binary on 48
instead of 64 ranks differs from itself by cons max rel 1.08e-4 after 3 cycles -- the key-off differences are
round-off reordering amplified by the stiff coupled solvers of the production configuration, not physics.
Resolution: a compile-time switch. CMake `-DATHENAPK_EOS_LEGACY_LOOKUPS=ON` compiles the original lookups verbatim
(key refused); that build is BIT-IDENTICAL to a6870124 with the default keys and with the continuity keys
(`eos_interp = p`, `vl2_predictor_recon = dc`). Use it to continue an existing run bit-continuously; the default
build carries the fix.

**E-F10 — a conservative scheme cannot keep a moving contact between EOS regimes in pressure equilibrium; this,
not the table, is the slow R2/R4 convergence (VERIFIED; FIXED opt-in with PLM, `hydro/double_flux`, default off).**
Sec. 6.3-6.4: not the table (four falsifications), not the integrator or time step; the star-region entropy and
Riemann invariant are right to ~1e-6; a pure moving contact carries up to 3-4 % spurious P (R4), converging as
N^-0.7, exact at rest (Abgrall 1996). Double flux (Abgrall & Karni 2001) with a per-cell contact sensor makes moving
contacts pressure-exact (1e-14) and brings the R2/R4 plateaus to ~1e-6 (65-280x below PLM at N = 3200-6400) with no
change in acoustic waves (bitwise) or in R1/R3; price: +1e-4 .. +4e-4 of the total energy at contacts between
regimes, converging at order ~0.6. LimO3 reduces the error 3-9x without fixing the mechanism. Double flux is
validated in 1D/2D/3D, on CPU and GPU and, with the sensor-gated EOS-consistent fill, on statically and adaptively
refined meshes (Sec. 6.5-6.6).

**E-F11 — every step deep-copied the table's host mirrors (performance; FIXED 2026-10-14, results bit-identical).**
Found during the port check of the dust module (`../dust/DUST_VALIDATION.md` Sec. 2.3): this repository's builds ran
tabulated-EOS problems 10-23x slower per step than the research code although every result was bitwise identical.
Stack samples (gdb) put half of the time in the `EOSTable::EosTable` / `AdiabaticGLMMHDEOS` copy constructors: every
`KOKKOS_LAMBDA` that captures the EOS object copies it by value on the host at each launch, and the table's two host
mirrors (`espT_h_`, `logT_h_`, 5.8 MB each for the rho26 table, read only by the setup-time host lookups) were
`std::vector` members, i.e. deep-copied each time. The research code had made them `std::shared_ptr` on 2026-10-09;
that change had not reached this repository. Now ported (`src/eos/eos_table.hpp`): one CPU core, `-ffp-contract=off`,
the dust module's D3a one-zone compression 29.4 s -> 1.34 s (research code 1.27 s), the ideal-MHD module's I1c acoustic
wave (N 128) 13.6 s -> 1.02 s (research code 1.09 s). Check: results bit-identical -- the 47 regression runs vs the dust commit (as built), and with `-ffp-contract=off` the 422 ideal-MHD and 118 + 4 dust runs of those modules' port checks vs the research code; the host mirrors are read only at setup, so no kernel changes (`results/tables/PORT_check_EF11_2026-10-14.txt`).

## 11. Defaults in this code

| item | status |
|---|---|
| E-F1 | `hydro/eos_interp = p_over_rho` is the default; `p` reproduces runs made before the fix |
| E-F5 / E-F6 | `hydro/eos_extrapolate`, default true. CMake `-DATHENAPK_EOS_LEGACY_LOOKUPS=ON` compiles the legacy lookups verbatim (key refused) to continue a pre-fix run bit-continuously |
| E-F5 floor | with the fix `pfloor` binds and drives a floored cell's T to ~0; `hydro/Tfloor` (e.g. 0.2726 = 2.73 K) works with the tabulated EOS and is counted (`cons-nTfloor`) |
| E-F6 table | `eos_table_hires_v5_cold_rho26.bin` (to 1e-26 g/cc, <= 2 % vs Saha, original rows bit-identical except row-0 cs2) is shipped and recommended |
| E-F10 contacts | `hydro/double_flux = true` (opt-in, default off) makes contacts pressure-exact with PLM in 1D/2D/3D, on CPU and GPU (Sec. 6.4-6.5); energy +1e-4..4e-4 (Riemann) to 1e-2 (slab, N = 128) at regime-changing contacts, converging. Refined meshes: pressure-exact with the sensor-gated EOS-consistent fill (`hydro/double_flux_ghost_fill`, default on with double flux; Sec. 6.6; two null-default hooks in the vendored Parthenon). Production restart: effect within the noise floor at <= 133 K, +6 % wall time. Alternative: LimO3. Matters past ~1500 K |
| E-F7 round trip | characterised; relevant to any T-based coupling above 1000 K |
| constant mu outside the EOS | the EOS has a variable mu(rho, T) (2.33 molecular -> 1.27 atomic -> 0.6 ionized); any other module that assumes mu = 2.33 is valid only below ~2000 K |
| `collapse_be` with `eos = hydrogen` | added after this report (2026-10-07): the barotropic step prescribes T(rho) = T0 sqrt(1 + (rho/rhocrit)^(2(gamma-1))) and sets e = e_table(rho, T). Checked on two collapse_be decks straddling rhocrit (rho26 table, gravity and AMR off, float64 output): e = e_table(rho, T_law) to <= 3e-13 in all 2 x 262 144 cells; T read back from the table matches T_law to 1.6e-5 (the E-F7 round trip). Ideal-gas collapse_be runs bitwise unchanged |
| table coverage | complete for the production footprint (rho 1.8e-20-1.5e-11, T 0.7-133 K) and the design range up to 3e5 K |

## 12. Test-infrastructure traps met in this module (not code findings)
* numpy 2 writes `np.float64(x)` for repr(); Parthenon's `GetReal` uses `atof`, which returns 0.0 for it
  SILENTLY (an E8 tlim of 0, an E10 energy of 0). Cast to float before writing decks.
* An environment that sets `OMP_PROC_BIND=spread` + `OMP_PLACES=threads` breaks concurrent runs: many independent 1-rank `mpirun`s in one
  job then pin every process to the job's first core (0.6 s/cycle instead of 0.01). Set `OMP_PROC_BIND=false`.
  Even then 48 concurrent 1-rank runs cost ~0.08 s/cycle vs 0.01 alone (not investigated).
* sbatch snapshots the script at submission: edits after `sbatch` do not reach a queued job.
* A homologous ghost BC that extrapolates each side from its own edge cell amplifies a uniform translation
  (ratio > 1) — the first E8 set stalled with dt -> 1e-72; the symmetric version is stable.
* `self_gravity/scripts/run_one.sh` uses `mpirun --bind-to core`: concurrent independent 1-rank runs all land on
  core 0 (use `scripts/run_one_nobind.sh`).
* sbatch passes the submitting shell's TMPDIR; if that directory is deleted before the job runs, the build/run
  fails ("can't open .../ccXXXX.s"). Submit with `--export=ALL,TMPDIR=/tmp`.
* Bit-identity gates must compare against a reference built from the SAME tree minus the change (a6870124 here);
  older references differ for unrelated reasons.

## 13. Reproduction map

| test | decks (`inputs/`) | run script (`runs/`) | analysis (`scripts/`) | tables (`results/tables/`) | figure |
|---|---|---|---|---|---|
| E1 | — | E1_rebuild.sh | — | — | — |
| E2/E3 | — | — | test_eos_consistency.py | E2_consistency_v5cold.txt | — |
| E5 | — | host/run_e5.sh | (the harness) | E5_inversion.txt | — |
| E6 | E6_wave_* (make_decks_eos.py) | E6_waves.sh, E6b_prho.sh | analyze_E6.py | E6_waves*.txt/json, E6_fix_P_over_rho.txt | — |
| E7 | E7e_* (make_decks_eos2.py e7) | E7e.sh; E7f_bisect.sh | e7_exact.py, e7_hugoniot_check.py, analyze_E7.py, e7f_eval.py | E7_exact.json, E7_hugoniot_check.txt, E7e_riemann.txt/json, E7f_bisect.txt (E7_riemann_v1_Tdeck.* = superseded first set) | E7e_riemann.png |
| E8 | E8_leg* (make_decks_eos2.py e8), E8_legs.json | E8_E10.sh | analyze_E8.py | E8_compress.txt/json, E8_adiabat_path.txt | E8_adiabat.png |
| E9 | as E6/E7e/E8/E10 | E9_gpu.sh, E9_gpu2.sh, E9_gpu3.sh, E9_cpu_mb100.sh, E6cpu_ref.sh | analyze_E9.py, e9_riemann_l1.py | E9_gpu_cpu.txt, E9_riemann_L1.txt, E9_decomposition.txt | — |
| E10 | E10_probe_* (make_decks_eos2.py e10) | E8_E10.sh | analyze_E10.py | E10_probe.txt | — |
| E-F5/E-F6 fix | E10_probe_*, E7e_*_n400, E6, E8_leg2 | EF56_final.sh | host/run_e5.sh, analyze_E10.py | E5_inversion_final.txt, E10_probe_fix_final.txt | — |
| E-F5/E-F6 closure | E10_probe_*_fctrue (+ `hydro/Tfloor`) | EF5_tfloor.sh, EF6_table.sh, EF56_regate.sh | ef56_validity.py, ef6_table_check.py, analyze_E10.py (`EF5_tfloor "E10_probe_*" E10_probe_Tfloor.txt`) | EF56_validity.txt, EF6_table_check.txt, E2_consistency_rho26.txt, E10_probe_Tfloor.txt | — |
| E7 EOS-consistent fill (refined meshes) | E7r_*, E7x_wave_S3_smr_n* | E7v_prolong.sh, E7w_ghostfill.sh, E7w2_pfield.sh, E7w3_sensor.sh, E7w4_sensor.sh, E7x_wave.sh, E9i_gf_gpu.sh | e7r_eval.py, e9i_eval.py | E7v_pc.txt, E7w_ghostfill.txt, E7w2_pfield.txt, E7w3_sensor.txt, E7w4_sensor.txt, E7x_wave_smr.txt, E9i_gf_gpu.txt | — |
| E7 double flux 2D/3D/AMR/GPU | E7q_*, E7r_*, E7k/E7e (E7t) | E7q_slab.sh, E7r_amr.sh, E7t_v6.sh, E7u_bitid37.sh, E7v_prolong.sh, E9g_dflux_gpu.sh + E9g_cpu.sh | e7q_eval.py, e7r_eval.py, e7m_eval.py (`E7t E7t_dflux_v6.txt`) | E7q_slab.txt, E7r_amr.txt, E7v_pc.txt, E7t_dflux_v6.txt, E9g_dflux_gpu.txt | — |
| E7 contact + double flux | E7k_*_{mov,rest}_n*, E7e_*_p_over_rho_n*, E6_wave_* | E7k_contact.sh, E7m_dflux.sh, E7n_dflux.sh, E7o_dflux.sh, E7p_dflux.sh, E6_dflux.sh | e7j_entropy.py, e7j_invariant.py, e7k_eval.py, e7m_eval.py (`E7p E7p_dflux_cell.txt`) | E7j_*.txt, E7k_contact.txt, E7m/E7n/E7o/E7p_*.txt | — |
| E7 diagnosis | E7c_* (archived), E7d_*, E7g_*, E7e (E7h/E7i overrides) | E7d.sh, E7g.sh, E7g_table_*.sh, E7h_scheme.sh, E7i_limo3.sh | e7g_kinks2.py, e7g_profiles.py, e7g_eval.py, e7f_eval.py, analyze_E7.py (E7_PREFIX) | E7g_kinks2.txt, E7g_tables.txt, E7d_riemann.txt, E7h_scheme.txt, E7i_limo3.txt | E7g_error_profiles.png |
| W (worked examples, Part A Sec. A11) | — | host/run_worked.sh | host/worked_examples.cpp (real eos_table.hpp), worked_saha.py | W_worked_examples.txt, W_worked_saha.txt | — |

## 14. References
Abgrall 1996, JCP 125, 150 (pressure oscillations of conservative schemes at material interfaces); Abgrall & Karni
2001, JCP 169, 594 (double flux); Billet & Abgrall 2003, Computers & Fluids 32, 1473; Cada & Torrilhon 2009, JCP 228,
4118 (LimO3); Colella & Glaz 1985, JCP 59, 264; Liu 1975, JDE 18, 218 (shock admissibility); Masunaga & Inutsuka
2000, ApJ 531, 350; Toro 2009, Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed.).

---

# Part C — Reading the validation, limits, open points, source index

## C1. What each test establishes, and which code it exercises

The results are in Part B (Sec. 4–9); this table says what each one proves and what it does not. "Runtime code"
means the C++ that a simulation executes; tests that re-implement the lookup in Python are marked as such.

| test | property tested | code exercised | reference / metric | establishes | does **not** establish |
|---|---|---|---|---|---|
| E1 provenance | the shipped bytes come from the shipped generator | `gen_eos_table.py build` | byte identity | reproducibility of `v5_cold` (and, in Sec. 9.4, `rho26`: md5 `cb68bd0f`) | that the physics is right |
| E2 thermodynamic gates | Saha solution converged; P and e mutually consistent; table ≈ direct Saha | `solve_saha`, `P_e_of_rho_T`; the table through a **Python** bilinear reader | neutrality / conservation to round-off; Maxwell identity to 3e-3; table vs Saha at off-node points | the generator's equilibrium solve and the table's interpolation accuracy (1e-3 level) | the C++ lookups; agreement with a real-gas EOS. Its Maxwell residual (2.46e-3) is the A3.5 ortho/para inconsistency, not finite-difference error (T-4) |
| E3 limits | $\Gamma_1$ in the known regimes | generator + table | known limits (5/3 cold, dips at dissociation/ionization) | the regimes are where physics puts them | quantitative agreement with an external EOS table |
| E5 inversions | monotonicity, bisection error, round trips, edge and floor behaviour | **real** `eos_table.hpp` compiled on the host | analytic bisection bound; identity of round trips | the exact numerical behaviour of the runtime lookups on CPU (E-F5, E-F6, E-F7) | behaviour inside the solver; GPU (see E9) |
| E6 acoustic waves | the scheme propagates sound at the table's speed | ConsToPrim (`Pof`), HLLD (`EspFromP`, $c_f$ via `Cs2of`), eos_test IC (`EintFromRhoTk`) | linear eigenmode after one period; phase speed vs the interpolant's $c_s$ | the solver sees the interpolant's sound speed (found E-F1, fixed by `p_over_rho`) | nonlinear flow; magnetised waves (`B = 0` in the E6 decks; magnetised table waves are I1c of the ideal-MHD report) |
| E7 Riemann problems | shocks and rarefactions across EOS regimes converge to the right weak solution | as E6, plus floors; double flux (Sec. 6.4–6.6) | exact general-EOS Riemann solution of the same interpolant (`riemann_exact.py`) | convergence for R1–R4; the slow R2/R4 convergence is the conservative-contact effect (E-F10), not the table; double flux makes contacts pressure-exact | MHD Riemann problems with the table (`B = 0` in E7) — magnetised table flows are covered by the dual-energy D2/D4 tests |
| E8 compression | the scheme's $P\,dV$ work follows the table adiabat over 10 K → 12 500 K | ConsToPrim, HLLD, `eos_homologous` BC | exact adiabat $d\varepsilon/d\ln\rho=P/\rho$ of the interpolant | ≤ 4e-7 after removing the known geometric term, through dissociation and into ionization | radiative (non-adiabatic) paths; the physical adiabat vs a real-gas EOS |
| E9 GPU | device lookups and solver = CPU | the same kernels on CUDA | CPU run of the same deck | lookups ≤ 3.3e-15; flows to round-off or 3–4 orders below discretisation error | Tfloor on the GPU (not run, Sec. 8) |
| E10 footprint + probe | the production state is inside the table; floored cells are visible | clamp counters, floors, `cons_diag` | counter values, energy budget | coverage of the measured footprint; E-F5/E-F6 found and fixed | coverage of later collapse stages ($T>133$ K was not reached by the runs analysed) |
| W1–W3 (this guide) | worked examples | real `eos_table.hpp` (host) + generator | hand calculation / generator | the step-by-step arithmetic of A7; the cs² bias of A9.2; O-1 | anything about the solver |

## C2. Limitations, assumptions and untested regimes

1. **Physics model** (A3.1, A3.7): ideal, non-degenerate Saha gas; no pressure ionization, Coulomb corrections, H$^-$,
   H$_2^+$, metals or radiation pressure; harmonic vibration, rigid rotor, ground-state atoms. Quantitatively
   questionable above ~$10^{-2}$ g cm$^{-3}$ in the ionization band (A3.7). Not tested against an external EOS.
2. **Thermodynamic consistency** (A3.5): energy and chemical equilibrium use different ortho/para models, so the table
   is not derived from one free energy where H$_2$ dissociates. T-4 (A3.5): Maxwell residual 2.46e-3, entirely the
   ortho zero-point offset; a consistent table would differ by 0.1–0.2 % in $u$ above ~1000 K. Not changed.
3. **Table top** (A4.3): flat $P(\varepsilon)$ above $3\times10^5$ K at high density; `EspFromP` cannot represent
   pressures above $P(\rho,\varepsilon_{\max})$ (counted as `esp>` since `5649acd`, O-1).
4. **Sound speed**: the tabulated $c_s^2$ is a finite-difference derivative of the tabulated $P$ with a known bias
   (+0.13 % in $\Gamma_1$ for molecular gas, A9.2) and up to 3.3 % deviation at kinks (E2).
5. **Two independent interpolants** for $T(\rho,\varepsilon)$ and $\varepsilon(\rho,T)$: round trip up to 1.3 % (E-F7).
6. **Bisection**: piecewise-constant error up to $8.1\times10^{-6}$ in $\varepsilon$, $9.2\times10^{-5}$ in $P$ (A7.3).
7. **Untested combinations**: `Tfloor` and double flux on the GPU (Sec. 8); MHD shock tubes with the table against an
   exact solution; the table above 133 K in a full collapse run (E10 covered the earlier epochs). FOFC and `Tceil`
   with the table are now exercised by single runs (C3), not by convergence tests.
8. **GPU**: the fixes of C3 were built and run on CPU only; the GPU build of this commit was not compiled.

## C3. Implementation observations from this documentation pass, and their fixes

Found while tracing the code for Part A (commit `a40eff5`), fixed in commit `5649acd`. None of them changed a result in
the configurations validated in Part B. Evidence for each fix: `results/tables/O_fixes_check_2026-10-16.txt` (runs,
NEW = `5649acd` CPU build `1455f3fc`, OLD = `a40eff5` build `446d9034`) and `results/tables/W_worked_examples.txt`
(host harness). **Regression:** the 47 self-gravity / EOS / dual-energy / `collapse_be` regression runs are bitwise
identical to `a40eff5` as built (every field of every dump, history files identical); five dual-energy legs (decks
with `Tfloor` = 2.73 K) bitwise; E10 probes with `Tfloor` = 2.73 K bitwise where the floor does not bind and within
1.8e-16 where it does. The changes to the lookups, `ConsToPrim` and the LLF solver are compiled only without
`ATHENAPK_EOS_LEGACY_LOOKUPS`; the legacy build keeps the previous code text (it compiles; it was not re-run against a
pre-fix restart).

* **O-1 — `EspFromP` silently clamped pressures above the table.** For $P>P(\rho,\varepsilon_{\max})$ the bisection
  returns $\approx\varepsilon_{\max}$ (target $10\,P_{\max}$ at $\rho_{\rm code}=1$: $0.999992\,\varepsilon_{\max}$, whose
  pressure is $1.13\times10^5$ instead of $1.13\times10^6$), and no counter was incremented. *Fix:* counted as `esp>`
  when the bisection never lowered its upper bound and the target exceeds $P(\rho,\varepsilon_{\max})$
  ([code][tab-xhigh]). The value is unchanged — there is no physics above $T_{\max}$ in the table — but the clamp
  report now sees it. *Check:* W3(f): `esp>` 0 → 1; a target $0.999\,P_{\max}$ is not counted.
* **O-2 — first-order flux correction used the ideal-gas energy under the table.** The LLF solver behind
  `hydro/first_order_flux_correct` formed $E=P/(\gamma-1)+\dots$ also with `eos = hydrogen`. *Fix:* the internal energy
  comes from `InternalEnergyFromPres` of the active EOS; the ideal-gas expression is unchanged ([code][llf-e]).
  *Check:* a low-β vacuum-forming flow (cold gas, $u=\pm300$, $B_y=10^4$; four other probes, including a $10^9$
  pressure-ratio blast, never triggered FOFC) where FOFC corrects 32 cells: after 200 cycles the old energy differs
  from the table-consistent one by up to 13 % in the pressure of single cells (0.77 % of the maximum).
* **O-3 — `Tceil` under the table was in kelvin with the ideal-gas conversion** (and needed `<units>` +
  `He_mass_fraction`), while `Tfloor` was a code temperature inverted through the table. *Fix:* under
  `eos = hydrogen` `Tceil` is a code temperature like `Tfloor` and the ceiling energy is $e(\rho,T_{\rm ceil})$ from the
  table at each density ([code][hydro-tceil], [code][c2p-tceil]). *Check:* 5000 K gas at $10^{-12}$ and $10^{-9}$ g
  cm$^{-3}$ with a 2000 K ceiling: 1999.89 and 2000.21 K (old: abort without `<units>`; with a units block and
  `Tceil` = 2000 K, 1726 and 2210 K).
* **O-4 — the temperature unit and the table's molecular weight differ by 1.8 %** (A5.4), so `collapse_be` with the
  table started at $P=0.98244\,\rho$, 1.76 % below the Bonnor–Ebert equilibrium $p=\rho$. *Fix:* the isothermal
  temperature $T_0$ of a table run is, by default, the one at which the table gives $P=\rho$ (10.19 K for the default
  units; [code][be-t0]); `problem/collapse_be/table_isothermal = temperature` keeps $T_0=T_u$. *Check:* L6 deck with
  the table: $P/\rho=1.00000000$ in every cell at $t=0$ (old 0.98244); with `temperature` bitwise equal to the old run
  after 20 cycles. This changes the default behaviour of `collapse_be` with `eos = hydrogen` (a modelling choice, A5.4).
* **O-5 — the temperature floor was converted at $\rho_{\rm code}=1$ only**, exact only where $\varepsilon(T)$ does not
  depend on density. *Fix:* the floor energy is $e(\rho,T_{\rm floor})$ at each cell's density, in `ConsToPrim`, the
  dual-energy fallback, `DualSelectEint` and the `cons-nTfloor` counter ([code][eos-tlim], [code][c2p-tfl]).
  *Check:* 100 K gas at $10^{-12}$ and $10^{-9}$ g cm$^{-3}$ with a 2000 K floor: 1999.89 and 2000.21 K (old 2175.73 and
  2779.46 K); W3(g) over four densities. At 2.73 K the two agree to the last bit (see the regression above).
* **O-6 — stale or misleading comments.** *Fixed:* the generator's docstring (the runtime reads the `.bin`, and how
  $c_s^2$ is formed) and the description of its default `build` target; `src/eos/tests/test_eos_consistency.py` now
  reads v2 tables and defaults to the shipped rho26 table (E-F2; all gates pass with the E2 numbers); the per-cycle
  clamp report no longer calls continued lookups "edge-clamped" and says that `esp>` includes inversions above the
  table; the `collapse_be` comments. The generator's code is unchanged (comments only), so the tables are unchanged.
* **O-7 — top-node lookups were counted as clamps.** *Fix:* the upper count fires only beyond the top node
  ([code][tab-bilin]). *Check:* W3(f): 0 counts for a lookup at $\varepsilon_{\max}$.

Not changed: the ortho/para modelling choice of A3.5 (a physics decision of the generator, quantified by T-4: its
inconsistency is the whole E2 Maxwell residual; the minimal consistent variant would change the table by 0.1–0.2 %
above ~1000 K) and the other limitations of C2.

## C4. Suggested additional tests

T-1, T-2, T-3 and T-6 were run for the fixes of C3, T-4 for A3.5 (status column); the others are proposals that have not
been run.

| id | test | what it would establish | status |
|---|---|---|---|
| T-1 | host-harness case: `EspFromP` above $P(\rho,\varepsilon_{\max})$ with a counter assertion | makes O-1 observable and regression-tested | done (W3(f)) |
| T-2 | E7 R2/R4 with `first_order_flux_correct = true` forced to trigger (e.g. very low `dfloor`/`pfloor` margins), table vs `eos_floor_consistent` energy audit | whether O-2 changes energies in practice | done with a low-β probe instead of R2/R4 (FOFC never fired in the Riemann/blast probes; C3) |
| T-3 | E10-style probe with `hydro/Tceil` under `eos = hydrogen` | the temperature the ceiling actually enforces (O-3) | done (C3) |
| T-4 | rebuild the table with the rotational energy taken from the same (equilibrium) partition function as $K_d$ and repeat the E2 Maxwell gate in the dissociation band | isolates the A3.5 contribution to the 2.46e-3 Maxwell residual | done without a table rebuild (gate 3 evaluates the generator directly): the whole residual, A3.5 |
| T-5 | rebuild with half the $\log\rho$ spacing; measure $\Gamma_1$ at 10 K | confirms the A9.2 prediction (bias ∝ $\Delta^2$: 0.133 % → 0.033 %) | not run |
| T-6 | `collapse_be` at $t=0$ with `eos = hydrogen` vs `adiabatic`: early velocity growth and central density drift before collapse | quantifies O-4 | done for $t=0$ and 20 cycles (C3); collapse dynamics not compared |
| T-7 | compare $P$, $\varepsilon$, $\Gamma_1$ with an independent EOS that includes non-ideal effects (e.g. Saumon, Chabrier & van Horn 1995) above $10^{-6}$ g cm$^{-3}$ | where the ideal-Saha model departs from a dense-plasma EOS (A3.7) | not run |
| T-8 | GPU runs of the Tfloor probe and of double flux on refined meshes with Tfloor | closes the GPU gaps listed in Sec. 8 (and C2 item 8) | not run |
| T-9 | radiation Fleck derivative $d\varepsilon/dT$ from the table vs analytic $c_v$ of the generator | accuracy of the numerical $c_v$ (A3.6) | not run |
| T-10 | magnetised general-EOS Riemann problem (e.g. Brio–Wu states mapped to the R2 densities/temperatures) against a converged reference | HLLD + table with $B\neq0$ in shocks (C2 item 7) | not run |

## C5. Source reference index

All links point to commit `5649acd` (the fixes of C3). Line ranges are those of that commit.

| file | functions / blocks | role |
|---|---|---|
| [`src/eos/gen_eos_table.py`][gen] | constants [L28–39][gen-const], `Zrot_ns` [L62–70][gen-zrot], `erot_per_H2` [L71–81][gen-erot], `Zvib`/`evib_per_H2` [L84–93][gen-evib], `solve_saha` [L95–110][gen-saha], `P_e_of_rho_T` [L140–149][gen-pe], `build_table` [L151–383][gen-build] | offline physics and table construction (A3, A4) |
| [`src/eos/eos_table_format.hpp`][fmt] | `ReadEosHeader` | file format v1/v2 (A4.4) |
| [`src/eos/eos_table.hpp`][tab] | `bilin` [L122–173][tab-bilin], legacy lookups [L175–225][tab-legacy], continuation helpers [L269–327][tab-x], `Pof`/`Cs2of`/`LogTof` [L329–346][tab-pof], `EspFromP` [L347–382][tab-espfromp], `XCountEspHighP` [L310–319][tab-xhigh], interface [L385–399][tab-iface], `TemperatureKFromPres` [L423–434][tab-tfromp], `EintFromRhoTk` [L435–447][tab-eintT], host twins [L454–540][tab-host], `Load` [L543–633][tab-load], counters [L635–655][tab-ctr2] | runtime lookups (A7) |
| [`src/eos/adiabatic_glmmhd.hpp`][eos-class] | constructor, `SetTableTemperatureLimits`/`EintFloor`/`EintCeil` [L69–90][eos-tlim], `InternalEnergyFromPres` [L56–63][eos-iefp], `SoundSpeed`/`FastMagnetosonicSpeed` [L92–114][eos-cf], `ConsToPrim` [L117–376][c2p] | the EOS interface of the solver (A8) |
| [`src/eos/adiabatic_glmmhd.cpp`][c2p-mesh] | `ConservedToPrimitive` | the per-cell kernel launch |
| [`src/hydro/hydro.cpp`][hydro] | clamp report [L131–196][clamp-report], EOS keys and load [L1102–1186][hydro-eos], floors [L1203–1245][hydro-tfloor], `Tceil` [L1247–1272][hydro-tceil], EOS object [L1661–1729][hydro-ctor], `EstimateHyperbolicTimestep` [L2147–2210][dt-hyp], `FirstOrderFluxCorrect` [L2578ff][fofc] | configuration, initialization, time step (A6, A10) |
| [`src/hydro/rsolvers/glmmhd_hlld.hpp`][hlld] | face energies and fast speeds [L104–126][hlld-e] | Riemann solver (A8.3) |
| [`src/hydro/rsolvers/glmmhd_dc_llf.hpp`][llf] | face energies [L122–143][llf-e] | FOFC fluxes (O-2) |
| [`src/units/physical_units.hpp`][units] | `PhysicalUnits`, `BuildPhysicalUnits` [L84–129][units-build] | units (A5) |
| [`src/pgen/eos_test.cpp`][eostest] | IC kernel | validation problem generator |
| [`src/pgen/collapse_be.cpp`][be] | isothermal $T_0$ [L174–213][be-t0], IC energy [L311–335][be-ic], [L428–440][be-p]; `ApplyBarotropicCooling` [L860–933][be-baro] | A5.4, A8.4 |
| [`src/hydro/double_flux.cpp`][dflux] | `EspFromPRefined` [L71–86][dflux-newton], freeze [L186–208][dflux-freeze] | Sec. 6.4–6.6 |
| [`src/radiation/radiation.cpp`][rad] / [`radiation_sn.cpp`][rad-dtde] | own table load [L288–319][rad-load]; $T$ and $dT/d\varepsilon$ [L923–963][rad-dtde] | A8.4 |
| [`src/hydro/diffusion/diffusion.cpp`][diff-T], [`src/dust/dust.cpp`][dust-T], [`src/refinement/jeans.cpp`][jeans], [`src/diagnostics/cons_diag.cpp`][diag-tfloor] | temperature and sound-speed consumers | A6.2 |
| [`src/eos/TABLE_PROVENANCE.md`][prov] | shipped tables, md5, rebuild commands | A4 |

[gen]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py
[gen-head]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L1-L25
[gen-const]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L28-L39
[gen-zrot]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L62-L70
[gen-erot]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L71-L81
[gen-zvib]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L82-L83
[gen-evib]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L84-L93
[gen-saha]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L95-L110
[gen-neut]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L111-L128
[gen-bisect]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L129-L136
[gen-nh2]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L137-L138
[gen-pe]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L140-L149
[gen-build]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L151-L383
[gen-rhoT]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L218-L225
[gen-pad]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L259-L266
[gen-eaxis]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L227-L258
[gen-sat]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L267-L312
[gen-invert]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L313-L320
[gen-cs2]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L321-L335
[gen-write]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/gen_eos_table.py#L347-L375
[prov]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/TABLE_PROVENANCE.md
[fmt]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table_format.hpp
[tab]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp
[tab-ctr]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L86-L120
[tab-bilin]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L122-L173
[tab-legacy]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L175-L225
[tab-x]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L269-L327
[tab-pof]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L329-L346
[tab-espfromp]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L347-L382
[tab-iface]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L385-L399
[tab-tfromp]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L423-L434
[tab-eintT]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L435-L447
[tab-host]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L454-L540
[tab-load]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L543-L633
[tab-prho]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L592-L602
[tab-ctr2]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L635-L655
[eos-class]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L25-L67
[eos-iefp]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L56-L63
[eos-cf]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L92-L114
[c2p]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L117-L376
[c2p-d]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L155-L162
[c2p-eint]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L174-L176
[c2p-p]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L232-L240
[c2p-pfl]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L266-L283
[c2p-tfl]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L285-L311
[c2p-tceil]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L313-L339
[c2p-mesh]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.cpp#L33-L69
[hydro]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp
[clamp-report]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L131-L196
[hydro-eos]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1102-L1186
[hydro-n14]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1154-L1172
[hydro-hlld]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1174-L1177
[hydro-he]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1188-L1201
[hydro-tfloor]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1203-L1245
[hydro-tceil]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1247-L1272
[hydro-ctor]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1661-L1729
[hydro-dflux]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L1026-L1061
[dt-hyp]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L2147-L2210
[fofc]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/hydro.cpp#L2598-L2632
[hlld]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/rsolvers/glmmhd_hlld.hpp
[hlld-e]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/rsolvers/glmmhd_hlld.hpp#L104-L126
[llf]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/rsolvers/glmmhd_dc_llf.hpp
[units]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/units/physical_units.hpp
[units-T]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/units/physical_units.hpp#L120-L127
[units-build]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/units/physical_units.hpp#L84-L129
[eostest]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/eos_test.cpp#L155-L230
[be]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/collapse_be.cpp
[be-ic]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/collapse_be.cpp#L311-L335
[be-p]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/collapse_be.cpp#L428-L440
[be-baro]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/collapse_be.cpp#L860-L933
[dflux]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/double_flux.cpp
[dflux-newton]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/double_flux.cpp#L71-L86
[dflux-freeze]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/double_flux.cpp#L186-L208
[rad]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/radiation/radiation.cpp
[rad-load]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/radiation/radiation.cpp#L288-L319
[rad-dtde]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/radiation/radiation_sn.cpp#L923-L963
[diff-T]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/diffusion/diffusion.cpp#L104-L127
[dust-T]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/dust/dust.cpp#L259-L270
[jeans]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/refinement/jeans.cpp#L36-L60
[diag-tfloor]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/diagnostics/cons_diag.cpp#L97-L109
[cmake]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/CMakeLists.txt#L69-L72
[eos-tlim]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/adiabatic_glmmhd.hpp#L69-L90
[tab-xhigh]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/eos/eos_table.hpp#L310-L319
[llf-e]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/hydro/rsolvers/glmmhd_dc_llf.hpp#L122-L143
[be-t0]: https://github.com/rahul-9670/athenapk-selfgravity-eos/blob/5649acdf8cc46a5340f768c1443b9870aad45125/src/pgen/collapse_be.cpp#L174-L213
