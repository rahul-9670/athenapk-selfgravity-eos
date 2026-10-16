# EOS table provenance — what ships, and how to rebuild it

Three tabulated protostellar-EOS tables are committed here. Both are produced by
`gen_eos_table.py` (multi-Saha: H2 dissociation + H ionization + He/He+ ionization + H2
rotational/vibrational levels + inert He), stored in **cgs** and converted at load by the run's
own unit system.

| file | ver | nr | ne / nT | bytes | md5 | rho range [g/cm^3] |
|---|---|---|---|---|---|---|
| `eos_table_hires_v5_cold.bin` | 2 (cgs) | 400 | ne 1399 / nT 1370 | 17 814 512 | `f712d6a806cb82889214fec95a6c2681` | 1e-20 … 1 |
| `eos_table_hires_v5_cold_rho26.bin` | 2 (cgs) | 520 | ne 1399 / nT 1370 | 23 158 832 | `cb68bd0f919a9a929af3601c5befb686` | 1e-26 … 1 |
| `eos_table_hires_v6_rho26.bin` | 2 (cgs) | 520 | ne 1399 / nT 1370 | 23 158 832 | `69aca3f9c4fa0a96a85dc65eed45d454` | 1e-26 … 1 |

`eos_table_hires_v6_rho26.bin` is the recommended table (2026-10-16). It is the `v5_cold_rho26` grid built with the
generator's **consistent** energy model: a dissociated H2 molecule is charged chi_d - (3/4) 2 k theta_rot, the
dissociation energy from the frozen 3:1 ortho:para ground mixture, instead of chi_d (EOS_VALIDATION.md A3.5, T-4).
The two `v5_cold` tables use the **legacy** model; their Maxwell-identity residual (2.46e-3 in the H2-dissociation
band) is that zero-point mismatch, the v6 residual is finite-difference truncation (4.9e-5). `eos_table_hires_v5_cold_rho26.bin`
is `v5_cold` with 120 rows prepended below 1e-20 g/cm^3 at the same 0.05-dex spacing; the original 400 rows are unchanged.
The `v5_cold` tables stay shipped so that runs made with them can be reproduced.
Select a table with `<hydro> eos = hydrogen` and `eos_table_file = <path>`.

The file version is the first `int64`: `0x454F535441424C31` ("EOSTABL1") for v2. `flags` bit 0
set = arrays stored in cgs.

## Rebuild

```bash
cd src/eos
python3 gen_eos_table.py build 400 1000 920 eos_table_hires_v5_cold.h5 399 450 3.0e5 0 0 0 legacy        # -> eos_table_hires_v5_cold.bin
python3 gen_eos_table.py build 400 1000 920 eos_table_hires_v5_cold_rho26.h5 399 450 3.0e5 0 0 120 legacy
python3 gen_eos_table.py build 400 1000 920 eos_table_hires_v6_rho26.h5 399 450 3.0e5 0 0 120 consistent
md5sum eos_table_hires_v5_cold*.bin                                                    # compare with the table above
```

(`build_hires_v5_cold.sh`, `build_hires_v5_cold_rho26.sh` and `build_hires_v6_rho26.sh` wrap the same commands.
Since 2026-10-16 the energy model is the 12th argument and defaults to `consistent`: the two `v5_cold` tables MUST be
rebuilt with `legacy`, which reproduces them byte-identically with the current generator (verified 2026-10-16).) The build is a
deterministic computation on fixed log grids, so the same generator reproduces the same bytes.
**If the md5 does not match, the generator changed and the EOS changed with it.**

### Arguments
* `400 1000 920` — base resolution in (rho, esp, T).
* `399 450` — `esp_pad_pts` / `T_pad_pts`: nodes **prepended** below the base esp / T axes at the
  same spacing, so the base nodes keep their exact values. They extend the cold end to
  log10 esp_code = -2.0308 (T ≈ 0.06 K) so cold gas is never edge-clamped. The generator hard-fails
  if the T pad does not cover the esp pad, rather than shipping a flat extrapolation.
* `consistent` / `legacy` — the energy model (see above).
* `3.0e5 0 0 120` (rho26 only) — T_max, two unused pads, and `rho_pad_pts = 120`. The esp and T
  axes are derived from the ORIGINAL rho rows only, so the extension does not rescale them
  (`esp_max` is set by the lowest density; see AUDIT F-19 in `gen_eos_table.py`).

### Why the axes are padded, not regenerated
The esp floor is set by `T_min` (at low T, esp = (3/2) kT/(mu m_H) exactly, mu = 2.331). Lowering
`T_min` at fixed `nT` would rescale the grid spacing and move *every* node; padding keeps the
validated nodes bit-identical.

## Regression (measured, not assumed)

* v6_rho26 vs v5_cold_rho26 (2026-10-16): identical header (the esp axis is pinned to the legacy energies, see
  `build_table`), so every node lines up. Nodes below 500 K: 99.5 % bit-identical, the rest within 2e-11 (traces of
  dissociation at the lowest densities). Above 1000 K, at fixed (rho, esp): P median 3.6e-4, 99 % 8.2e-3, max 2.3e-2;
  cs2 max 5.0e-2; log T max 3.1e-3 over 1e-18 .. 1e-2 g/cm^3 (largest where dissociation is nearly complete and the
  energy is mostly dissociation energy; at the 1e-26 rows up to 3.7e-2 in P). At fixed (rho, T) the energy changes by
  -0.12 % (2000 K, 1e-9) to -0.20 % (3000 K, 1e-8) (T-4). Consistency gates (`tests/test_eos_consistency.py`): all
  pass; Maxwell 4.89e-5 (tolerance 2e-4 for the consistent model), table vs Saha P median 1.56e-3.

* Padding vs. the unpadded base table, over the base domain: `P`, `log10 T`, `espRT` **bit-identical**;
  `cs2` differs only on the former edge column (`np.gradient` one-sided -> centred, ≤ 2.5e-3).
* rho26 vs v5_cold, original 400 rows: `P`, `log10 T`, `espT` **bit-identical**; `cs2` differs only
  on the former edge row (up to 6.1 %, the old one-sided value being the worse one).
  `rho_pad_pts = 0` reproduces `eos_table_hires_v5_cold.bin` byte-identically.
* New low-density rows vs a direct Saha calculation at off-node densities: max 2.0e-2, typically
  1e-3 (the code's ideal-gas continuation below the table, `hydro/eos_extrapolate`, is up to 3.2e-1
  off in the H2-dissociation and H-ionization bands).
* Cold-end physics check, independent of the generator: `T(esp) = (2 mu m_H / 3k) esp` to 1.7e-14,
  density-independent to 1.6e-15 across all rho rows.

`verify_table_extension.py <old.bin> <new.bin>` runs the node-level checks; `tests/test_eos_consistency.py`
runs the thermodynamic-consistency gate. The full EOS validation is in
`validation/eos_tabulated/EOS_VALIDATION.md`.

## Edge behaviour
Lookups outside the table domain are clamped (or, with `hydro/eos_extrapolate = true`, the default,
continued analytically); every out-of-table lookup is counted per edge and reported, so clamping is
never silent.
