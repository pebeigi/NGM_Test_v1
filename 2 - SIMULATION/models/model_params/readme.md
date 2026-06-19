# model_params

Calibrated parameter libraries sampled at runtime by the scenario runners in
[`models/`](../readme.md). They originate from the calibration work in
[`1 - PARAMETRIC INPUT`](../../../1%20-%20PARAMETRIC%20INPUT/readme.md). When "Default Parameters" is
selected in the GUI, per-class Mean/Std defaults are derived from these files.

> These CSVs are **versioned** in the repo (exempt from the root `*.csv` ignore rule) so a fresh clone
> can run the GUI with calibrated defaults. Regenerate or replace them from Stage 1 calibration outputs
> when you update the parameter pools.

## Car-following pools — `merged_IDM_*.csv`, `merged_PT_*.csv`

One calibrated fit per row. The suffix encodes the vehicle class:

| Suffix | Meaning | Sim type code |
|--------|---------|---------------|
| `S` | Small vehicle (SV) | 0 |
| `A` | Automated vehicle (AV); CAV also draws from `A` for IDM/C-IDM priors | 1 (and 3) |
| `L` | Large / heavy vehicle (HV) | 2 |
| `CL` | Connected large — connected heavy vehicle (CAHV) | 4 |

- `merged_IDM_{S,A,L,CL}.csv` — IDM fields `T`, `a`, `b`, `v0`, `s0`, `delta` plus fit metrics and `source`.
- `merged_PT_{S,A,L,CL}.csv` — Prospect Theory fields `Tmax`, `Alpha`, `Beta`, `Wc`, `Gamma1`,
  `Gamma2`, `Wm` plus fit metrics and `source`.

## Lane-changing — `MOBIL_results.csv`

Calibrated MOBIL rows with discretionary/mandatory parameters (`Discretionary_p_optimal`,
`Discretionary_a_th`, `Discretionary_b_safe`, `Mandatory_b_safe`, …). **Not** split by vehicle class —
one pool is sampled for all human-driven vehicles.

## Active travel (pedestrians / bikes)

| File | Model |
|------|-------|
| `SF_atm_params.csv` | Social Force pedestrian/bike parameters (`A_pp`, `B_wall`, `v_gamma0`, …). |
| `PT_atm_params.csv` | Prospect Theory active-travel parameters (`Wc_*`, `eta_ped`, `tau_bike`, …). |

> If a parameter file is missing, the code falls back to zeros or hard-coded defaults.
