# 2 — Algorithm Description

This stage is the **reference implementation** of the freeway simulation framework. It takes the
parameters calibrated in [`1 - PARAMETRIC INPUT`](../1%20-%20PARAMETRIC%20INPUT/readme.md) and drives
**custom car-following models inside SUMO** (via TraCI) on three real-world freeway corridors. It also
documents how the behavioral models are integrated with SUMO and, optionally, OMNeT++ for
communication-based experiments. For full narrative detail, see the *Algorithm Description Document*.

> The production, GUI-driven simulator lives in [`3 - SIMULATION`](../3%20-%20SIMULATION/README.md);
> this stage is the simpler, notebook-based reference that the production tool extends.

## Contents

| Path | Description |
|------|-------------|
| `Simulate_Freeway.ipynb` | End-to-end freeway microsimulation notebook (details below). |
| [`configs/`](configs/readme.md) | SUMO network, trip, and lane-connectivity files for the three freeway corridors. |
| [`model_params/`](model_params/readme.md) | Calibrated IDM and Prospect Theory parameter pools for SV / AV / HV. |
| [`result/`](result/readme.md) | Output location for post-simulation trajectory files. |
| [`2.5.3 Data Analysis/`](2.5.3%20Data%20Analysis/readme.md) | Placeholder for the data-analysis subsection. |

## `Simulate_Freeway.ipynb`

A SUMO + TraCI freeway simulation that replaces SUMO's built-in car-following with **custom IDM or
Prospect Theory** models:

1. **Load parameters** — reads `merged_IDM_{S,A,L}.csv` and `merged_PT_{S,A,L}.csv` from `model_params/`,
   indexed by vehicle type (`0` = SV, `1` = AV, `2` = HV).
2. **`initialize()`** — site-specific setup for `I90_94`, `I294`, or `I395`: on/off-ramp edge IDs, the
   origin–destination demand matrix, mandatory lane-change edges, and the `next_lanes_*.pickle`
   downstream-lane lookup.
3. **`generate_vehicles()`** — Poisson arrivals per OD pair; assigns vehicle type by mix
   `[sv_rate, av_rate, lv_rate]`, samples parameters from the CSV rows, sets length (4 m for cars,
   12 m for trucks), and writes `{site}_trial{t}.trips.xml`.
4. **`IDM()` / `PT()`** — custom acceleration models computed each step from TraCI leader/gap state.
5. **`run_simulation()`** — runs SUMO at 0.1 s steps, overrides speeds with the custom model, and
   records per-step trajectories (`time, veh_id, type, length, x, y, v, road, lane, lane_pos`), saved as
   `trajs_{site}_{scenarios}_trial{t}.csv`.

> The notebook uses hard-coded site demand and example fleet mixes; lane-changing logic is partially
> stubbed. It is a reference/demonstration rather than a turn-key tool.

## Document outline (reference)

- **2.1** SUMO
- **2.2** Extension Modules — 2.2.1 Network Construction · 2.2.2 Trip Generation · 2.2.3 Model
  Customization · 2.2.4 Data Collection
- **2.3** Simulation Framework
- **2.4** OMNeT++ Integration — 2.4.1 Setup · 2.4.2 Communication and Control Modules
- **2.5** Simulation Output — 2.5.1 Visualization · 2.5.2 Output File · 2.5.3 Data Analysis
