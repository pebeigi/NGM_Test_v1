# 1.1.1.2 — Waymo Motion

IDM car-following calibration and scenario visualization using the **Waymo Open Motion Dataset**.

## Code

| File | Description |
|------|-------------|
| `Waymo_IDM_CF_Calibration.py` | Calibrates the **Intelligent Driver Model (IDM)** on pre-processed Waymo leader–follower trajectories with a genetic algorithm (40 population × 80 generations). Groups events by **follower type** into `Waymo_AV` (SDC), `Waymo_SV` (length < 6 m), and `Waymo_HV` (≥ 6 m); skips segments shorter than 12 s. Longitudinal position is **distance along the follower's assigned lane polyline** (leader projected onto the same lane); falls back to path arc length if map geometry is missing. Fits `(T, a, b, v0, s0, δ)` per event, simulates closed-loop, and scores against observed kinematics. CLI flags include `--max-events`, `--plot-only`, `--summary-only`. |
| `Waymo_vis_scenarios.py` | Visualizes agent trajectories from the pre-processed Waymo CSVs (no TFRecords). `--mode map` overlays lane/road geometry; `--mode nomap` plots trajectories only; `--mode both` (default) writes both. Outputs PNGs to `Results/plots_scenarios/`. |

> `Waymo_vis_scenarios.py` only needs **matplotlib**, **numpy**, and **pandas** — no TensorFlow or
> `waymo-open-dataset`.

## Input data (pre-processed, local only)

> These raw Waymo CSVs are large (~800 MB total, individual files exceed GitHub's size limit) and are
> **not versioned**. Download them from the
> [NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) and place them in
> [`0 - Datasets/`](../../../0%20-%20Datasets/readme.md) at the repo root. Both scripts import
> `DATASETS_DIR` from [`ngm_paths.py`](../../../ngm_paths.py).

| File pattern (in `0 - Datasets/`) | Description |
|-----------------------------------|-------------|
| `March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv` | Per-timestep vehicle tracks with assigned lane, leader/follower IDs, SDC flag, position and velocity. Main calibration input (all matching files in `0 - Datasets/` are loaded). |
| `March2023waymo_map_features_*_data.csv` | Per-scenario map geometry: lane polylines, road edges/lines, crosswalks, speed limits. Paired by suffix with trajectory files; used for lane-centered longitudinal coordinates, visualization, and diagnostics. |

## Results (`Results/`)

| File | Description |
|------|-------------|
| `IDM_Params_Waymo_{AV,SV,HV}.csv` | Per-event calibrated IDM parameters and fit metrics by follower type (AV / SV / HV). |
| `IDM_Simulated_Waymo_{AV,SV,HV}.csv` | Simulated follower trajectories matched to observed time windows. |
| `IDM_Summary_Waymo.csv` | Mean IDM parameters and RMSE aggregated by follower type (AV, SV, HV). |
| `plots/IDM_Waymo_*_FID_*_LID_*_sc_*.png` | Per-event validation: x–y plan view (reference lane highlighted), lane distance vs time, speed vs time, time–x, time–y, lane-s error. Generated locally; not versioned. |
| `plots_scenarios/{scenario_id}_{map,nomap}.png` | Scenario trajectory plots from `Waymo_vis_scenarios.py`. Generated locally; not versioned. |
