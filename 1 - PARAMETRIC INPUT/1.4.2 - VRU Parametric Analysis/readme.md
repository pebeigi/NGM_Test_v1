# 1.4.2 — VRU Parametric Analysis

Calibration of **vulnerable-road-user (VRU)** behavior — pedestrians and bicycles — using a mixed
**Social Force (SF)** model and a **Prospect Theory (PT)** model, followed by optional reinforcement-
learning fine-tuning.

## Shared input

`Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv` — TGSIM Foggy Bottom
trajectories (pedestrians `type_most_common = 0`, bicycles `= 1`), filtered to lanes `[1, 39, 40, 41]`,
coordinates in meters. Not stored in the repo; set the path in each script.

## Code

| File | Description |
|------|-------------|
| `SF_calibration_code.py` | GA calibration of a **mixed Social Force model** per agent ID. Pedestrians: 4 free params (`A_pp`, `B_pp`, `A_wall`, `B_wall`); bicycles: 5 free params (`eps_m`, `A_w`, `B_w`, `A_s`, `B_s`) with Wang et al. fixed bicycle dynamics. Outputs `calibration_per_id.csv`, `calibration_predictions.csv`, and metric plots in `sfm_plots/`. |
| `PT_Calibration_code.py` | GA calibration of a **Prospect Theory VRU model** with expanded collision weights (6 `Wc_*` terms) plus per-mode PT params (`eta`, `xi`, `tau`, fixed `v_pref`) — 14 params per ID. Outputs to `pt_collision_weights_expanded/`. |
| `corr_plot.py` | Post-calibration **parameter correlation** analysis (pairwise KDE/histogram grids and heatmaps) from the SF results. |
| `err_plot.py` | **Error-metric distributions** (RMSE, MAE, R²) from the SF calibration, by VRU type. |
| `plot_code.py` | Overlays **observed vs. predicted** x/y/vx/vy time series for the highest-presence agent of each VRU type. |
| [`RL_finetuning/`](RL_finetuning/readme.md) | PPO reinforcement-learning fine-tuning of the GA-calibrated SF/PT parameters. |

## Data / results (in this folder)

| File | Description |
|------|-------------|
| `SF_calibrated_params.csv` | Per-ID Social Force calibration results (params + lane + type + fit metrics; ~540 agents). |
| `SF_calibration_predictions.csv` | Timestep-level observed vs. predicted positions/velocities for calibrated SF agents. |
| `PT_calibrated_params.csv` | Per-ID PT calibration (14 params + fit metrics). |
| `pt_expanded_collision_calib.csv` | Primary PT GA output with collision weights and PT mode parameters. |
