# 1 — Parametric Input

This stage turns **raw trajectory data into calibrated behavioral-model parameters**. Each subfolder
targets one behavioral domain, fits one or more models to field data (mostly with genetic algorithms),
and exports per-class parameter CSVs that feed the simulator in [`3 - SIMULATION`](../3%20-%20SIMULATION/README.md)
and the reference notebook in [`2 - ALGORITHM DESCRIPTION`](../2%20-%20ALGORITHM%20DESCRIPTION/readme.md).

## Subfolders

| Folder | Domain | Models calibrated |
|--------|--------|-------------------|
| [`1.1 - Car-Following`](1.1%20-%20Car-Following/readme.md) | Longitudinal (car-following) | IDM, Prospect Theory (PT) |
| [`1.2 - Lane Changing`](1.2%20-%20Lane%20Changing/readme.md) | Lane-change decisions | MOBIL, Drift Diffusion Model (DDM) |
| [`1.3 - Lateral Calibration`](1.3%20-%20Lateral%20Calibration/readme.md) | Lateral motion during lane changes | Curvilinear transform + polynomial shape models |
| [`1.4.2 - VRU Parametric Analysis`](1.4.2%20-%20VRU%20Parametric%20Analysis/readme.md) | Pedestrians & bicycles (VRU) | Social Force (SF), Prospect Theory (PT) |

## Conventions

- **Vehicle classes:** `S` = small vehicle, `A` = automated vehicle (AV), `L` = large/heavy vehicle.
  Follower–leader pairs are written e.g. `S-L`, `A-S`.
- **Datasets:** car-following and lane-changing use **TGSIM** corridors (I-395, I-90/94, I-294 lanes 1/2)
  and **Waymo Open Motion**; VRU work uses the **TGSIM Foggy Bottom** pedestrian/bicycle trajectories.
- **Calibration method:** most scripts use a genetic algorithm to minimize trajectory error
  (RMSE/MAE/R²) per event, then aggregate parameters by class and follower–leader type.

## Important: data paths

Download datasets from [Kaggle — NGM Datasets](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets)
and extract the CSV files into `0 - Datasets/` at the repository root (see `0 - Datasets/readme.md`).

Scripts and notebooks resolve paths through **`ngm_paths.py`** at the repo root — no machine-specific
paths need to be edited. If you add a new script in a subfolder, import helpers from `ngm_paths`
(e.g. `calibration_dataset_paths()`, `load_tgsim_csv()`, `dataset_path()`).

## Outputs that feed downstream stages

Calibrated CSVs produced here are the source of the parameter pools copied into
`3 - SIMULATION/models/model_params/` and `2 - ALGORITHM DESCRIPTION/model_params/` — for example the
merged IDM/PT parameter tables and the MOBIL results used to sample per-vehicle behavior at runtime.
