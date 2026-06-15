# 0 — Datasets

This folder holds **raw input data** for calibration scripts in
[`1 - PARAMETRIC INPUT`](../1%20-%20PARAMETRIC%20INPUT/readme.md). It is **not** included in the
repository (files are too large).

## Download

Download **all** dataset bundles from the
[NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) and place the
extracted files directly in this folder (`0 - Datasets/`).

| Dataset bundle | Used by |
|----------------|---------|
| **TGSIM freeway trajectories** (I-395, I-90/94, I-294) | Car-following calibration, lane-changing preprocessing, lateral calibration |
| **TGSIM Foggy Bottom VRU trajectories** | VRU Social Force / PT calibration |
| **Waymo Open Motion** (pre-processed leader–follower + map CSVs) | Waymo IDM calibration, scenario visualization |
| **Lane-change / DDM organized data** (`organized_data.pkl`, lane-change CSVs) | Drift Diffusion Model lane-change calibration |

---

## Expected layout

After downloading, your folder should look like this (filenames match those expected by
[`ngm_paths.py`](../ngm_paths.py) — no per-script path edits needed):

```
0 - Datasets/
├── readme.md
│
├── March2023waymo_scenario_lane_leader_follower_assigned_{196–199}_data.csv
├── March2023waymo_map_features_{196–199}_data.csv
│
├── Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv
├── Third_Generation_Simulation_Data__TGSIM__I-90_I-94_Moving_Trajectories.csv
├── Third_Generation_Simulation_Data__TGSIM__I-90_I-94_Stationary_Trajectories.csv
├── Third_Generation_Simulation_Data__TGSIM__I-294_L1_Trajectories.csv
├── Third_Generation_Simulation_Data__TGSIM__I-294_L2_Trajectories.csv
├── Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv
│
├── all_lane_changes_output.csv          # optional: from 1.2.1 Data Processing (or download pre-built)
├── Dis-surrounding_info_with_event_id.csv
├── organized_data.pkl                   # for DDM fit_DDM_Lane_Change.py
```

---

## Notes

- Stage 1 scripts resolve paths through **`ngm_paths.py`** at the repo root (`dataset_path()`,
  `calibration_dataset_paths()`, `load_tgsim_csv()`, etc.). Place downloaded files here using the
  filenames in the layout above.
- `Waymo_IDM_CF_Calibration.py` and `Waymo_vis_scenarios.py` read the pre-processed Waymo CSVs from
  this folder via `DATASETS_DIR`.
- Raw data files are **git-ignored** (see root `.gitignore`). Only this `readme.md` is versioned.
- If you add new datasets, document them in the table above and add constants/helpers to `ngm_paths.py`.
