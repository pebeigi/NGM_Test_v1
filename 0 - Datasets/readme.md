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
| **TGSIM Foggy Bottom VRU trajectories** | VRU Social Force / PT calibration, RL fine-tuning |
| **Waymo Open Motion** (pre-processed leader–follower + map CSVs) | Waymo IDM calibration, scenario visualization |
| **Lane-change / DDM organized data** (`organized_data.pkl`, lane-change CSVs) | Drift Diffusion Model lane-change calibration |

---

## Expected layout

After downloading, your folder should look like this (exact filenames may vary — update paths in
each script/notebook to match):

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

- Scripts under `1 - PARAMETRIC INPUT` often use **hard-coded absolute paths**. Point them at files
  under this folder (or copy/symlink data next to the script if that is easier).
- `Waymo_IDM_CF_Calibration.py` reads the pre-processed Waymo CSVs from this folder by default.
- If you add new datasets, document them in the table above so others know where to download and
  where to put files.
