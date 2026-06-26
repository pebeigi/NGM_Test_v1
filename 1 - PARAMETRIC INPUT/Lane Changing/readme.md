# Lane Changing

Detection of lane-change events from trajectory data and calibration of **lane-change decision models**.

## Contents

| Path | Description |
|------|-------------|
| `Data Processing.py` | Scans TGSIM trajectory CSVs (I-395, I-90/94, I-294 L1/L2) for **lane-change events** by detecting `lane-kf` transitions, and records the nearest front/behind vehicles in the origin and destination lanes. Writes `all_lane_changes_output.csv` to `0 - Datasets/` and optional scatter plots in `lane_change_plots/`. |
| [`MOBIL`](MOBIL/readme.md) | Calibration of the **MOBIL** lane-change model (politeness factor). |
| [`Drift Diffusion Model`](Drift%20Diffusion%20Model/readme.md) | Calibration of a **Drift Diffusion Model (DDM)** for discrete left/right lane-change decisions. |

## Workflow

`Data Processing.py` produces the event table (changing vehicle plus surrounding traffic). The
MOBIL and DDM subfolders then consume the extracted events to fit their respective decision models. The
resulting parameters feed the lane-changing logic in [`2 - SIMULATION`](../../2%20-%20SIMULATION/README.md).

> TGSIM CSV paths and lane-change outputs are resolved through [`ngm_paths.py`](../../ngm_paths.py)
> (`calibration_dataset_paths()`, `lane_change_dataset_paths()`, `dataset_path()`).
