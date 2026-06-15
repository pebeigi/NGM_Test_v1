# 1.2 — Lane Changing

Detection of lane-change events from trajectory data and calibration of **lane-change decision models**.

## Contents

| Path | Description |
|------|-------------|
| `1.2.1 - Data Processing.py` | Scans TGSIM trajectory CSVs (I-395, I-90/94, I-294 L1/L2) for **lane-change events** by detecting `lane-kf` transitions, and records the nearest front/behind vehicles in the origin and destination lanes. Outputs `all_lane_changes_output.csv` and optional scatter plots in `lane_change_plots/`. |
| [`1.2.2.1 - MOBIL`](1.2.2.1%20-%20MOBIL/readme.md) | Calibration of the **MOBIL** lane-change model (politeness factor). |
| [`1.2.2.2 - Drift Diffusion Model`](1.2.2.2%20-%20Drift%20Diffusion%20Model/readme.md) | Calibration of a **Drift Diffusion Model (DDM)** for discrete left/right lane-change decisions. |

## Workflow

`1.2.1 - Data Processing.py` produces the event table (changing vehicle plus surrounding traffic). The
MOBIL and DDM subfolders then consume the extracted events to fit their respective decision models. The
resulting parameters feed the lane-changing logic in [`3 - SIMULATION`](../../3%20-%20SIMULATION/README.md).

> Dataset paths are hard-coded to local copies; update them before running.
