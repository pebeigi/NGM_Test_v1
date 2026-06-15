# 1.1 — Car-Following

Calibration of **longitudinal (car-following) models** against real leader–follower trajectories. Two
models are fit: the **Intelligent Driver Model (IDM)** and a **Prospect Theory (PT)** car-following
model. Calibration uses two data sources, split into two subfolders.

## Subfolders

| Folder | Data source | Purpose |
|--------|-------------|---------|
| [`1.1.1.2 - Waymo Motion`](1.1.1.2%20-%20Waymo%20Motion/readme.md) | Waymo Open Motion | IDM calibration on Waymo leader–follower pairs, plus scenario/map visualization notebooks. |
| [`1.1.2 - Car-Following Parametric Analysis`](1.1.2%20-%20Car-Following%20Parametric%20Analysis/readme.md) | TGSIM (I-395, I-90/94, I-294) | IDM and PT calibration on freeway trajectories; consolidated results and summary tables. |

## Vehicle-class grouping

Events are grouped by follower vehicle class — **S** (small, length < ~6 m), **L** (large/heavy,
length ≥ ~6 m), and **A** (automated / self-driving vehicle, flagged in the data). Results are reported
both per event and aggregated by follower–leader type combination (S-S, S-L, …, A-A).

## Outputs

Per-event calibrated parameters, simulated trajectories, fit metrics (MSE/RMSE/R²/…), validation plots,
and aggregate summary tables. The summary parameters become the IDM/PT pools used by the simulator.

> TGSIM and Waymo inputs are read from [`0 - Datasets/`](../../0%20-%20Datasets/readme.md) via
> [`ngm_paths.py`](../../ngm_paths.py).
