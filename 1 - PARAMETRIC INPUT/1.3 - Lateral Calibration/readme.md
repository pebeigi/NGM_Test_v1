# 1.3 — Lateral Calibration

Calibration of **lateral vehicle motion during lane changes**. Because some TGSIM corridors are not
aligned to an axis, trajectories are first transformed into curvilinear (road-aligned) coordinates, then
the lateral shape of each lane change is fit with polynomials.

## Notebooks

| File | Description |
|------|-------------|
| `Find_Center_of_Lanes.ipynb` | **Curvilinear coordinate transform.** Uses LOWESS-smoothed lane centerlines to straighten trajectories relative to a reference lane for TGSIM corridors that are not axis-aligned (I-294 L1/L2, I-90/94). I-395 is already y-axis aligned and skipped. Outputs `Processed/*_Curvilinear_Transformation_for_Lateral_Calibration.csv` plus before/after diagnostic plots. |
| `LC_Lateral_Calibration.ipynb` | **Lateral lane-change shape calibration** on the curvilinear data. (1) Detects lane changes and extracts ±20 s windows → `lc_processed_data.csv`; (2) fits cubic/quintic polynomials to lateral position during each change and computes RMSE/MAE/R²; (3) filters high-R² events and fits a linear model of lane-change duration vs. speed. Outputs `lc_processed_data.csv`, `Lateral_Calibration.csv`, per-event PNGs in `LC_Plots/`, and a duration-vs-speed plot. |

## Workflow

Run `Find_Center_of_Lanes.ipynb` first to produce curvilinear-transformed trajectories, then
`LC_Lateral_Calibration.ipynb` to extract lane-change windows and fit the lateral-motion models.

> Inputs are local TGSIM CSV copies; update the paths at the top of each notebook.
