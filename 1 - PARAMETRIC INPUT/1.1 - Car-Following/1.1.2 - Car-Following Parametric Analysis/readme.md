# 1.1.2 — Car-Following Parametric Analysis

IDM and Prospect Theory (PT) car-following calibration on **TGSIM** freeway corridors
(I-395, I-90/94 moving, I-294 lanes 1 & 2). Each script extracts the longest stable leader–follower
segment per follower event, fits the model with a genetic algorithm, simulates the result, and exports
parameters, simulated trajectories, and validation plots.

## Code

| File | Description |
|------|-------------|
| `IDM_CF_Calibration.py` | Reference **IDM** GA calibration over predefined vehicle groups across the TGSIM datasets. Outputs `IDM_Params_{group}.csv`, `IDM_Simulated_{group}.csv`, and per-event PNGs. |
| `IDM_CF_Calibration.ipynb` | Notebook version of the IDM workflow with hard-coded vehicle-group lists and local dataset paths, plus extra cells for stop-and-go analysis, trajectory export, and batch plotting. |
| `PT_CF_Calibration.py` | **Prospect Theory** car-following calibration (Talebpour-style). GA fits `(Tmax, Alpha, Beta, Wc, Gamma1, Gamma2, Wm)` per event. Outputs `PT_Params_{group}.csv`, `PT_Simulated_{group}.csv`, plots. |
| `PT_CF_Calibration.ipynb` | Notebook mirror of the PT calibration using the same groups and inputs. |

> Expected input columns (TGSIM): `time`, `ID`, `run-index`, `xloc-kf`/`yloc-kf`, `speed-kf`,
> `lane-kf`, `type-most-common`, and an `ACC`/`AV` flag. Dataset paths in the scripts are placeholders;
> point them at your local TGSIM CSVs.

## Results (`Results/`)

| File | Description |
|------|-------------|
| `IDM_results.csv` | Consolidated table (~13k rows) of all IDM calibration runs across datasets/groups: params + fit metrics per event. |
| `PT_results.csv` | Same structure for PT calibrations (~13k rows). |
| `CF_Type_Summary_Table.csv` | Pivot summary of mean IDM and PT parameters and mean RMSE/MAE/R² **by follower–leader type** (S-S … A-A), with event counts. This is the parametric input consumed downstream. |
