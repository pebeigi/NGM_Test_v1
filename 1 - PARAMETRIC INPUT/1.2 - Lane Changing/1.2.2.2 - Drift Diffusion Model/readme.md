# 1.2.2.2 — Drift Diffusion Model (DDM)

Calibration of a **Drift Diffusion Model (DDM)** for discrete **left / right lane-change decisions**.

## Code

| File | Description |
|------|-------------|
| `fit_DDM_Lane_Change.py` | Fits the DDM with an **MPI-parallel genetic algorithm** (`geneticalgorithm` + `mpi4py`, with numba acceleration). Estimates population-level means, standard deviations, and covariances for the drift-rate coefficients (`beta_L/R/G/V/MLC`, `G0`, `sigma`, `alpha`) — 18 parameters in total. Writes incremental checkpoints to `progress/gen{N}ind{M}.pkl` and a final `progress/final_result.pkl`. |
| `examine_results.ipynb` | Loads the `progress/gen*.pkl` checkpoints, finds the run with minimum log-likelihood, and prints the best DDM parameter vector. |
| `readme` | One-line instruction: run the `.py` to calibrate, then read results with the notebook. |
| `progress/` | Holds intermediate GA-iteration pickles (see `progress/readme.md`). |

## Input

`organized_data.pkl` — a nested dictionary of vehicle trajectories with lane labels, from
[`0 - Datasets/`](../../../0%20-%20Datasets/readme.md) (resolved by `dataset_path()` in
[`ngm_paths.py`](../../../ngm_paths.py)). Download from the Kaggle bundle or build from TGSIM
preprocessing. (`organized_data_I94_AV.pkl` in this folder, if present, is a related AV-subset file.)

## Running

```bash
mpiexec -n <N> python fit_DDM_Lane_Change.py   # MPI-parallel calibration
# then open examine_results.ipynb to inspect the best parameters
```
