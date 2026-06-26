# MOBIL

Calibration of the **MOBIL** lane-change model.

## Code

| File | Description |
|------|-------------|
| `MOBIL.py` | Calibrates the MOBIL **politeness factor `p`** per lane-change event with a small genetic algorithm, using the speed changes of the changing vehicle, the new-lane follower, and the old-lane follower. Produces a list of optimal `p` values and bar/line plots of `p` and the associated speed changes. |

## Input

`Dis-surrounding_info_with_event_id.csv` — lane-change events with surrounding-vehicle information,
from [`0 - Datasets/`](../../../0%20-%20Datasets/readme.md) (resolved by `dataset_path()` in
[`ngm_paths.py`](../../../ngm_paths.py)). Produced by lane-change preprocessing or supplied from
the Kaggle bundle.

## Output

Per-event optimal politeness factors (in memory / printed) and diagnostic plots. The calibrated MOBIL
parameters correspond to the `MOBIL_results.csv` pool used by the simulator.
