# model_params

Calibrated **car-following parameter pools** produced in
[`1 - PARAMETRIC INPUT`](../../1%20-%20PARAMETRIC%20INPUT/readme.md) and sampled at runtime by
[`Simulate_Freeway.ipynb`](../readme.md). Each file holds one calibrated fit per row (follower / run).

Copy or symlink the `merged_IDM_*.csv` and `merged_PT_*.csv` files from
[`3 - SIMULATION/models/model_params/`](../../3%20-%20SIMULATION/models/model_params/readme.md)
(or regenerate them in Stage 1) before running the notebook. CSV files are **git-ignored** in this
repository.

## Vehicle-class suffix

| Suffix | Meaning | Notebook type index | Length |
|--------|---------|---------------------|--------|
| `S` | Small vehicle (human-driven car) | 0 | 4 m |
| `A` | Automated vehicle (AV/CAV) | 1 | 4 m |
| `L` | Large / heavy vehicle (truck) | 2 | 12 m |

At runtime a vehicle's type is drawn with `random.choice([0, 1, 2], p=[sv_rate, av_rate, lv_rate])`, and
its parameters are sampled from the matching CSV.

## Files

| File | Model | Key columns used by the simulation |
|------|-------|------------------------------------|
| `merged_IDM_{S,A,L}.csv` | Intelligent Driver Model | `T` (time headway), `a` (max accel), `b` (comfort decel), `v0` (desired speed), `so` (min gap), `delta` (accel exponent) — plus fit metrics and `source`. |
| `merged_PT_{S,A,L}.csv` | Prospect Theory | `Tmax`, `Alpha`, `Beta`, `Wc`, `Gamma1`, `Gamma2`, `Wm` — plus fit metrics and `source`. |

The `source` column records which calibration dataset each row came from (e.g. `I294l1`, `I294l2`,
`I395`, `I9094`).
