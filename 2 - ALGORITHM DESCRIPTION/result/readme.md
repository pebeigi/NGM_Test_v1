# result

Output location for **post-simulation trajectory files** (Algorithm Description Document §2.5.2).

[`Simulate_Freeway.ipynb`](../readme.md) writes per-step trajectory CSVs named
`trajs_{site}_{scenarios}_trial{t}.csv`, containing columns such as
`time, veh_id, type, length, x, y, v, road, lane, lane_pos`.

> By default the notebook saves these in its working directory; point the save path here to keep outputs
> organized. This folder is otherwise a placeholder.
