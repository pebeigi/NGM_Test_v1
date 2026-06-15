# configs

SUMO simulation assets for the three **real-world freeway corridors** used by
[`Simulate_Freeway.ipynb`](../readme.md). Each corridor has its own subfolder:

| Subfolder | Corridor |
|-----------|----------|
| [`I90_94/`](I90_94/readme.md) | Chicago I-90 / I-94 (Edens Expressway area) |
| [`I294/`](I294/readme.md) | Chicago I-294 (Tri-State Tollway) |
| [`I395/`](I395/readme.md) | Washington, D.C. I-395 |

## File types

| Extension | Role |
|-----------|------|
| `.net.xml` | SUMO **road network**: edges, lanes, junctions, geometry, speed limits. |
| `.sumocfg` | SUMO **master config** pointing to the network + trip files and time settings (0–3600 s at 0.1 s steps). |
| `.trips.xml` | **Vehicle demand**: departure times and origin/destination edge pairs; SUMO routes each trip. |
| `next_lanes_*.pickle` | Pre-computed **downstream-lane map** (`lane_id → next lane_id`). Not a standard SUMO input — it lets the custom car-following search for a leader across SUMO's segmented lanes. |

> The `.trips.xml` files here are example demand for `trial0`; the notebook regenerates trips for new
> scenarios.
