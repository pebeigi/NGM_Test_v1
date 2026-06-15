# I-395 (Washington, D.C.)

SUMO assets for the **Washington, D.C. I-395** corridor (~38.88°N, 77.02°W).

| File | Description |
|------|-------------|
| `I395.net.xml` | SUMO network (~15 edges) defining the corridor geometry, junctions, and lanes. |
| `I395_trial0.sumocfg` | SUMO config loading the network + `I395_trial0.trips.xml`; includes a `lanechange.duration` setting. |
| `I395_trial0.trips.xml` | ~2,400 example trips between this corridor's ramps and exits. |
| `next_lanes_I395.pickle` | Downstream-lane map (~41 entries) for the custom car-following leader search. |
| `I90_94_trial0.trips.xml` | **Misplaced duplicate** — byte-identical to the I-90/94 trips file and not referenced by `I395_trial0.sumocfg`. Safe to remove (see the root cleanup notes). |
