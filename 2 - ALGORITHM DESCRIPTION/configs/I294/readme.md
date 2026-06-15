# I-294 (Chicago)

SUMO assets for the **Chicago I-294** corridor (Tri-State Tollway, ~41.81–41.87°N, 87.91–87.92°W).

| File | Description |
|------|-------------|
| `I294.net.xml` | SUMO network (~29 edges) defining the motorway geometry, junctions, and lanes. |
| `I294_trial0.sumocfg` | SUMO config loading the network + `I294_trial0.trips.xml`; simulates 0–3600 s at 0.1 s steps. |
| `I294_trial0.trips.xml` | ~4,400 example trips from four on-ramps to three off-ramps / mainline exits. |
| `next_lanes_I294.pickle` | Downstream-lane map (~98 entries) for the custom car-following leader search. |
