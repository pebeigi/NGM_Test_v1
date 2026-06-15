# I-90 / I-94 (Chicago)

SUMO assets for the **Chicago I-90 / I-94** corridor (Edens Expressway area, ~41.95°N, 87.73°W).

| File | Description |
|------|-------------|
| `I90_94.net.xml` | SUMO network (~29 edges) defining the motorway geometry, junctions, and lanes. The static road graph for this corridor. |
| `I90_94_trial0.sumocfg` | SUMO config loading the network + `I90_94_trial0.trips.xml`; simulates 0–3600 s at 0.1 s steps. |
| `I90_94_trial0.trips.xml` | ~3,800 example trips (`id`, `depart`, `from`/`to` edges, `departLane="random"`, sampled `departSpeed`) from the corridor's on-ramps to its exits. |
| `next_lanes_I90_94.pickle` | Downstream-lane map (~108 entries): each lane ID → its next lane ID (or `None` at chain ends), used by the custom car-following leader search. |
