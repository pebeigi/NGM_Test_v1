# models

The simulation engine for [`3 - SIMULATION`](../README.md): the **scenario runners** dispatched by
`GUI.py`, the **custom physics and networking** they rely on, **signal control**, and the SUMO
artifacts generated at runtime.

## Scenario runners

Each runner builds a network from a [`templates/`](../templates/readme.md) skeleton, writes SUMO inputs,
and runs the TraCI loop with custom car-following, lane-changing, and (for connected classes)
cooperative behavior. They return a trajectory `DataFrame`.

| File | Scenario | Notes |
|------|----------|-------|
| `single_inter_sim.py` | Single intersection | `run_sim_single_inter`. Signals or all-way stop; supports pedestrians/bikes (ATM SF/PT). |
| `multi_inter_sim.py` | Arterial (3 signalized intersections) | `run_sim_multi_inter`. 8×8 OD matrix, Webster/manual timing, signal offsets. |
| `freeway_sim.py` | Freeway | `run_sim_freeway`. Procedural geometry by lane count and layout (`on_off`, `on_off_on_off`, `on_on_off_off`); weaving and mandatory-lane-change logic. Largest module. |
| `tgsim_sim.py` | TGSIM (fixed real network) | `run_sim_tgsim`. Currently the I-90/94 network used as-is (no procedural stretch). |

## Physics, networking, and control

| File | Role |
|------|------|
| `coop_models.py` | Numba-accelerated kernels: **IDM** (`idm_accel`), **C-IDM** cooperative car-following (`c_idm_accel`/`c_idm_kernel`) with platoon terms and virtual-vehicle merge intent, and **C-MOBIL** connected lane-change decisions (`c_mobil_decision`). |
| `ns.py` | Connected-vehicle networking: `VehicleState`, `VirtualVehicle` (cooperative intent on a target lane), and `CommunicationBus` (range, latency, packet loss, technology filter) for V2X. |
| `signal_control.py` | Signal timing for SUMO `tlLogic`: **Webster** optimal cycle/green splits from volumes, plus manual overrides; single-intersection and three-intersection arterial plans with offsets. |
| `paths.py` | Central path constants (`PROJECT_ROOT`, `MODELS_DIR`, `TEMPLATES_DIR`, `RESULTS_DIR`, `MODEL_PARAMS_DIR`) and MOBIL-CSV loading helpers used for GUI defaults. |
| `__init__.py` | Package marker. |

## Calibration parameters

See [`model_params/`](model_params/readme.md) for the IDM/PT/MOBIL/ATM parameter CSVs sampled at runtime.

## Generated SUMO artifacts (runtime outputs)

These are **written or overwritten when a simulation runs** — not hand-edited:

| File | Description |
|------|-------------|
| `freeway.net.xml` / `freeway.sumocfg` | Generated freeway network + config. |
| `single_intersection.net.xml` / `single_intersection.sumocfg` | Generated single-intersection network + config. |
| `multi_intersections.net.xml` / `multi_intersections.sumocfg` | Generated arterial network + config. |
| `i90_94.net.xml` / `i90_94.rou.xml` / `i90_94.sumocfg` | TGSIM I-90/94 network, routes, and config. |
| `all_trips.trips.xml` | Generated vehicle trips/routes for procedural scenarios. |
