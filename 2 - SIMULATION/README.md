# NGM Simulation

**NGM Simulation** is a traffic microsimulation tool. It wraps **[Eclipse SUMO](https://eclipse.dev/sumo/)** with a **PyQt5** configuration wizard and **Python** control logic. While the simulation runs, **TraCI** connects to SUMO so the code can inject custom **car-following**, **lane-changing**, **cooperative (CAV)**, and **signal-timing** behavior beyond default SUMO driver models.

This document explains what the software does, how the user interface is organized, what each code module is for, and how to run it.

> **Subfolder guides:** [`models/`](models/readme.md) (engine, physics, networking, signals) ·
> [`models/model_params/`](models/model_params/readme.md) (calibration CSVs) ·
> [`templates/`](templates/readme.md) (network skeletons) ·
> [`results/`](results/readme.md) (analysis, plots, batch outputs).

---

## Table of contents

1. [Concept and architecture](#concept-and-architecture)
2. [What you can simulate](#what-you-can-simulate)
3. [The configuration wizard (step by step)](#the-configuration-wizard-step-by-step)
4. [Driving and cooperation models](#driving-and-cooperation-models)
5. [Active travel (pedestrians and bikes)](#active-travel-pedestrians-and-bikes)
6. [Repository layout and generated files](#repository-layout-and-generated-files)
7. [Requirements and installation](#requirements-and-installation)
8. [How to run](#how-to-run)
9. [Output data](#output-data)

---

## Concept and architecture

| Layer | Technology | Role |
|-------|------------|------|
| **Microsimulation engine** | SUMO | Vehicle movement on lanes, intersections, traffic lights, default routing. |
| **Runtime control** | TraCI (`traci`) | Python reads/writes vehicle state, speeds, routes, and signals each time step. |
| **Network build** | `sumolib`, XML templates | Networks and trip files are generated from `templates/*.net.xml` and user inputs. |
| **User interface** | PyQt5 (`GUI.py`) | Multi-page wizard collecting all parameters into a single `responses` object. |
| **Custom physics / behavior** | `models/coop_models.py`, `models/ns.py`, etc. | IDM/C-IDM accelerations, V2X-style communication, virtual vehicles for cooperation. |
| **Signal timing** | `models/signal_control.py` | Webster-based cycle and green times; writes SUMO-compatible phase definitions. |

The main entry point is **`GUI.py`**. When you finish the wizard, **`run_simulation`** dispatches to one of:

- `models/single_inter_sim.py` → **Single Intersection**
- `models/multi_inter_sim.py` → **Arterial**
- `models/freeway_sim.py` → **Freeway**
- `models/tgsim_sim.py` → **TGSIM** (currently I-90/94 - for Validation Purposes)

If **visualization** is enabled, **SUMO-GUI** may need to run on the **main thread** (notably on Windows); otherwise the heavy work can run in a **background thread** so the window stays responsive.

---

## What you can simulate

### Scenarios (geometry types)

1. **Freeway**  
   - Configurable **number of mainline lanes** (1–6) and **ramp length**.  
   - **Freeway layout type** (how ramps are sequenced along the corridor):
     - `on_off` — one on-ramp and one off-ramp style segment (four OD flows: Main–Main, OnRamp–Main, Main–OffRamp, OnRamp–OffRamp).
     - `on_on_off_off` and `on_off_on_off` — **longer** layouts with **multiple weaving sections** and more origin–destination pairs (volumes per route are entered separately).  
   - Additional **segment lengths** (input-to-weaving, weaving lengths, tapers, etc.) depend on the selected layout; labels in the GUI match the order used when building the network.

2. **Arterial (multi-intersection)**  
   - **East–west arterial** with **three signalized intersections** (West, Central, East in the logic).  
   - **North–south and east–west lane counts** (2 or 3 lanes per direction, from templates).  
   - **Spacing** between the west–central and central–east intersections (meters).  
   - **Demand** is an **8×8 origin–destination matrix** among nodes: West, North West, South West, North, South, North East, South East, East. You can fill the matrix manually or **import a CSV** (8×8 numeric block with the expected layout).  
   - **Per-intersection signal timing** and **offsets** along the main road (progression between Int1 → Int2 → Int3).

3. **Single intersection**  
   - One junction.  
   - **Control type**: **traffic signals** or **all-way stop**.  
   - **Lanes**: NS and EW counts depend on control — **signals** allow 2–3 lanes each way; **all-way stop** allows 1–2 lanes each way.  
   - **Approach length** from each leg to the junction (meters).  
   - **Pedestrian walkway width** (meters) for sidewalks when pedestrians are modeled.  
   - **Per-approach volumes** (veh/h) for East-Bound, West-Bound, North-Bound, South-Bound, each with **left-turn** and **right-turn** sliders (percentages are coupled so left + right ≤ 100%).

4. **TGSIM**  
   - Fixed **real-world freeway network** (currently **I-90/94** only) loaded from `templates/I90_94_simple.net.xml` — no procedural geometry editing.  
   - **Volume** is set per named route on the corridor (mainline and ramp flows as defined in `tgsim_sim.py`).  
   - Skips signal control and active-travel (pedestrian/bike) pages; goes straight to car-following and lane-changing models like the procedural freeway workflow.

### Vehicle mix

On the **Volume** page, **five sliders** must sum to **exactly 100%**:

| Class | Typical meaning in this project |
|-------|----------------------------------|
| **SV** — Small vehicle | Human-driven passenger cars. |
| **AV** — Automated vehicle | Automated cars (distinct color in SUMO-GUI). |
| **HV** — Heavy vehicle | Trucks / heavy vehicles. |
| **CAHV** — Connected heavy vehicle | Connected heavy vehicles. |
| **CAV** — Connected & autonomous | Uses cooperative car-following / lane-changing when implemented in the simulation loop. |

The GUI shows a **color legend** that matches the **`COLOR_MAP`** used in the simulation code so vehicles are easy to identify in SUMO-GUI.

---

## The configuration wizard (step by step)

The wizard progress bar lists: **Geometry → Network → Volume → Signal Control → ATM Demand → Models → Simulation**. Not every page appears for every scenario (see branches below).

### 1. Welcome

Title: **“Welcome to NGM Simulation!”** — entry point; **Start** goes to geometry selection.

### 2. Geometry (scenario selection)

You choose exactly one:

- **Freeway**
- **Arterial** (labeled “Arterial” in the UI — the multi-intersection corridor)
- **Single Intersection**
- **TGSIM** (fixed I-90/94 network; sub-selector under the radio button)

This sets `Scenario` in the internal configuration (`"Freeway"`, `"Arterial"`, `"Single Intersection"`, or `"TGSIM"`).

### 3. Network configuration

Scenario-specific geometry is stored under `Geometry` in the responses object.

- **Freeway**: lanes, ramp length, freeway type, then all segment-length fields for the selected layout.  
- **TGSIM**: read-only summary of the selected network (geometry is fixed by the template).  
- **Single intersection**: intersection control (signal vs all-way stop), NS/EW lanes, road length, walkway width.  
- **Arterial**: NS/EW lanes, west–central distance, central–east distance.

### 4. Volume configuration

- **Vehicle mix** (must total 100%) as above.  
- **Freeway**: volumes per **named route** (depends on `Freeway_Type`).  
- **TGSIM**: volumes per **named route** on the I-90/94 network.  
- **Single intersection**: per-boundary volume + left/right turn ratios.  
- **Arterial**: OD matrix **manual** or **CSV upload** (validation requires a 9×9 table with an 8×8 numeric interior).

Saved as `Vehicle_Flows` (including `SV_rate`, `AV_rate`, …).

### 5. Signal control (conditional)

- Shown for **Single Intersection** only if **Signal** control was chosen (not for all-way stop).  
- Shown for **Arterial** (three intersections).

Options:

- **Webster’s formula** — cycle length and effective greens estimated from volumes (defaults shown in the UI).  
- **Manual** — you edit cycle length, green for EW, green for NS (single junction) or **three separate plans** (arterial).  

For **arterial**, you can also set **offsets** between intersections along the east–west mainline (seconds) to model signal progression.

Internal structure: `Signal_Control` with `use_webster` or manual fields; arterial adds `manual_plans`, `offset_1_2`, `offset_2_3`.

### 6. ATM demand (conditional)

**ATM** here means **Active Traffic Mode** demand — **pedestrians and bicycles**.

This page appears for **Single Intersection** after volume (and after signal control if the intersection is signalized). You can:

- Enable **pedestrians** and set **pedestrians per hour** (randomized O–D in the UI description).  
- Enable **bikes** and set **bikes per hour**.

Saved as `Allow_Ped`, `Allow_Bike`, `Ped_Volume`, `Bike_Volume`.

**Freeway**, **TGSIM**, and **Arterial** workflows **skip** this page and go straight to **car-following** models.

### 7. Driving models — Car-following (CF)

Choose **IDM** (Intelligent Driver Model) or **Prospect Theory (PT)** for longitudinal behavior.

- For each of **Small Vehicle**, **Automated Vehicle**, **Heavy Vehicle**, you set **Mean** and **Std** for every parameter row (or use **Default Parameters** loaded from calibration CSVs under `models/model_params/`).  
- **IDM parameters**: `T`, `a`, `b`, `v_0`, `s_0`.  
- **PT parameters**: `T_max`, `α`, `β`, `W_c`, `W_m`, `Gamma1`, `Gamma2`.

Defaults for IDM/PT are computed from:

- `merged_IDM_S.csv`, `merged_IDM_A.csv`, `merged_IDM_L.csv`  
- `merged_PT_S.csv`, `merged_PT_A.csv`, `merged_PT_L.csv`

**Connected / automated extras**

- **CAV (C-IDM)**: optional default or manual **K_v**, **K_a**, **s_ref** (cooperative ACC-style gains).  
- **V2X-style communication**: optional default or manual **Range**, **Lookahead**, **Latency**, **Loss** — used with the cooperative logic in `ns.py` / `coop_models.py`.

Stored as `CF_Model`, `CF_Default_Params`, `CF_Parameters`, `CIDM_Params`, `Comm_Params`.

### 8. Driving models — Lane changing (LC)

Choose **MOBIL** or **Drift Diffusion Model (DDM)**.

- **MOBIL**: discretionary politeness and threshold (`Disc: p_opt`, `Disc: a_th`); safety braking limits (`Disc: b_safe`, `Mand: b_safe`). With default parameters checked, each vehicle draws a full row from `MOBIL_results.csv` (`Discretionary_*` and `Mandatory_b_safe`). Uncheck defaults to set Mean/Std per class in the GUI.  
- **DDM**: eight parameters with fixed default means if using defaults.

Per vehicle class: Mean/Std columns again.

**CAV (C-MOBIL)**: **kappa** (intent urgency) and **gamma** (lane-change safety time), with a default/manual toggle.

Stored as `LC_Model`, `LC_Default_Params`, `LC_Parameters`, `CMOBIL_Params`.

### 9. ATM models (conditional)

Only if **Single Intersection** **and** (**pedestrians or bikes** enabled). You choose:

- **Social Force (SF)** — pedestrian social-force style parameters and bike dynamics parameters (see `SF_atm_params.csv` mapping in code).  
- **Prospect Theory (PT)** — alternative parameter set for active modes (`PT_atm_params.csv`).

Stored as `ATM_Model`, `ATM_Default_Params`, `ATM_Parameters`.

### 10. Simulation settings

- **Step size (s)**: 0.1, 0.2, 0.5, or 1.0 — SUMO simulation step.  
- **Simulation time (s)**: total duration (wide range, e.g. up to hours).  
- **Enable visualization** — launch SUMO-GUI when supported.  
- **Enable data collection** — if on, you must **browse** to an **output folder**.  
- **Data collection frequency (s)** — how often rows are kept in the exported CSV (independent of step size; the runner downsamples).

Keys: `Sim_StepSize`, `Sim_Time`, `Sim_Visualization`, `Sim_DataCollection`, `Sim_DataFreq`, `Data_Folder`.

### 11. Simulation running

Progress bar and **Back** / **Close**. Simulation starts when this page is shown. On success, the app may close after a short delay; if SUMO is closed early, you can go **Back** and change settings.

---

## Driving and cooperation models

| Piece | File(s) | Purpose |
|-------|---------|---------|
| **IDM / C-IDM** | `models/coop_models.py` | Longitudinal acceleration; C-IDM adds platoon-style terms using gaps and communicated speeds/accelerations. |
| **Vehicle state & V2X** | `models/ns.py` | `VehicleState`, `CommunicationBus` (range, latency, packet loss), `VirtualVehicle` for cooperative gaps. |
| **Signals** | `models/signal_control.py` | Webster saturation flow and lost time; builds phase durations for SUMO `tlLogic`. |

The GUI separates **human-like** models (IDM, MOBIL, …) from **CAV-specific** knobs (C-IDM, C-MOBIL, communication), so you can study mixed traffic.

---

## Active travel (pedestrians and bikes)

- **Demand** is set on **ATM Demand** (single-intersection scenarios only in the current wizard).  
- **Behavior** is set on **ATM Models** if either mode is enabled: **SF** vs **PT**, with **Mean/Std** per parameter (defaults from `SF_atm_params.csv` / `PT_atm_params.csv` when available).

Pedestrians are described in the UI as having **randomized origins and destinations** within the scenario logic.

---

## Repository layout and generated files

| Path | Description |
|------|-------------|
| **`GUI.py`** | Full application: pages, `SimulationThread`, `run_simulation`, global `Responses` store. |
| **`models/single_inter_sim.py`** | Builds `single_intersection.net.xml`, routes, runs TraCI loop for one junction. |
| **`models/multi_inter_sim.py`** | Three-intersection arterial; offsets and per-intersection plans. |
| **`models/freeway_sim.py`** | Freeway nets and OD/route flows by layout type. |
| **`models/tgsim_sim.py`** | TGSIM I-90/94 network, routes, and TraCI loop. |
| **`models/signal_control.py`** | Webster and manual timing integration. |
| **`models/coop_models.py`** | Numba-accelerated IDM/C-IDM kernels. |
| **`models/ns.py`** | Networking abstractions for connected vehicles. |
| **`models/paths.py`** | `PROJECT_ROOT`, `MODELS_DIR`, `TEMPLATES_DIR`, `RESULTS_DIR`, `MODEL_PARAMS_DIR`. |
| **`templates/`** | Base `.net.xml` files; lane counts and topology variants (freeway and intersection templates). |
| **`models/model_params/`** | Calibration CSVs for default Means/Stds (IDM, PT, MOBIL, SF/PT ATM). Missing files fall back to zeros or hard-coded defaults in code. |

During a run, generated SUMO inputs (network, configuration, trips) are written under **`models/`** (see each `*_sim.py` for exact filenames such as `single_intersection.sumocfg`, `all_trips.trips.xml`).

---

## Requirements and installation

1. **SUMO** — Install from [Eclipse SUMO](https://eclipse.dev/sumo/). Add the installation’s `bin` directory to your **PATH** so `sumo` and `sumo-gui` work from a terminal.

2. **Python 3** with at least:

   ```text
   PyQt5
   numpy
   pandas
   scipy
   matplotlib
   numba
   scikit-learn
   ray
   ```

3. **TraCI / sumolib** — Usually shipped with SUMO under the `tools` folder. Add that folder to **`PYTHONPATH`**, or use a compatible **`pip`** install; see the [official TraCI Python guide](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html).

Example (from the repo root; use a virtual environment if you prefer):

```bash
pip install -r requirements.txt
```

See also the [root README quick start](../README.md#quick-start-run-the-simulator-guipy).

---

## How to run

From the **`2 - SIMULATION`** folder (the directory that contains `GUI.py`):

```bash
cd "2 - SIMULATION"
python GUI.py
```

For a full install guide (Python, SUMO, TraCI, virtual environment), start with the
[repository root README](../README.md#quick-start-run-the-simulator-guipy).

Complete the wizard. If you enable **data collection**, choose an **output directory** before
**Next** on the simulation settings page (the app warns if the folder is empty).

---

## Output data

When **data collection** is enabled, the runner writes **`test_sim.csv`** into the folder you selected (`Data_Folder`).  

The internal time series is **downsampled**: only steps aligned to **`Sim_DataFreq`** relative to **`Sim_StepSize`** are kept (see `run_simulation` in `GUI.py` for the exact indexing).  

The CSV columns depend on what each scenario script returns in the pandas `DataFrame` (typically time-varying quantities collected each step before downsampling).
