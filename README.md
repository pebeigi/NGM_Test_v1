# Next Generation Modeling (NGM) Project

**NGM Project** is an end-to-end research framework for **calibrating, describing, and simulating
next-generation transportation behavior models**. It covers the full pipeline — from raw vehicle and
vulnerable-road-user (VRU) trajectory data, through behavioral-model calibration, to large-scale
microscopic traffic simulation in [Eclipse SUMO](https://eclipse.dev/sumo/) with connected/automated
vehicle (CAV/AV) and communication (V2X) behavior.

The repository is organized into three numbered stages that mirror this workflow:

| Stage | Folder | What it does |
|-------|--------|--------------|
| **1. Parametric input** | [`1 - PARAMETRIC INPUT`](1%20-%20PARAMETRIC%20INPUT/readme.md) | Calibrates behavioral models (car-following, lane-changing, lateral, VRU) against real trajectory data and produces parameter libraries. |
| **2. Algorithm description** | [`2 - ALGORITHM DESCRIPTION`](2%20-%20ALGORITHM%20DESCRIPTION/readme.md) | Reference SUMO + TraCI implementation of the calibrated models on three real freeway corridors. |
| **3. Simulation** | [`3 - SIMULATION`](3%20-%20SIMULATION/README.md) | Full desktop microsimulation tool (PyQt5 wizard) for running and comparing scenarios at scale. |

> **Data convention used across the project:** vehicle classes are abbreviated **S** = Small vehicle
> (human-driven car), **A** = Automated vehicle (AV/CAV), **L** = Large/heavy vehicle (truck).
> Some simulation parameter files add **CL** = Connected heavy vehicle (CAHV).

---

## Pipeline at a glance

```
   Raw trajectory data                Calibrated parameters             Microsimulation
 (Waymo, TGSIM, I-94/294/395)              (CSV libraries)              (SUMO + TraCI + GUI)

  1 - PARAMETRIC INPUT      ───────►   model_params / *.csv   ───────►   3 - SIMULATION
        │                                      ▲                              │
        └──────────────────────────────────────┴──────────────────────────────┘
                          2 - ALGORITHM DESCRIPTION
                  (reference freeway implementation of the models)
```

1. **Stage 1** fits behavioral models — IDM and Prospect Theory (PT) for car-following, MOBIL and the
   Drift Diffusion Model (DDM) for lane-changing, polynomial models for lateral motion, and Social Force
   (SF) / PT for pedestrians and bicycles — and exports per-class parameter CSVs.
2. **Stage 2** is a self-contained notebook that loads those parameters and drives custom car-following
   models inside SUMO for the I-90/94, I-294, and I-395 corridors.
3. **Stage 3** is the production simulator: a GUI-driven, scenario-configurable SUMO front-end that
   supports freeways, arterials, and intersections, plus CAV cooperation, V2X communication, signal
   control, and safety analysis.

---

## Repository structure

```
NGM_Test_v1/
├── README.md                     # This file
├── LICENSE
├── ngm_paths.py                  # Shared dataset/output path helpers (repo root)
├── main.py                       # Placeholder entry point
├── .gitignore                    # Ignores raw datasets and large generated CSVs/plots
│
├── 0 - Datasets/                 # Raw datasets — download links & layout (see readme)
│
├── 1 - PARAMETRIC INPUT/         # Model calibration  →  see readme.md
│   ├── 1.1 - Car-Following/      #   IDM / PT calibration (Waymo + TGSIM)
│   ├── 1.2 - Lane Changing/      #   MOBIL + DDM calibration
│   ├── 1.3 - Lateral Calibration/#   Curvilinear transform + lateral LC shape
│   └── 1.4.2 - VRU Parametric Analysis/  # Social Force / PT VRU calibration
│
├── 2 - ALGORITHM DESCRIPTION/    # Reference SUMO freeway implementation  →  see readme.md
│   ├── Simulate_Freeway.ipynb
│   ├── configs/                  #   SUMO networks for I-90/94, I-294, I-395
│   └── model_params/             #   Calibrated IDM/PT parameter pools
│
└── 3 - SIMULATION/               # Production microsimulation tool  →  see README.md
    ├── GUI.py                    #   PyQt5 configuration wizard (main entry point)
    ├── models/                   #   Scenario runners, physics, networking, signals
    ├── templates/                #   SUMO network skeletons
    └── results/                  #   Analysis notebooks, scripts, and outputs
```

---

## Stage summaries

### [1 — Parametric Input](1%20-%20PARAMETRIC%20INPUT/readme.md)

Data preprocessing and behavioral-model calibration across four domains:

- **[Car-following](1%20-%20PARAMETRIC%20INPUT/1.1%20-%20Car-Following/readme.md)** — IDM and Prospect
  Theory models fit to Waymo Open Motion and TGSIM trajectories with genetic algorithms.
- **[Lane changing](1%20-%20PARAMETRIC%20INPUT/1.2%20-%20Lane%20Changing/readme.md)** — MOBIL politeness
  calibration and a Drift Diffusion Model for discrete lane-change decisions.
- **[Lateral calibration](1%20-%20PARAMETRIC%20INPUT/1.3%20-%20Lateral%20Calibration/readme.md)** —
  curvilinear coordinate transforms and polynomial models of lateral lane-change shape.
- **[VRU parametric analysis](1%20-%20PARAMETRIC%20INPUT/1.4.2%20-%20VRU%20Parametric%20Analysis/readme.md)** —
  Social Force and Prospect Theory models for pedestrians/bicycles.

### [2 — Algorithm Description](2%20-%20ALGORITHM%20DESCRIPTION/readme.md)

A reference implementation that takes the calibrated parameters into SUMO. `Simulate_Freeway.ipynb`
drives **custom IDM/PT car-following** through TraCI on three real-world freeway corridors
(**I-90/94**, **I-294**, **I-395**), with SUMO network/trip configs under
[`configs/`](2%20-%20ALGORITHM%20DESCRIPTION/configs/readme.md) and calibrated parameter pools under
[`model_params/`](2%20-%20ALGORITHM%20DESCRIPTION/model_params/readme.md).

### [3 — Simulation](3%20-%20SIMULATION/README.md)

The production tool: a **PyQt5 wizard (`GUI.py`)** that configures networks, demand, signal timing,
driver models, and optional CAV/V2X behavior, then runs SUMO via **TraCI** with custom physics. It
supports **freeway**, **arterial**, **single-intersection**, and **TGSIM** scenarios, and ships with
analysis tools for trajectories, flow–density relationships, and surrogate safety metrics. See the
subfolder guides:

- [`models/`](3%20-%20SIMULATION/models/readme.md) — scenario runners, car-following/cooperation physics,
  V2X networking, and signal control.
- [`templates/`](3%20-%20SIMULATION/templates/readme.md) — SUMO network skeletons and their naming scheme.
- [`results/`](3%20-%20SIMULATION/results/readme.md) — post-processing notebooks, plotting, safety
  metrics, and batch outputs.

---

## Getting started

Download all raw datasets from the
[NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) into
[`0 - Datasets/`](0%20-%20Datasets/readme.md) before running calibration scripts.

**Stage 1** scripts and notebooks resolve dataset paths through [`ngm_paths.py`](ngm_paths.py) at the
repo root — no machine-specific paths need to be edited after the data is in place. **Stage 2** uses
repo-relative paths under `2 - ALGORITHM DESCRIPTION/` (e.g. `model_params/`, `configs/`). **Stage 3**
(the simulator) is the most turn-key component.

To run the simulator:

```bash
# 1. Install SUMO and add its bin/ to PATH (see Eclipse SUMO docs)
# 2. Install Python dependencies
pip install PyQt5 numpy pandas scipy matplotlib numba scikit-learn ray

# 3. Launch the configuration wizard
cd "3 - SIMULATION"
python GUI.py
```

See the [Stage 3 README](3%20-%20SIMULATION/README.md) for full installation, usage, and output details.

---

## Notes on data

- Raw datasets are **not** stored in the repo (too large). Download all bundles from the
  [NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) and extract
  them into [`0 - Datasets/`](0%20-%20Datasets/readme.md). See [`0 - Datasets/readme.md`](0%20-%20Datasets/readme.md)
  for the expected filenames and layout. Calibrated parameter CSVs (what the simulator needs) are
  versioned. Bulk simulation outputs remain git-ignored (see `.gitignore`).
- Primary datasets referenced throughout: **Waymo Open Motion** (pre-processed leader–follower and
  map CSVs), and **TGSIM** (Third Generation Simulation Data) for I-395, I-90/94, I-294, and
  the Foggy Bottom VRU trajectories.

## License

See [LICENSE](LICENSE).
