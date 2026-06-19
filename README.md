# Next Generation Modeling (NGM) Project

The **NGM Project** is an end-to-end research framework for **calibrating and simulating
next-generation transportation behavior models**. It covers the full pipeline — from raw vehicle and
vulnerable-road-user (VRU) trajectory data, through behavioral-model calibration, to large-scale
microscopic traffic simulation in [Eclipse SUMO](https://eclipse.dev/sumo/) with connected/automated
vehicle (CAV/AV) and communication (V2X) behavior.

The repository is organized into two numbered stages that mirror this workflow:

| Stage | Folder | What it does |
|-------|--------|--------------|
| **1. Parametric input** | [`1 - PARAMETRIC INPUT`](1%20-%20PARAMETRIC%20INPUT/readme.md) | Calibrates behavioral models (car-following, lane-changing, lateral, VRU) against real trajectory data and produces parameter libraries. |
| **2. Simulation** | [`2 - SIMULATION`](2%20-%20SIMULATION/README.md) | Full desktop microsimulation tool (PyQt5 wizard) for running and comparing scenarios at scale. |


---

## Pipeline at a glance

```
   Raw trajectory data                Calibrated parameters             Microsimulation
 (Waymo, TGSIM, I-94/294/395)              (CSV libraries)              (SUMO + TraCI + GUI)

  1 - PARAMETRIC INPUT      ───────►   model_params / *.csv   ───────►   2 - SIMULATION
```

1. **Stage 1** fits behavioral models — IDM and Prospect Theory (PT) for car-following, MOBIL and the
   Drift Diffusion Model (DDM) for lane-changing, polynomial models for lateral motion, and Social Force
   (SF) / PT for pedestrians and bicycles — and exports per-class parameter CSVs.
2. **Stage 2** is the production simulator: a GUI-driven, scenario-configurable SUMO front-end that
   supports freeways, arterials, and intersections, plus CAV cooperation, V2X communication, signal
   control, and safety analysis.

---

## Repository structure

```
NGM_Test_v1/
├── README.md                     # This file
├── LICENSE
├── requirements.txt              # Python deps for GUI.py (SUMO installed separately)
├── ngm_paths.py                  # Shared dataset/output path helpers (repo root)
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
└── 2 - SIMULATION/               # Production microsimulation tool  →  see README.md
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

### [2 — Simulation](2%20-%20SIMULATION/README.md)

The production tool: a **PyQt5 wizard (`GUI.py`)** that configures networks, demand, signal timing,
driver models, and optional CAV/V2X behavior, then runs SUMO via **TraCI** with custom physics. It
supports **freeway**, **arterial**, **single-intersection**, and **TGSIM** scenarios, and ships with
analysis tools for trajectories, flow–density relationships, and surrogate safety metrics. See the
subfolder guides:

- [`models/`](2%20-%20SIMULATION/models/readme.md) — scenario runners, car-following/cooperation physics,
  V2X networking, and signal control.
- [`templates/`](2%20-%20SIMULATION/templates/readme.md) — SUMO network skeletons and their naming scheme.
- [`results/`](2%20-%20SIMULATION/results/readme.md) — post-processing notebooks, plotting, safety
  metrics, and batch outputs.

---

## Quick start: run the simulator (`GUI.py`)

This is the fastest path if you only want to **open the desktop wizard and run a SUMO
simulation**. You do **not** need to download Kaggle datasets or run Stage 1 calibration first.

| What you need | Required for GUI? |
|---------------|-------------------|
| Python 3.9+ and pip | Yes |
| Eclipse SUMO (`sumo`, `sumo-gui` on PATH) | Yes |
| Python packages (see below) | Yes |
| Trajectory datasets in `0 - Datasets/` | No (Stage 1 only) |
| Calibrated CSVs in `models/model_params/` | Shipped in repo (optional to refresh from Stage 1) |

### Step 1 — Clone the repository

```bash
git clone https://github.com/pebeigi/NGM_Test_v1.git
cd NGM_Test_v1
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Python packages

From the **repo root**:

```bash
pip install -r requirements.txt
```

This installs PyQt5, NumPy/Pandas/SciPy, Matplotlib, Numba, scikit-learn, Ray, and the Python
**TraCI** bindings (`traci`, `sumolib`).

### Step 4 — Install Eclipse SUMO

1. Download the installer for your OS from [eclipse.dev/sumo](https://eclipse.dev/sumo/).
2. Run the installer. On **Windows**, leave **“Add SUMO to PATH”** enabled if the installer offers it.
3. Confirm the binaries work in a **new** terminal (PATH updates apply to new shells):

```bash
sumo --version
sumo-gui --version
```

If those commands are not found, add SUMO’s `bin` folder to your system `PATH` manually (typical
Windows location: `C:\Program Files (x86)\Eclipse\Sumo\bin`).

### Step 5 — Verify imports

Still in your virtual environment:

```bash
python -c "import traci, sumolib, PyQt5; print('Dependencies OK')"
```

If `traci` fails, reinstall bindings that match your SUMO version:

```bash
pip install --upgrade traci sumolib
```

### Step 6 — Launch the GUI

```bash
cd "2 - SIMULATION"
python GUI.py
```

The PyQt5 wizard opens. Work through the pages (Geometry → Network → Volume → … → Simulation) and
click **Run** on the last page. See the [Stage 2 README](2%20-%20SIMULATION/README.md) for a
full walkthrough of every wizard page and scenario type.

**First run tip:** choose **Freeway** or **Single Intersection**, keep default volumes, enable
**visualization** if you want to watch SUMO-GUI, and leave **data collection** off for a quick smoke
test.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'traci'` | `pip install traci sumolib` (Step 3) |
| `ModuleNotFoundError: No module named 'PyQt5'` | Activate your venv, then `pip install -r requirements.txt` |
| `sumo` / `sumo-gui` not recognized | Install SUMO (Step 4) and reopen the terminal |
| GUI opens but simulation errors immediately | Confirm `sumo-gui --version` works; on Windows, keep visualization enabled only if SUMO-GUI is installed |
| Empty or missing `models/model_params/*.csv` | Re-clone or copy from Stage 1 calibration outputs. The GUI falls back to code defaults if a file is absent |

### Optional — use calibrated behavior parameters

The simulator samples driver-model parameters from CSV libraries under
[`2 - SIMULATION/models/model_params/`](2%20-%20SIMULATION/models/model_params/readme.md)
(`merged_IDM_*.csv`, `merged_PT_*.csv`, `MOBIL_results.csv`, …). Generate or copy these by running
[Stage 1 calibration](1%20-%20PARAMETRIC%20INPUT/readme.md), or download the pre-built parameter
bundles from the [NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets)
if available there.

---

## Getting started: Stage 1 calibration (optional)

Download all raw trajectory datasets from the
[NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) into
[`0 - Datasets/`](0%20-%20Datasets/readme.md) before running calibration scripts.

Stage 1 scripts and notebooks resolve paths through [`ngm_paths.py`](ngm_paths.py) at the repo root —
no machine-specific paths need to be edited after the data is in place.

---

## Notes on data

- Raw datasets are **not** stored in the repo (too large). Download all bundles from the
  [NGM Datasets Kaggle page](https://www.kaggle.com/datasets/pedrambeigi/ngm-datasets) and extract
  them into [`0 - Datasets/`](0%20-%20Datasets/readme.md). See [`0 - Datasets/readme.md`](0%20-%20Datasets/readme.md)
  for the expected filenames and layout. Simulator calibration CSVs under
  `2 - SIMULATION/models/model_params/` are versioned; other large CSV outputs remain git-ignored
  (see `.gitignore`).
- Primary datasets referenced throughout: **Waymo Open Motion** (pre-processed leader–follower and
  map CSVs), and **TGSIM** (Third Generation Simulation Data) for I-395, I-90/94, I-294, and
  the Foggy Bottom VRU trajectories.

## License

See [LICENSE](LICENSE).
