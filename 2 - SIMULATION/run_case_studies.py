"""
Batch runner for Case Study 2 (and helpers for CS1 / CS3).

Workflow
--------
1. Open the GUI and start one simulation per environment / freeway geometry you
   want to automate. Each run writes:
        results/last_gui_config.pkl                   (most recent, any scenario)
        results/baseline_<scenario>.pkl               (e.g. baseline_freeway.pkl)
        results/baseline_freeway_<freeway_type>.pkl   (Freeway only, per geometry)

   For the case studies in this project you need:
        results/baseline_freeway_on_off.pkl           (used by CS1)
        results/baseline_freeway_on_off_on_off.pkl    (used by CS2 freeway cases)
        results/baseline_arterial.pkl                 (used by CS2 arterial cases)
        results/baseline_single_intersection.pkl      (used by CS3)
2. Run this script from the `2 - SIMULATION` folder:

        python run_case_studies.py                # runs every CS1 + CS2 + CS3 scenario
        python run_case_studies.py --cs2          # CS2 scenarios only (MPR freeway + arterial)
        python run_case_studies.py --cs1          # run CS1 freeway V2V-quality set
        python run_case_studies.py --cs3          # run CS3 single intersection (50/50 mixes)
        python run_case_studies.py --only CS2_Fwy_80CAV_20CAHV
        python run_case_studies.py --visualize    # open SUMO-GUI for every run

The script loads the appropriate baseline pickle for each scenario, applies
per-scenario overrides (vehicle mix, V2V comm params, etc.), then enforces the
fleet driving-model policy documented below, calls the matching
`run_sim_*` function directly and writes the resulting trajectory DataFrame to
`results/case_runs/<scenario_label>/run_##.csv` (see `--repeats`; default 5).
The registry `output` keys below are legacy flat filenames (notebook mapping),
not paths written by this batch runner:

        Legacy notebook IDs (sequential within each case study, match registry order):
        CS1 (10–16): HDV baseline (100% SV), Ideal, HighLoss, MidLatency Low/Mid loss, HighLatency, HighLatency+Loss
        CS2 freeway (21–29): 100SV … 80CAV_20CAHV, 90SV_10CAHV, 90CAV_10HV
        CS2 arterial (30–33): 90SV_10HV … 80CAV_20CAHV
        CS3 (34–35): 50SV_50CAV, 50SV_50AV

Missing baselines or templates cause the affected scenarios to be skipped with
a clear message — the rest of the queue still runs.

Driving models (GUI keys CF_Model / LC_Model):
    * Human-driven fleet (SUMO tech SV, HV, AV — not CAV/CAHV) uses PT
      car-following and DDM lane-changing with default calibrated parameters.
    * CAV / CAHV use C-IDM + C-MOBIL internally; CIDM_Params / CMOBIL_Params from
      the baseline apply when saved from the GUI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# Make sibling packages importable when running from the project folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.freeway_sim import run_sim_freeway          # noqa: E402
from models.multi_inter_sim import run_sim_multi_inter  # noqa: E402
from models.single_inter_sim import run_sim_single_inter  # noqa: E402


# ---------------------------------------------------------------------------
# Paths & baseline handling
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Repeated batch outputs: ``results/case_runs/<scenario_label>/run_01.csv`` …
CASE_RUNS_SUBDIR = "case_runs"
MANIFEST_FILENAME = "batch_manifest.csv"
# Global simulation duration (seconds) for every case/scenario in this batch.
# Change this one value to avoid editing GUI baselines for run length.
DEFAULT_SIM_TIME_S = 1200
DEFAULT_REPEATS = 3

BASELINE_ARTERIAL   = os.path.join(RESULTS_DIR, "baseline_arterial.pkl")
BASELINE_SINGLE     = os.path.join(RESULTS_DIR, "baseline_single_intersection.pkl")
BASELINE_FREEWAY    = os.path.join(RESULTS_DIR, "baseline_freeway.pkl")  # generic fallback
BASELINE_FALLBACK   = os.path.join(RESULTS_DIR, "last_gui_config.pkl")


def _load_pickle(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as ex:
        print(f"  [baseline] failed to load {path}: {ex}")
        return None


def _freeway_baseline_path(freeway_type: str) -> str:
    return os.path.join(RESULTS_DIR, f"baseline_freeway_{freeway_type.lower()}.pkl")


def load_baseline(scenario: str, *, freeway_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return a fresh deep-copy of the baseline dict matching `scenario`, or None.

    For `scenario='Freeway'`, `freeway_type` selects between the three templates
    (on_off / on_off_on_off / on_on_off_off). Each freeway_type has its own
    geometry + Vehicle_Flows key structure, so they cannot be interchanged.

    Lookup order (first hit wins, deep-copied):
        1. baseline_freeway_<freeway_type>.pkl          (exact match)
        2. baseline_freeway.pkl  — only if its Freeway_Type matches
        3. last_gui_config.pkl   — only if Scenario + Freeway_Type both match

    For Arterial / Single Intersection the regular baselines are used.
    """
    if scenario == "Freeway":
        if freeway_type:
            cfg = _load_pickle(_freeway_baseline_path(freeway_type))
            if cfg is not None:
                return copy.deepcopy(cfg)

        for path in (BASELINE_FREEWAY, BASELINE_FALLBACK):
            cfg = _load_pickle(path)
            if cfg is None:
                continue
            if str(cfg.get("Scenario", "")).strip() != "Freeway":
                continue
            if freeway_type and str(cfg.get("Geometry", {}).get("Freeway_Type", "")).strip().lower() != freeway_type.lower():
                continue
            print(f"  [baseline] using {path} as Freeway/{freeway_type} baseline")
            return copy.deepcopy(cfg)
        return None

    wanted = {
        "Arterial":            BASELINE_ARTERIAL,
        "Single Intersection": BASELINE_SINGLE,
    }.get(scenario)
    cfg = _load_pickle(wanted) if wanted else None
    if cfg is not None:
        return copy.deepcopy(cfg)

    fallback = _load_pickle(BASELINE_FALLBACK)
    if fallback is not None and str(fallback.get("Scenario", "")).strip() == scenario:
        print(f"  [baseline] using {BASELINE_FALLBACK} as {scenario} baseline")
        return copy.deepcopy(fallback)
    return None


# ---------------------------------------------------------------------------
# Progress adapter (mirrors what GUI.py passes to the sim functions)
# ---------------------------------------------------------------------------

class _Progress:
    """Tiny stand-in for Qt's pyqtSignal; prints percentages to stdout."""
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._last = -1

    def emit(self, pct: int):
        pct = int(pct)
        if pct == self._last:
            return
        self._last = pct
        if pct % 10 == 0 or pct == 100:
            print(f"    {self.prefix}progress: {pct}%", flush=True)


def _is_running_always():
    return True


# ---------------------------------------------------------------------------
# Common helpers for applying overrides
# ---------------------------------------------------------------------------

def set_vehicle_mix(cfg: Dict[str, Any], *, sv=0.0, hv=0.0, av=0.0, cav=0.0, cahv=0.0) -> None:
    """Set the SV / HV / AV / CAV / CAHV rates (fractions in [0, 1])."""
    total = sv + hv + av + cav + cahv
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Vehicle mix must sum to 1.0, got {total}")
    vf = cfg.setdefault("Vehicle_Flows", {})
    vf["SV_rate"]   = float(sv)
    vf["HV_rate"]   = float(hv)
    vf["AV_rate"]   = float(av)
    vf["CAV_rate"]  = float(cav)
    vf["CAHV_rate"] = float(cahv)


def set_comm_params(cfg: Dict[str, Any], *, latency_s: float, packet_loss: float,
                    comm_range_m: float = 30.0, max_lookahead: int = 3) -> None:
    """Inject V2V comm reliability parameters used by CAV controllers.

    These keys are read by the cooperative CF / LC code. Unknown keys are simply
    ignored by the sim, so setting them is safe even on older baselines.
    """
    cfg["V2V_Latency_s"]    = float(latency_s)
    cfg["V2V_PacketLoss"]   = float(packet_loss)       # 0.0 – 1.0
    cfg["V2V_Range_m"]      = float(comm_range_m)
    cfg["V2V_Max_Lookahead"] = int(max_lookahead)


def case_runs_dir(label: str) -> str:
    root = os.path.join(RESULTS_DIR, CASE_RUNS_SUBDIR)
    return os.path.join(root, label)


def _scenario_run_seed(base_seed: int, label: str, run_index: int) -> int:
    """Deterministic 32-bit seed from base, scenario label, and run index."""
    ih = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")
    combined = int(base_seed) + ih + int(run_index) * 97783
    return combined % (2**31)


def apply_stochastic_seed(seed: int) -> None:
    """Reset Python ``random`` and NumPy RNGs before one simulation replicate."""
    import random

    import numpy as np

    s = int(seed) % (2**31)
    random.seed(s)
    np.random.seed(s)


def _append_batch_manifest(case_runs_root: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = os.path.join(case_runs_root, MANIFEST_FILENAME)
    new_file = not os.path.exists(path)
    fieldnames = [
        "ts_utc",
        "scenario_label",
        "case_study",
        "gui_scenario",
        "freeway_type",
        "legacy_output",
        "run_index",
        "seed",
        "csv_path",
        "n_rows",
        "wall_s",
    ]
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def apply_case_study_driving_models(cfg: Dict[str, Any]) -> None:
    """Set GUI-equivalent longitudinal / lateral model choice for human-driven traffic.

    Non-connected vehicles follow ``CF_Model`` and ``LC_Model``. Connected CAV /
    CAHV ignore these for dynamics and instead use C-IDM + C-MOBIL inside the
    simulation loop. Default-parameter toggles mirror checking "Default parameters"
    on CF / LC wizard pages so sampling uses ``models/model_params`` CSVs (PT/DDM
    draws and built-in DDM fallbacks).
    """
    cfg["CF_Model"] = "PT"
    cfg["LC_Model"] = "DDM"
    cfg["CF_Default_Params"] = True
    cfg["LC_Default_Params"] = True


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

# Each scenario is a small object:
#   case:     "CS1" | "CS2" | "CS3"
#   scenario: the GUI "Scenario" field required ("Freeway" / "Arterial" / "Single Intersection")
#   output:   legacy flat filename stem (e.g. 11.csv) for documentation / manifests
#   apply:    function (cfg) -> None that mutates the baseline cfg in-place
#   runner:   which run_sim_* to call (looked up by Scenario automatically)


# Freeway geometry types per case study:
#   CS1 -> "on_off"            (baseline_freeway_on_off.pkl)
#   CS2 -> "on_off_on_off"     (baseline_freeway_on_off_on_off.pkl)
CS1_FWY_TYPE = "on_off"
CS2_FWY_TYPE = "on_off_on_off"


def _cs1(name_bits):
    """Factory for CS1 scenarios: freeway, 100% CAV, only V2V comm params change."""
    def _apply(cfg):
        set_vehicle_mix(cfg, cav=1.0)
        set_comm_params(cfg, **name_bits)
    return _apply


def _cs1_hdv_baseline(cfg):
    """100% SV human-driven fleet; PT + DDM applied in apply_case_study_driving_models()."""
    set_vehicle_mix(cfg, sv=1.0)
    set_comm_params(cfg, latency_s=0.0, packet_loss=0.0,
                    comm_range_m=30.0, max_lookahead=3)


# Ordered low -> high comm stress: (latency, loss) then 1.0 s latency pairs.
CS1_SCENARIOS = {
    "CS1_HDV_Baseline": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "10.csv",
        "apply": _cs1_hdv_baseline,
    },
    "CS1_Ideal": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "11.csv",
        "apply": _cs1({"latency_s": 0.0, "packet_loss": 0.00,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
    "CS1_HighLoss": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "12.csv",
        "apply": _cs1({"latency_s": 0.0, "packet_loss": 0.50,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
    "CS1_MidLatency_LowLoss": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "13.csv",
        "apply": _cs1({"latency_s": 0.5, "packet_loss": 0.25,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
    "CS1_MidLatency_MidLoss": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "14.csv",
        "apply": _cs1({"latency_s": 0.5, "packet_loss": 0.50,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
    "CS1_HighLatency": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "15.csv",
        "apply": _cs1({"latency_s": 1.0, "packet_loss": 0.00,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
    "CS1_HighLatency_HighLoss": {
        "case": "CS1", "scenario": "Freeway", "freeway_type": CS1_FWY_TYPE,
        "output": "16.csv",
        "apply": _cs1({"latency_s": 1.0, "packet_loss": 0.50,
                       "comm_range_m": 30.0, "max_lookahead": 3}),
    },
}


# Freeway first (human -> blend -> connected); arterial cases last in CS2.
CS2_SCENARIOS = {
    "CS2_Fwy_100SV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "21.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=1.0),
    },
    "CS2_Fwy_90SV_10HV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "22.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.90, hv=0.10),
    },
    "CS2_Fwy_80SV_20HV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "23.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.80, hv=0.20),
    },
    "CS2_Fwy_45CAV_5CAHV_45SV_5HV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "24.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.45, hv=0.05, cav=0.45, cahv=0.05),
    },
    "CS2_Fwy_100CAV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "25.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=1.0),
    },
    "CS2_Fwy_90CAV_10CAHV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "26.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=0.90, cahv=0.10),
    },
    "CS2_Fwy_80CAV_20CAHV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "27.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=0.80, cahv=0.20),
    },
    "CS2_Fwy_90SV_10CAHV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "28.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.90, cahv=0.10),
    },
    "CS2_Fwy_90CAV_10HV": {
        "case": "CS2", "scenario": "Freeway", "freeway_type": CS2_FWY_TYPE,
        "output": "29.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=0.90, hv=0.10),
    },
    "CS2_Art_90SV_10HV": {
        "case": "CS2", "scenario": "Arterial", "output": "30.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.90, hv=0.10),
    },
    "CS2_Art_80SV_20HV": {
        "case": "CS2", "scenario": "Arterial", "output": "31.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, sv=0.80, hv=0.20),
    },
    "CS2_Art_90CAV_10CAHV": {
        "case": "CS2", "scenario": "Arterial", "output": "32.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=0.90, cahv=0.10),
    },
    "CS2_Art_80CAV_20CAHV": {
        "case": "CS2", "scenario": "Arterial", "output": "33.csv",
        "apply": lambda cfg: set_vehicle_mix(cfg, cav=0.80, cahv=0.20),
    },
}


def _cs3_enable_vru(cfg: Dict[str, Any]) -> None:
    """Pedestrians + bikes ON for CS3 (single intersection safety study)."""
    cfg["Ped_Allowed"] = True
    cfg["Bike_Allowed"] = True


def _cs3_apply_mix(cfg: Dict[str, Any], *, sv: float, av: float = 0.0, cav: float = 0.0) -> None:
    set_vehicle_mix(cfg, sv=sv, av=av, cav=cav)
    _cs3_enable_vru(cfg)


CS3_SCENARIOS = {
    "CS3_50SV_50CAV": {
        "case": "CS3", "scenario": "Single Intersection", "output": "34.csv",
        "apply": lambda cfg: _cs3_apply_mix(cfg, sv=0.50, cav=0.50),
    },
    "CS3_50SV_50AV": {
        "case": "CS3", "scenario": "Single Intersection", "output": "35.csv",
        "apply": lambda cfg: _cs3_apply_mix(cfg, sv=0.50, av=0.50),
    },
}


ALL_SCENARIOS: Dict[str, Dict[str, Any]] = {**CS1_SCENARIOS, **CS2_SCENARIOS, **CS3_SCENARIOS}


# ---------------------------------------------------------------------------
# Runner dispatcher
# ---------------------------------------------------------------------------

_RUNNERS: Dict[str, Callable[..., pd.DataFrame]] = {
    "Freeway":             run_sim_freeway,
    "Arterial":            run_sim_multi_inter,
    "Single Intersection": run_sim_single_inter,
}


def run_scenario(
    label: str,
    spec: Dict[str, Any],
    *,
    visualize: bool = False,
    overwrite: bool = True,
    repeats: int = DEFAULT_REPEATS,
    base_seed: int = 90210,
    sim_time_s: int = DEFAULT_SIM_TIME_S,
) -> Tuple[bool, str]:
    """Run a registered scenario ``repeats`` times into ``case_runs/<label>/``.

    Returns (ok, summary) where ok is True iff every replicate either completed
    this session or was skipped cleanly with ``--no-overwrite`` because the CSV
    already existed.
    """
    repeats = max(1, int(repeats))
    scen = spec["scenario"]
    freeway_type = spec.get("freeway_type")  # only meaningful for Freeway scenarios

    cd = case_runs_dir(label)
    case_runs_root = os.path.join(RESULTS_DIR, CASE_RUNS_SUBDIR)
    os.makedirs(case_runs_root, exist_ok=True)

    legacy_name = spec["output"]
    if not overwrite:
        pending = []
        for r in range(1, repeats + 1):
            p = os.path.join(cd, f"run_{r:02d}.csv")
            if not os.path.exists(p):
                pending.append(r)
        if not pending:
            msg = (
                f"[SKIP] {label}: all {repeats} runs exist -> {cd} "
                "(use --overwrite via clearing files or omit --no-overwrite)"
            )
            print(msg)
            return True, f"{label}: skipped ({repeats}/{repeats} already on disk)"

    cfg = load_baseline(scen, freeway_type=freeway_type)
    if cfg is None:
        extra = f" ({freeway_type})" if freeway_type else ""
        msg = (
            f"[SKIP] {label}: no baseline for '{scen}'{extra}. "
            f"Open the GUI, configure a {scen}{extra} run, start it once, then retry."
        )
        print(msg)
        return False, f"{label}: no baseline"

    cfg["Scenario"] = scen
    cfg["Sim_Visualization"] = bool(visualize)
    cfg["Sim_DataCollection"] = True
    cfg["Data_Folder"] = RESULTS_DIR
    cfg["Sim_Time"] = int(sim_time_s)

    if scen == "Freeway" and freeway_type:
        baseline_type = str(cfg.get("Geometry", {}).get("Freeway_Type", "")).strip().lower()
        if baseline_type != freeway_type.lower():
            msg = (
                f"[SKIP] {label}: loaded baseline has Freeway_Type='{baseline_type}' "
                f"but scenario needs '{freeway_type}'. Run a GUI simulation with "
                f"Freeway_Type='{freeway_type}' to produce baseline_freeway_{freeway_type}.pkl."
            )
            print(msg)
            return False, f"{label}: baseline freeway_type mismatch"

        cfg["Geometry"]["Freeway_Type"] = freeway_type

    try:
        spec["apply"](cfg)
    except Exception as ex:
        msg = f"[SKIP] {label}: apply() failed -> {ex}"
        print(msg)
        return False, f"{label}: apply failed"

    apply_case_study_driving_models(cfg)
    cfg_template = copy.deepcopy(cfg)

    runner = _RUNNERS[scen]
    os.makedirs(cd, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    ok_count = 0
    skipped = 0
    failed_any = False
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    print(f"\n>>> Running {label}  [{scen}]  x{repeats}  -> {cd}/run_##.csv")
    ft = freeway_type or ""

    for run_idx in range(1, repeats + 1):
        out_path = os.path.join(cd, f"run_{run_idx:02d}.csv")
        prefix = f"{label} run {run_idx:02d}/{repeats:d} "
        seed = _scenario_run_seed(base_seed, label, run_idx)

        if os.path.exists(out_path) and not overwrite:
            print(f"    [{prefix}] SKIP (exists) -> {out_path}")
            skipped += 1
            continue

        apply_stochastic_seed(seed)
        cfg_run = copy.deepcopy(cfg_template)

        t0 = time.time()
        try:
            data = runner(
                cfg_run,
                progress_cb=_Progress(prefix=prefix).emit,
                is_running_check=_is_running_always,
            )
        except Exception as ex:
            failed_any = True
            elapsed = time.time() - t0
            print(f"[FAIL] {label} run {run_idx}: simulation raised: {ex}")
            traceback.print_exc()
            try:
                import traci
                traci.close()
            except Exception:
                pass
            manifest_rows.append({
                "ts_utc": ts,
                "scenario_label": label,
                "case_study": spec.get("case", ""),
                "gui_scenario": scen,
                "freeway_type": ft,
                "legacy_output": legacy_name,
                "run_index": run_idx,
                "seed": seed,
                "csv_path": os.path.relpath(out_path, RESULTS_DIR),
                "n_rows": "",
                "wall_s": round(elapsed, 2),
            })
            continue

        elapsed = time.time() - t0

        if not isinstance(data, pd.DataFrame) or data.empty:
            failed_any = True
            print(f"[WARN] {label} run {run_idx}: no data — CSV not written")
            manifest_rows.append({
                "ts_utc": ts,
                "scenario_label": label,
                "case_study": spec.get("case", ""),
                "gui_scenario": scen,
                "freeway_type": ft,
                "legacy_output": legacy_name,
                "run_index": run_idx,
                "seed": seed,
                "csv_path": os.path.relpath(out_path, RESULTS_DIR),
                "n_rows": 0,
                "wall_s": round(elapsed, 2),
            })
            continue

        data.to_csv(out_path, index=False)
        ok_count += 1
        manifest_rows.append({
            "ts_utc": ts,
            "scenario_label": label,
            "case_study": spec.get("case", ""),
            "gui_scenario": scen,
            "freeway_type": ft,
            "legacy_output": legacy_name,
            "run_index": run_idx,
            "seed": seed,
            "csv_path": os.path.relpath(out_path, RESULTS_DIR),
            "n_rows": len(data),
            "wall_s": round(elapsed, 2),
        })
        print(f"[OK]   {prefix}{len(data):,} rows in {elapsed:.1f}s -> {out_path}")

    _append_batch_manifest(case_runs_root, manifest_rows)

    total_ok = skipped + ok_count
    summary = f"{label}: {ok_count} written, {skipped} skipped-existing, failures={failed_any}"
    batch_ok = (not failed_any) and (total_ok == repeats)

    if not batch_ok and not failed_any and total_ok < repeats:
        summary = f"{label}: incomplete ({total_ok}/{repeats}); see logs"
    elif batch_ok:
        summary = f"{label}: OK ({ok_count} new runs, {skipped} skipped)"

    return batch_ok, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch runner for Case Studies 1, 2, 3.")
    parser.add_argument("--cs1", action="store_true", help="Run Case Study 1 (freeway, V2V comm)")
    parser.add_argument("--cs2", action="store_true", help="Run Case Study 2 (MPR, freeway + arterial)")
    parser.add_argument(
        "--cs3",
        action="store_true",
        help="Run Case Study 3 (single intersection: 50/50 SV+CAV and 50/50 SV+AV)",
    )
    parser.add_argument("--all", action="store_true",
                        help="Run every scenario (same as default when no CS flags are passed)")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only the named scenario(s); repeatable. Names come from the registry.")
    parser.add_argument("--visualize", action="store_true", help="Open SUMO-GUI for every run")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip a replicate if results/case_runs/<scenario>/run_##.csv already exists")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Independent replicates per scenario (folders under results/case_runs/)")
    parser.add_argument("--base-seed", type=int, default=90210,
                        dest="base_seed",
                        help="Deterministic RNG base for Python random + NumPy before each replicate")
    parser.add_argument("--sim-time", type=int, default=DEFAULT_SIM_TIME_S,
                        dest="sim_time",
                        help=f"Simulation duration in seconds for all scenarios (default: {DEFAULT_SIM_TIME_S})")
    args = parser.parse_args(argv)

    selected: Dict[str, Dict[str, Any]] = {}

    if args.only:
        for name in args.only:
            if name not in ALL_SCENARIOS:
                print(f"[ERROR] Unknown scenario '{name}'. Available:")
                for k in ALL_SCENARIOS:
                    print(f"  - {k}")
                return 2
            selected[name] = ALL_SCENARIOS[name]
    else:
        if args.all:
            selected.update(ALL_SCENARIOS)
        elif args.cs1 or args.cs2 or args.cs3:
            if args.cs1:
                selected.update(CS1_SCENARIOS)
            if args.cs2:
                selected.update(CS2_SCENARIOS)
            if args.cs3:
                selected.update(CS3_SCENARIOS)
        else:
            selected.update(ALL_SCENARIOS)

    nrep = max(1, args.repeats)
    print("Scenarios queued:")
    print(f"  - Sim_Time: {int(args.sim_time)} s")
    for name, spec in selected.items():
        env = spec["scenario"]
        if spec.get("freeway_type"):
            env = f"{env}/{spec['freeway_type']}"
        loc = os.path.join("results", CASE_RUNS_SUBDIR, name)
        print(f"  - {name}  [{env}]  x{nrep}  -> {loc}/run_##.csv  (manifest: {CASE_RUNS_SUBDIR}/{MANIFEST_FILENAME})")
    print()

    results = []
    for name, spec in selected.items():
        ok, summ = run_scenario(
            name,
            spec,
            visualize=args.visualize,
            overwrite=not args.no_overwrite,
            repeats=args.repeats,
            base_seed=args.base_seed,
            sim_time_s=args.sim_time,
        )
        results.append((name, ok, summ))

    print("\n================= BATCH SUMMARY =================")
    for name, ok, summ in results:
        flag = "OK " if ok else "SKIP/FAIL "
        print(f"  {flag} {summ}")
    print("==================================================")
    return 1 if any(not ok for _, ok, __ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
