"""
Grid search for TGSIM (I90/94) with PT car-following + MOBIL lane-changing.

Each grid point runs one 900 s simulation (headless SUMO) and writes a single
trajectory CSV under ``results/tgsim_mobil_grid/``.

Fixed settings (unless you edit ``apply_fixed_settings``):
  - Scenario: TGSIM / I90/94 (geometry + volumes from baseline_tgsim.pkl)
  - CF_Model: PT with default calibrated parameters (CF_Default_Params=True)
  - LC_Model: MOBIL with per-run LC_Parameters (LC_Default_Params=False)
  - Fleet: 98% SV, 2% HV (no AV / CAV / CAHV)
  - Sim_Time: 900 s

Usage (from ``3 - SIMULATION``):

    python run_tgsim_mobil_grid.py
    python run_tgsim_mobil_grid.py --list
    python run_tgsim_mobil_grid.py --visualize
    python run_tgsim_mobil_grid.py --no-overwrite

Requires ``results/baseline_tgsim.pkl`` from at least one GUI TGSIM run
(or edit ``build_fallback_baseline()`` if missing).

Edit ``MOBIL_PRESET_LEVELS`` below (consistent tuples: all params push toward
fewer or more lane changes together; not a blind Cartesian product).
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.tgsim_sim import run_sim_tgsim  # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
OUTPUT_SUBDIR = "tgsim_mobil_grid"
MANIFEST_NAME = "grid_manifest.csv"
BASELINE_TGSIM = os.path.join(RESULTS_DIR, "baseline_tgsim.pkl")

VEH_CLASSES = ("Small Vehicle", "Automated Vehicle", "Heavy Vehicle")

# Must match GUI.py MOBIL_PARAMS / LC_Parameters keys exactly.
MOBIL_PARAM_NAMES = (
    "Disc: p_opt",
    "Disc: a_th",
    "Disc: b_safe",
    "Mand: b_safe",
)

# GUI defaults when a parameter is not in the grid.
MOBIL_DEFAULTS: Dict[str, float] = {
    "Disc: p_opt": 0.5,
    "Disc: a_th": 0.8,
    "Disc: b_safe": 4.5,
    "Mand: b_safe": 7.5,
}

# Each preset moves all four parameters in a consistent direction:
#   fewer LC:  high p, high a_th, LOW Disc/Mand b_safe
#   more LC:   low p, low a_th, HIGH Disc/Mand b_safe
MOBIL_PRESET_LEVELS: List[Dict[str, Any]] = [
  # many LC — low threshold, loose safety, low politeness
  {"tag": "L0_many_LC",  "Disc: p_opt": 0.1, "Disc: a_th": 1.0, "Disc: b_safe": 6.5, "Mand: b_safe": 8.0},
  {"tag": "L1_mid_many", "Disc: p_opt": 0.2, "Disc: a_th": 1.5, "Disc: b_safe": 5.5, "Mand: b_safe": 7.5},
  {"tag": "L2_default",  "Disc: p_opt": 0.3, "Disc: a_th": 2.0, "Disc: b_safe": 4.5, "Mand: b_safe": 6.5},
  {"tag": "L3_mid_few",  "Disc: p_opt": 0.4, "Disc: a_th": 2.5, "Disc: b_safe": 4.0, "Mand: b_safe": 6.0},
  {"tag": "L4_few_LC",   "Disc: p_opt": 0.5, "Disc: a_th": 3.0, "Disc: b_safe": 3.5, "Mand: b_safe": 5.5},
]


def _load_baseline() -> Optional[Dict[str, Any]]:
    if not os.path.isfile(BASELINE_TGSIM):
        return None
    try:
        with open(BASELINE_TGSIM, "rb") as fh:
            return copy.deepcopy(pickle.load(fh))
    except Exception as ex:
        print(f"[baseline] failed to load {BASELINE_TGSIM}: {ex}")
        return None


def build_fallback_baseline() -> Dict[str, Any]:
    """Minimal TGSIM config if baseline pickle is missing."""
    vf = {
        "SV_rate": 0.98,
        "HV_rate": 0.02,
        "AV_rate": 0.0,
        "CAV_rate": 0.0,
        "CAHV_rate": 0.0,
    }
    defaults = {
        0: (1450, 0),
        1: (1450, 0),
        2: (315, 190),
        3: (15, 890),
        4: (0, 1150),
        5: (0, 1150),
    }
    for k in range(6):
        vf[f"MainL{k}-NB"] = float(defaults[k][0])
        vf[f"MainL{k}-SB"] = float(defaults[k][1])
    return {
        "Scenario": "TGSIM",
        "TGSIM_Network": "I90/94",
        "Geometry": {"TGSIM_Network": "I90/94"},
        "Vehicle_Flows": vf,
        "CF_Model": "PT",
        "LC_Model": "MOBIL",
        "CF_Default_Params": True,
        "LC_Default_Params": True,
        "CIDM_Params": {"Default": True, "K_v": 0.1, "K_a": 0.03, "s_ref": 35.0},
        "CMOBIL_Params": {"Default": True, "kappa": 0.1, "gamma": 1.0},
        "Comm_Params": {"Default": True, "Range": 30.0, "Lookahead": 3, "Latency": 0, "Loss": 0.0},
        "Sim_StepSize": 0.1,
        "Sim_Time": 900,
        "Sim_Visualization": False,
        "Sim_DataCollection": True,
        "Sim_DataFreq": 0.5,
        "Data_Folder": RESULTS_DIR,
        "PostSim_Viz": [],
    }


def apply_fixed_settings(cfg: Dict[str, Any], *, sim_time_s: int, visualize: bool) -> None:
    cfg["Scenario"] = "TGSIM"
    cfg.setdefault("Geometry", {})["TGSIM_Network"] = cfg.get("TGSIM_Network", "I90/94")
    cfg["TGSIM_Network"] = cfg.get("TGSIM_Network", "I90/94")
    cfg["CF_Model"] = "PT"
    cfg["LC_Model"] = "MOBIL"
    cfg["CF_Default_Params"] = True
    cfg["Sim_Time"] = int(sim_time_s)
    cfg["Sim_Visualization"] = bool(visualize)
    cfg["Sim_DataCollection"] = True
    cfg["Data_Folder"] = RESULTS_DIR
    set_vehicle_mix(cfg, sv=0.98, hv=0.02)


def set_vehicle_mix(cfg: Dict[str, Any], *, sv: float, hv: float) -> None:
    vf = cfg.setdefault("Vehicle_Flows", {})
    vf["SV_rate"] = float(sv)
    vf["HV_rate"] = float(hv)
    vf["AV_rate"] = 0.0
    vf["CAV_rate"] = 0.0
    vf["CAHV_rate"] = 0.0


def set_mobil_parameters(cfg: Dict[str, Any], values: Dict[str, float]) -> None:
    """Inject MOBIL means (Std=0) for all three vehicle classes."""
    merged = dict(MOBIL_DEFAULTS)
    merged.update(values)
    lc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pname in MOBIL_PARAM_NAMES:
        mean_v = float(merged[pname])
        lc[pname] = {
            cls: {"Mean": mean_v, "Std": 0.0}
            for cls in VEH_CLASSES
        }
    cfg["LC_Parameters"] = lc
    cfg["LC_Default_Params"] = False
    cfg["LC_Model"] = "MOBIL"


def iter_grid() -> Iterable[Tuple[str, Dict[str, float], Dict[str, float]]]:
    """Yield (preset_tag, overrides, full_mobil_dict) for each level."""
    for level in MOBIL_PRESET_LEVELS:
        tag = str(level["tag"])
        overrides = {k: float(level[k]) for k in MOBIL_PARAM_NAMES}
        full = dict(MOBIL_DEFAULTS)
        full.update(overrides)
        yield tag, overrides, full


def _param_slug(overrides: Dict[str, float]) -> str:
    short = {
        "Disc: p_opt": "pDisc",
        "Disc: a_th": "aDisc",
        "Disc: b_safe": "bDisc",
        "Mand: b_safe": "bMand",
    }
    parts = []
    for k in sorted(overrides.keys()):
        v = overrides[k]
        tag = short.get(k, k.replace(":", "").replace(" ", ""))
        parts.append(f"{tag}{v:g}".replace(".", "p").replace("-", "m"))
    return "_".join(parts) if parts else "defaults"


def _run_seed(base_seed: int, run_index: int, slug: str) -> int:
    ih = int.from_bytes(hashlib.sha256(slug.encode("utf-8")).digest()[:4], "big")
    return (int(base_seed) + ih + int(run_index) * 7919) % (2**31)


def apply_stochastic_seed(seed: int) -> None:
    import random

    import numpy as np

    s = int(seed) % (2**31)
    random.seed(s)
    np.random.seed(s)


class _Progress:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._last = -1

    def emit(self, pct: int) -> None:
        pct = int(pct)
        if pct == self._last:
            return
        self._last = pct
        if pct % 10 == 0 or pct == 100:
            print(f"    {self.prefix}progress: {pct}%", flush=True)


def _param_display(name: str) -> str:
    return {
        "Disc: p_opt": "Disc_p_opt",
        "Disc: a_th": "Disc_a_th",
        "Disc: b_safe": "Disc_b_safe",
        "Mand: b_safe": "Mand_b_safe",
    }.get(name, name)


def _run_slug(level_tag: str, overrides: Dict[str, float]) -> str:
    return level_tag


def list_grid() -> int:
    combos = list(iter_grid())
    print(f"MOBIL grid ({len(combos)} runs) — consistent presets (few LC -> many LC):")
    for i, (level_tag, ov, full) in enumerate(combos, start=1):
        print(f"  {i:3d}. {level_tag}")
        for k in MOBIL_PARAM_NAMES:
            print(
                f"       {_param_display(k)} = {full[k]:g}  "
                f"(default was {MOBIL_DEFAULTS[k]:g})"
            )
    return 0


def append_manifest(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TGSIM PT+MOBIL parameter grid search.")
    parser.add_argument("--list", action="store_true", help="Print grid combinations and exit")
    parser.add_argument("--visualize", action="store_true", help="Open SUMO-GUI (slow)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip if output CSV exists")
    parser.add_argument("--sim-time", type=int, default=900, dest="sim_time", help="Simulation duration (s)")
    parser.add_argument("--base-seed", type=int, default=90210, dest="base_seed")
    args = parser.parse_args(argv)

    if args.list:
        return list_grid()

    cfg_base = _load_baseline()
    if cfg_base is None:
        print(f"[warn] {BASELINE_TGSIM} not found — using built-in fallback volumes/geometry.")
        cfg_base = build_fallback_baseline()

    if str(cfg_base.get("Scenario", "")).strip() != "TGSIM":
        print(f"[error] Baseline Scenario is '{cfg_base.get('Scenario')}', expected TGSIM.")
        return 2

    out_dir = os.path.join(RESULTS_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    combos = list(iter_grid())
    print(f"TGSIM MOBIL grid: {len(combos)} run(s) -> {out_dir}/")
    print(f"  PT + MOBIL | Sim_Time={args.sim_time}s | 98% SV / 2% HV")

    fieldnames = [
        "ts_utc",
        "run_index",
        "grid_tag",
        "seed",
        "csv_path",
        "n_rows",
        "wall_s",
        "status",
    ] + [f"mobil_{k.replace(': ', '_').replace(' ', '_')}" for k in MOBIL_PARAM_NAMES]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for run_idx, (level_tag, overrides, full_mobil) in enumerate(combos, start=1):
        slug = _run_slug(level_tag, overrides)
        csv_name = f"MOBIL_run_{run_idx:03d}_{slug}.csv"
        out_path = os.path.join(out_dir, csv_name)

        if os.path.isfile(out_path) and args.no_overwrite:
            print(f"[SKIP] {run_idx}/{len(combos)} exists -> {csv_name}")
            skip_count += 1
            continue

        cfg = copy.deepcopy(cfg_base)
        apply_fixed_settings(cfg, sim_time_s=args.sim_time, visualize=args.visualize)
        set_mobil_parameters(cfg, overrides)

        seed = _run_seed(args.base_seed, run_idx, slug)
        apply_stochastic_seed(seed)

        prefix = f"[{run_idx}/{len(combos)} {slug}] "
        print(f"\n>>> {prefix}seed={seed}")
        t0 = time.time()
        status = "ok"
        n_rows = 0
        try:
            data = run_sim_tgsim(
                cfg,
                progress_cb=_Progress(prefix=prefix).emit,
                is_running_check=lambda: True,
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                data.to_csv(out_path, index=False)
                n_rows = len(data)
                ok_count += 1
                print(f"[OK]   {n_rows:,} rows in {time.time() - t0:.1f}s -> {out_path}")
            else:
                status = "empty"
                fail_count += 1
                print("[WARN] No data collected — CSV not written")
        except Exception as ex:
            status = f"fail:{type(ex).__name__}"
            fail_count += 1
            print(f"[FAIL] {ex}")
            traceback.print_exc()
            try:
                import traci

                traci.close()
            except Exception:
                pass

        row = {
            "ts_utc": ts,
            "run_index": run_idx,
            "grid_tag": level_tag,
            "seed": seed,
            "csv_path": os.path.relpath(out_path, RESULTS_DIR),
            "n_rows": n_rows,
            "wall_s": round(time.time() - t0, 2),
            "status": status,
        }
        for k in MOBIL_PARAM_NAMES:
            col = f"mobil_{k.replace(': ', '_').replace(' ', '_')}"
            row[col] = full_mobil[k]
        append_manifest(manifest_path, row, fieldnames)

    print("\n================= GRID SUMMARY =================")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Manifest: {manifest_path}")
    print("==================================================")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
