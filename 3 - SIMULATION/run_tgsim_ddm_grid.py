"""
Grid search for TGSIM (I90/94) with PT car-following + DDM lane-changing.

Each grid point runs one 900 s simulation (headless SUMO) and writes a single
trajectory CSV under ``results/tgsim_ddm_grid/``.

Fixed settings (unless you edit ``apply_fixed_settings``):
  - Scenario: TGSIM / I90/94 (geometry + volumes from baseline_tgsim.pkl)
  - CF_Model: PT with default calibrated parameters (CF_Default_Params=True)
  - LC_Model: DDM with per-run LC_Parameters (LC_Default_Params=False)
  - Fleet: 98% SV, 2% HV (no AV / CAV / CAHV)
  - Sim_Time: 900 s

Usage (from ``3 - SIMULATION``):

    python run_tgsim_ddm_grid.py
    python run_tgsim_ddm_grid.py --list
    python run_tgsim_ddm_grid.py --visualize
    python run_tgsim_ddm_grid.py --no-overwrite

Requires ``results/baseline_tgsim.pkl`` from at least one GUI TGSIM run
(or edit ``build_fallback_baseline()`` if missing).

Edit ``DDM_COUPLED_LEVELS`` (beta_0 + beta_MLC together) and ``DDM_GRID``
(independent sweeps, e.g. sigma, beta_V) below.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
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
OUTPUT_SUBDIR = "tgsim_ddm_grid"
MANIFEST_NAME = "grid_manifest.csv"
BASELINE_TGSIM = os.path.join(RESULTS_DIR, "baseline_tgsim.pkl")

VEH_CLASSES = ("Small Vehicle", "Automated Vehicle", "Heavy Vehicle")

# GUI / tgsim_sim.py DDM parameter names (must match LC_Parameters keys exactly).
DDM_PARAM_NAMES = (
    "α_h",
    "β_0_left",
    "β_0_right",
    "β_G",
    "G_0",
    "β_V",
    "β_MLC",
    "σ",
)

# Defaults when a parameter is not in the grid (same as GUI.py DDM_DEFAULTS).
DDM_DEFAULTS: Dict[str, float] = {
    "α_h": 0.08,
    "β_0_left": -3.5,
    "β_0_right": -4.2,
    "β_G": 0.2737,
    "G_0": 8.69,
    "β_V": 0.6808,
    "β_MLC": 87.0,
    "σ": 8.458,
}

# beta_0 and beta_MLC are swept TOGETHER so mandatory LC stays roughly similar
# while discretionary LC is reduced (more negative beta_0, higher beta_MLC).
COUPLED_PARAM_KEYS = ("β_0_left", "β_0_right", "β_MLC")

# Each level: more negative beta_0 <-> higher beta_MLC (rough parity when MLC=1).
DDM_COUPLED_LEVELS: List[Dict[str, Any]] = [
    {
        "tag": "L0_default",
        "β_0_left": -3.5,
        "β_0_right": -4.2,
        "β_MLC": 87.0,
    },
    {
        "tag": "L1_mild",
        "β_0_left": -1.2,
        "β_0_right": -1.6,
        "β_MLC": 66.5,
    },
    {
        "tag": "L2_medium",
        "β_0_left": -2.0,
        "β_0_right": -2.5,
        "β_MLC": 73.0,
    },
    {
        "tag": "L3_strong",
        "β_0_left": -2.8,
        "β_0_right": -3.4,
        "β_MLC": 80.0,
    },
    {
        "tag": "L4_extreme",
        "β_0_left": -3.5,
        "β_0_right": -4.2,
        "β_MLC": 87.0,
    },
]

# Independent parameters (Cartesian product across levels above).
DDM_GRID: Dict[str, List[float]] = {
    "σ": [8.458],
    "β_V": [0.6808],
}


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
        "LC_Model": "DDM",
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
    cfg["LC_Model"] = "DDM"
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


def set_ddm_parameters(cfg: Dict[str, Any], values: Dict[str, float]) -> None:
    """Inject DDM means (Std=0) for all three vehicle classes."""
    merged = dict(DDM_DEFAULTS)
    merged.update(values)
    lc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pname in DDM_PARAM_NAMES:
        mean_v = float(merged[pname])
        lc[pname] = {
            cls: {"Mean": mean_v, "Std": 0.0}
            for cls in VEH_CLASSES
        }
    cfg["LC_Parameters"] = lc
    cfg["LC_Default_Params"] = False
    cfg["LC_Model"] = "DDM"


def iter_grid() -> Iterable[Tuple[str, Dict[str, float], Dict[str, float]]]:
    """Yield (couple_level_tag, overrides, full_ddm_dict) for each combination."""
    indep_keys = [k for k in DDM_PARAM_NAMES if k in DDM_GRID and DDM_GRID[k]]
    indep_combos: List[Tuple[float, ...]] = (
        list(itertools.product(*[DDM_GRID[k] for k in indep_keys]))
        if indep_keys
        else [()]
    )
    for level in DDM_COUPLED_LEVELS:
        tag = str(level["tag"])
        for combo in indep_combos:
            overrides: Dict[str, float] = {
                k: float(level[k]) for k in COUPLED_PARAM_KEYS
            }
            for i, k in enumerate(indep_keys):
                overrides[k] = float(combo[i])
            full = dict(DDM_DEFAULTS)
            full.update(overrides)
            yield tag, overrides, full


def _param_slug(overrides: Dict[str, float]) -> str:
    """Filesystem-safe tag from swept parameters only."""
    short = {
        "α_h": "ah",
        "β_0_left": "b0L",
        "β_0_right": "b0R",
        "β_G": "bG",
        "G_0": "G0",
        "β_V": "bV",
        "β_MLC": "bMLC",
        "σ": "sigma",
    }
    parts = []
    for k in sorted(overrides.keys()):
        v = overrides[k]
        tag = short.get(k, k.replace("β_", "b").replace("α_", "a"))
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
    """ASCII label for console output on Windows cp1252."""
    return {
        "α_h": "alpha_h",
        "β_0_left": "beta_0_left",
        "β_0_right": "beta_0_right",
        "β_G": "beta_G",
        "G_0": "G_0",
        "β_V": "beta_V",
        "β_MLC": "beta_MLC",
        "σ": "sigma",
    }.get(name, name)


def _run_slug(level_tag: str, overrides: Dict[str, float]) -> str:
    param_part = _param_slug(overrides) or "defaults"
    indep = [k for k in overrides if k not in COUPLED_PARAM_KEYS]
    if indep:
        return f"{level_tag}_{param_part}"
    return level_tag


def list_grid() -> int:
    combos = list(iter_grid())
    print(f"DDM grid ({len(combos)} runs):")
    print(f"  Coupled levels ({len(DDM_COUPLED_LEVELS)}): beta_0_left, beta_0_right, beta_MLC")
    for level in DDM_COUPLED_LEVELS:
        print(
            f"    {level['tag']}: "
            f"beta_0_left={level['β_0_left']:g}, "
            f"beta_0_right={level['β_0_right']:g}, "
            f"beta_MLC={level['β_MLC']:g}"
        )
    swept = [_param_display(k) for k in DDM_GRID if DDM_GRID.get(k)]
    print(f"  Independent sweeps: {swept or '(none)'}")
    for i, (level_tag, ov, full) in enumerate(combos, start=1):
        print(f"  {i:3d}. {_run_slug(level_tag, ov)}")
        for k in COUPLED_PARAM_KEYS:
            print(
                f"       {_param_display(k)} = {full[k]:g}  "
                f"(default was {DDM_DEFAULTS[k]:g})"
            )
        for k in DDM_PARAM_NAMES:
            if k in ov and k not in COUPLED_PARAM_KEYS:
                print(
                    f"       {_param_display(k)} = {full[k]:g}  "
                    f"(default was {DDM_DEFAULTS[k]:g})"
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
    parser = argparse.ArgumentParser(description="TGSIM PT+DDM DDM parameter grid search.")
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
    print(f"TGSIM DDM grid: {len(combos)} run(s) -> {out_dir}/")
    print(
        f"  PT + DDM | Sim_Time={args.sim_time}s | 98% SV / 2% HV | "
        f"{len(DDM_COUPLED_LEVELS)} coupled beta_0/beta_MLC levels"
    )

    fieldnames = [
        "ts_utc",
        "run_index",
        "couple_level",
        "seed",
        "csv_path",
        "n_rows",
        "wall_s",
        "status",
    ] + [f"ddm_{k}" for k in DDM_PARAM_NAMES]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for run_idx, (level_tag, overrides, full_ddm) in enumerate(combos, start=1):
        slug = _run_slug(level_tag, overrides)
        csv_name = f"run_{run_idx:03d}_{slug}.csv"
        out_path = os.path.join(out_dir, csv_name)

        if os.path.isfile(out_path) and args.no_overwrite:
            print(f"[SKIP] {run_idx}/{len(combos)} exists -> {csv_name}")
            skip_count += 1
            continue

        cfg = copy.deepcopy(cfg_base)
        apply_fixed_settings(cfg, sim_time_s=args.sim_time, visualize=args.visualize)
        set_ddm_parameters(cfg, overrides)

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
                print(f"[WARN] No data collected — CSV not written")
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
            "couple_level": level_tag,
            "seed": seed,
            "csv_path": os.path.relpath(out_path, RESULTS_DIR),
            "n_rows": n_rows,
            "wall_s": round(time.time() - t0, 2),
            "status": status,
        }
        for k in DDM_PARAM_NAMES:
            row[f"ddm_{k}"] = full_ddm[k]
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
