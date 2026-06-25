"""
Shared repository paths for the NGM project.

All calibration scripts should import dataset locations from here so a fresh
clone works after downloading data into ``0 - Datasets/``.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(REPO_ROOT, "0 - Datasets")

# Freeway TGSIM trajectories (Kaggle filenames)
TGSIM_CALIBRATION_DATASETS: Dict[str, str] = {
    "I395": "Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv",
    "I9094": "Third_Generation_Simulation_Data__TGSIM__I-90_I-94_Moving_Trajectories.csv",
    "I294l1": "Third_Generation_Simulation_Data__TGSIM__I-294_L1_Trajectories.csv",
    "I294l2": "Third_Generation_Simulation_Data__TGSIM__I-294_L2_Trajectories.csv",
}

TGSIM_LANE_CHANGE_DATASETS: Dict[str, str] = {
    "I395": TGSIM_CALIBRATION_DATASETS["I395"],
    "I9094": TGSIM_CALIBRATION_DATASETS["I9094"],
    "I294_L1": TGSIM_CALIBRATION_DATASETS["I294l1"],
    "I294_L2": TGSIM_CALIBRATION_DATASETS["I294l2"],
}

FOGGY_BOTTOM_CSV = "Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv"
ORGANIZED_DATA_PKL = "organized_data.pkl"
DIS_SURROUNDING_CSV = "Dis-surrounding_info_with_event_id.csv"
ALL_LANE_CHANGES_CSV = "all_lane_changes_output.csv"

WAYMO_CF_GLOB = "March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv"
WAYMO_MAP_GLOB = "March2023waymo_map_features_*_data.csv"

_TGSIM_COLUMN_RENAME = {
    "id": "ID",
    "xloc_kf": "xloc-kf",
    "yloc_kf": "yloc-kf",
    "lane_kf": "lane-kf",
    "speed_kf": "speed-kf",
    "acceleration_kf": "acceleration-kf",
    "type_most_common": "type-most-common",
    "run_index": "run-index",
    "acc": "ACC",
}


def ensure_repo_on_path(caller_file: str) -> str:
    """Insert repo root on ``sys.path`` so ``import ngm_paths`` works from subfolders."""
    root = os.path.dirname(os.path.abspath(caller_file))
    while True:
        if os.path.isdir(os.path.join(root, "0 - Datasets")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


def dataset_path(filename: str) -> str:
    return os.path.join(DATASETS_DIR, filename)


def calibration_dataset_paths() -> Dict[str, str]:
    return {key: dataset_path(name) for key, name in TGSIM_CALIBRATION_DATASETS.items()}


def lane_change_dataset_paths() -> Dict[str, str]:
    return {key: dataset_path(name) for key, name in TGSIM_LANE_CHANGE_DATASETS.items()}


def discover_waymo_datasets(
    datasets_dir: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Discover Waymo leader–follower and map CSV pairs under ``0 - Datasets/``.

    Scans ``March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv`` and
    pairs each file with ``March2023waymo_map_features_{suffix}_data.csv`` when present.
    Trajectory files without a matching map are still included (lane projection falls
    back to path arc length). Returns ``(trajectory_paths, map_paths)`` keyed by
    ``Waymo_{suffix}``.
    """
    root = datasets_dir or DATASETS_DIR
    data_files: Dict[str, str] = {}
    map_files: Dict[str, str] = {}
    for cf_path in sorted(glob.glob(os.path.join(root, WAYMO_CF_GLOB))):
        suffix = cf_path.split("_assigned_")[-1].replace("_data.csv", "")
        tag = f"Waymo_{suffix}"
        data_files[tag] = cf_path
        map_path = os.path.join(root, f"March2023waymo_map_features_{suffix}_data.csv")
        if os.path.isfile(map_path):
            map_files[tag] = map_path
    return data_files, map_files


def load_tgsim_csv(path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a TGSIM trajectory CSV and normalize column names for legacy scripts.

    Kaggle exports use underscores (``xloc_kf``); older NGM code expects hyphens
    (``xloc-kf``) and a ``run-index`` column.
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    rename = {c: _TGSIM_COLUMN_RENAME[c] for c in df.columns if c in _TGSIM_COLUMN_RENAME}
    df = df.rename(columns=rename)
    if "run-index" not in df.columns:
        df["run-index"] = df["ID"]
    if "ACC" in df.columns:
        df["ACC"] = df["ACC"].astype(str).str.strip()
    return df


def simulation_results_dir() -> str:
    return os.path.join(REPO_ROOT, "2 - SIMULATION", "results")


def lateral_calibration_dir() -> str:
    return os.path.join(REPO_ROOT, "1 - PARAMETRIC INPUT", "1.3 - Lateral Calibration")


def lateral_processed_dir() -> str:
    return os.path.join(lateral_calibration_dir(), "Processed")
