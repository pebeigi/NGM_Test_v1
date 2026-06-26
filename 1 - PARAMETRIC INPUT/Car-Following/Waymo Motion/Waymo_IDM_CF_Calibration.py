"""
Waymo Open Motion — IDM car-following calibration (GA).

Mirrors the workflow in ../TGSIM/IDM_CF_Calibration.py
but reads pre-processed Waymo leader–follower CSVs from ``0 - Datasets/`` at the repo root.
All ``March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv`` files in that
folder are loaded automatically (paired map CSVs used when present).

Input columns (per file):
  scenario_id, time, vehicle_id, object_type, center_x/y, velocity_x/y, length,
  lane_changing, leader_vehicle_id, is_sdc, assigned_lane_feature_id

Calibration: skip cases where stable leader–follower overlap lasts less than MIN_CF_DURATION_S.
Kinematics: speed from velocity components; 1D position = distance along the follower's
  assigned lane polyline (map CSV + assigned_lane_feature_id). Leader is projected onto
  the same lane. Falls back to path arc length if map geometry is missing.
Simulation: closed-loop follower roll-out (IC at t=0 only; IDM integrates forward).
  Leader trajectory stays observed (open-loop).

Vehicle groups (follower type only — leader type is recorded but not used for grouping):
  Waymo_AV — autonomous / SDC follower (is_sdc == True)
  Waymo_SV — small passenger follower (length < LARGE_LENGTH_M)
  Waymo_HV — heavy follower (length >= LARGE_LENGTH_M)

Outputs under ./Results/:
  IDM_Params_Waymo_{AV,SV,HV}.csv
  IDM_Simulated_Waymo_{AV,SV,HV}.csv
  plots/IDM_Waymo_*_FID_*_LID_*_sc_*.png  (x–y, time–x, time–y, time–speed)
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & GA settings
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

_NGM_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import DATASETS_DIR, REPO_ROOT, discover_waymo_datasets

RESULTS_DIR = os.path.join(THIS_DIR, "Results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

DATA_FILES: Dict[str, str] = {}
MAP_FILES: Dict[str, str] = {}


def refresh_waymo_datasets(datasets_dir: Optional[str] = None) -> Dict[str, str]:
    """Scan ``0 - Datasets/`` for Waymo CSVs and update module-level path maps."""
    global DATA_FILES, MAP_FILES
    DATA_FILES, MAP_FILES = discover_waymo_datasets(datasets_dir)
    return DATA_FILES


def _print_waymo_dataset_summary(datasets: Dict[str, str]) -> None:
    print(f"Waymo trajectory files ({len(datasets)}) in {DATASETS_DIR}:")
    for tag in sorted(datasets):
        map_note = " + map" if tag in MAP_FILES else " (no map CSV)"
        print(f"  {tag}: {os.path.basename(datasets[tag])}{map_note}")

_scenario_source: Dict[str, str] = {}
_map_scenario_cache: Dict[str, pd.DataFrame] = {}
_lane_geom_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_loaded_map_tags: set = set()

# Returned by extract_subject_and_leader_data when lane projection succeeds:
# (ref_lane_feature_id, polyline_x, polyline_y, cumulative_s_along_polyline)
LaneGeom = Tuple[int, np.ndarray, np.ndarray, np.ndarray]

population_size = 50
num_generations = 100
mutation_rate = 0.12

MIN_CF_DURATION_S = 15.0  # skip if stable leader–follower overlap is shorter than this (s)
MIN_LEADER_SPEED_MPS = 1.0
LARGE_LENGTH_M = 6.0
TIME_STEP_S = 0.1
CONTIGUOUS_DT_MAX_S = 0.2

# IDM GA parameter ranges (same as IDM_CF_Calibration.py, wider v0 lower bound)
T_RANGE = (0.5, 4.0)
A_RANGE = (0.3, 3.5)
B_RANGE = (0.5, 4.0)
V0_RANGE = (1.0, 35.0)  # desired speed: 2–35 m/s (reference uses 5–35)
SO_RANGE = (1.0, 8.0)
DELTA_RANGE = (3.8, 4.2)

IDM_PARAM_RANGES = (T_RANGE, A_RANGE, B_RANGE, V0_RANGE, SO_RANGE, DELTA_RANGE)


def clip_idm_params(params) -> list:
    """Keep GA individuals within declared IDM parameter bounds."""
    return [float(np.clip(p, lo, hi)) for p, (lo, hi) in zip(params, IDM_PARAM_RANGES)]


IDM_PARAM_SUMMARY = [
    ("T", "T", T_RANGE),
    ("a", "a", A_RANGE),
    ("b", "b", B_RANGE),
    ("v0", "v0", V0_RANGE),
    ("so", "s0", SO_RANGE),
    ("delta", "δ", DELTA_RANGE),
]
GROUP_ORDER = ["Waymo_AV", "Waymo_SV", "Waymo_HV"]
SOURCE_TO_FOLLOWER_TYPE = {"Waymo_AV": "AV", "Waymo_SV": "SV", "Waymo_HV": "HV"}
FOLLOWER_TYPE_ORDER = ["AV", "SV", "HV"]
_LEGACY_FOLLOWER_TYPE = {"A": "AV", "S": "SV", "L": "HV"}

# Simulation globals (set per event before GA, same pattern as IDM_CF_Calibration.py)
pos = "lane-s"  # distance along reference lane polyline (m)
T = a = b = v0 = so = delta = None
most_leading_leader_id = None
sdf = ldf = None
follower_length_m = leader_length_m = 4.5
total_time = time_step = 0.0
timex = leader_position = leader_speed = target_position = target_speed = None


# ---------------------------------------------------------------------------
# Waymo CSV helpers
# ---------------------------------------------------------------------------
def parse_coord_list(val) -> List[float]:
    """Parse a CSV column that stores a coordinate list, e.g. '[1.0, 2.0, ...]'."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple)):
            return [float(x) for x in out]
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


def parse_id_list(val) -> List[int]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [int(x) for x in out]
        return [int(out)]
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_lane_feature_id(val) -> Optional[int]:
    """Parse map lane feature id from assigned_lane_feature_id (e.g. '272[121]' -> 272)."""
    if pd.isna(val):
        return None
    m = re.match(r"^(\d+)", str(val).strip())
    return int(m.group(1)) if m else None


def reference_lane_feature_id(lane_assignments: pd.Series) -> Optional[int]:
    """Most common lane feature id on a follower track segment."""
    ids = [x for x in lane_assignments.map(parse_lane_feature_id) if x is not None]
    if not ids:
        return None
    return int(pd.Series(ids, dtype=int).mode().iloc[0])


def get_lane_geometry(
    scenario_id: str, feature_id: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Lane polyline vertices and cumulative arc length (cached per scenario + feature)."""
    key = (str(scenario_id), int(feature_id))
    if key in _lane_geom_cache:
        return _lane_geom_cache[key]

    map_df = get_map_features_for_scenario(scenario_id)
    if map_df.empty:
        return None

    rows = map_df[map_df["feature_id"] == int(feature_id)]
    if rows.empty and "id" in map_df.columns:
        rows = map_df[map_df["id"] == int(feature_id)]
    if rows.empty:
        return None

    row = rows.iloc[0]
    xs = np.asarray(parse_coord_list(row.get("lane_polyline_x")), dtype=float)
    ys = np.asarray(parse_coord_list(row.get("lane_polyline_y")), dtype=float)
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    ds = np.hypot(np.diff(xs), np.diff(ys))
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    geom = (xs, ys, s_cum)
    _lane_geom_cache[key] = geom
    return geom


def project_xy_to_lane_s(px: float, py: float, xs: np.ndarray, ys: np.ndarray, s_cum: np.ndarray) -> float:
    """Distance along lane polyline from its start to the closest point to (px, py)."""
    best_s = 0.0
    best_d2 = np.inf
    for j in range(len(xs) - 1):
        x0, y0, x1, y1 = xs[j], ys[j], xs[j + 1], ys[j + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
            seg_len = 0.0
        else:
            t = float(np.clip(((px - x0) * dx + (py - y0) * dy) / seg_len2, 0.0, 1.0))
            seg_len = float(np.sqrt(seg_len2))
        qx = x0 + t * dx
        qy = y0 + t * dy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s_cum[j] + t * seg_len)
    return best_s


def project_xy_array_to_lane_s(
    center_x: np.ndarray, center_y: np.ndarray, xs: np.ndarray, ys: np.ndarray, s_cum: np.ndarray
) -> np.ndarray:
    return np.array(
        [project_xy_to_lane_s(float(x), float(y), xs, ys, s_cum) for x, y in zip(center_x, center_y)],
        dtype=float,
    )


def lane_s_to_xy(
    lane_s: np.ndarray, xs: np.ndarray, ys: np.ndarray, s_cum: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Map distance-along-lane values back to x/y on the polyline."""
    lane_s = np.asarray(lane_s, dtype=float)
    sx = np.zeros(len(lane_s))
    sy = np.zeros(len(lane_s))
    max_s = float(s_cum[-1])
    for i, s in enumerate(lane_s):
        s = float(np.clip(s, 0.0, max_s))
        j = int(np.searchsorted(s_cum, s, side="right") - 1)
        j = min(max(j, 0), len(xs) - 2)
        seg_len = float(s_cum[j + 1] - s_cum[j])
        if seg_len < 1e-9:
            t = 0.0
        else:
            t = (s - s_cum[j]) / seg_len
        sx[i] = xs[j] + t * (xs[j + 1] - xs[j])
        sy[i] = ys[j] + t * (ys[j + 1] - ys[j])
    return sx, sy


def arc_length_xy(center_x: np.ndarray, center_y: np.ndarray) -> np.ndarray:
    """Cumulative path length along raw Waymo center_x / center_y (m)."""
    x = np.asarray(center_x, dtype=float)
    y = np.asarray(center_y, dtype=float)
    if len(x) < 2:
        return np.zeros(len(x))
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def sim_xy_from_arc(
    sim_arc: np.ndarray,
    center_x_obs: np.ndarray,
    center_y_obs: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct simulated x/y by marching along observed heading using sim arc length."""
    n = len(sim_arc)
    sx = np.zeros(n)
    sy = np.zeros(n)
    sx[0] = float(center_x_obs[0])
    sy[0] = float(center_y_obs[0])
    headings = np.arctan2(velocity_y, velocity_x)
    for i in range(1, n):
        ds = float(sim_arc[i] - sim_arc[i - 1])
        h = float(headings[i - 1])
        if np.hypot(velocity_x[i - 1], velocity_y[i - 1]) < 0.5:
            h = float(headings[max(0, i - 2)])
        sx[i] = sx[i - 1] + ds * np.cos(h)
        sy[i] = sy[i - 1] + ds * np.sin(h)
    return sx, sy


def longest_contiguous_segment(times: np.ndarray, dt_max: float) -> List[float]:
    """Longest contiguous time segment from a sorted array of timestamps."""
    if len(times) == 0:
        return []
    max_continuous: List[float] = []
    continuous: List[float] = []
    prev_time = None
    for t in times:
        t = float(t)
        if prev_time is None or t - prev_time < dt_max:
            continuous.append(t)
        else:
            if len(continuous) > len(max_continuous):
                max_continuous = continuous
            continuous = [t]
        prev_time = t
    if len(continuous) > len(max_continuous):
        max_continuous = continuous
    return max_continuous


def mutual_cf_times(
    follower_times: np.ndarray,
    leader_times: np.ndarray,
) -> List[float]:
    """Longest contiguous mutual leader/follower timestamps."""
    mutual = np.intersect1d(follower_times, leader_times)
    return longest_contiguous_segment(mutual, CONTIGUOUS_DT_MAX_S)


def build_combined_dataframe(
    datasets: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int]]:
    """Load all trajectory CSVs, assign global run-index, and index scenario -> source file."""
    global _scenario_source

    parts: List[pd.DataFrame] = []
    for tag, path in datasets.items():
        if not os.path.isfile(path):
            print(f"[skip] missing file: {path}")
            continue
        df = load_waymo_cf_dataframe(path)
        df["source_tag"] = tag
        parts.append(df)

    combined_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _scenario_source = (
        combined_df.groupby("scenario_id")["source_tag"].first().to_dict()
        if not combined_df.empty
        else {}
    )
    global_run = {
        sc: i for i, sc in enumerate(sorted(combined_df["scenario_id"].unique()))
    }
    if not combined_df.empty:
        combined_df["run-index"] = combined_df["scenario_id"].map(global_run).astype(int)
    return combined_df, _scenario_source, global_run


def load_map_file_index(source_tag: str) -> None:
    """Lazy-load one map CSV and cache rows grouped by scenario_id."""
    global _loaded_map_tags, _map_scenario_cache

    if source_tag in _loaded_map_tags:
        return
    path = MAP_FILES.get(source_tag)
    if not path or not os.path.isfile(path):
        _loaded_map_tags.add(source_tag)
        return

    map_df = pd.read_csv(path)
    for sc, grp in map_df.groupby("scenario_id", sort=False):
        _map_scenario_cache[str(sc)] = grp
    _loaded_map_tags.add(source_tag)


def get_map_features_for_scenario(scenario_id: str) -> pd.DataFrame:
    """Return map feature rows for one scenario (lanes, edges, lines, crosswalks)."""
    sc = str(scenario_id)
    if sc in _map_scenario_cache:
        return _map_scenario_cache[sc]

    source_tag = _scenario_source.get(sc)
    if not source_tag:
        return pd.DataFrame()

    load_map_file_index(source_tag)
    return _map_scenario_cache.get(sc, pd.DataFrame())


def plot_map_on_ax(ax, map_df: pd.DataFrame) -> None:
    """Draw Waymo map polylgons/polylines behind vehicle trajectories."""
    if map_df is None or map_df.empty:
        return

    layer_specs = [
        ("lane_polyline_x", "lane_polyline_y", {"color": "#bdbdbd", "lw": 1.0, "alpha": 0.95}),
        ("road_edge_polyline_x", "road_edge_polyline_y", {"color": "#757575", "lw": 0.8, "alpha": 0.9}),
        ("road_line_polyline_x", "road_line_polyline_y", {"color": "#e0e0e0", "lw": 0.5, "ls": "--", "alpha": 0.85}),
        ("crosswalk_polygon_x", "crosswalk_polygon_y", {"color": "#eeeeee", "lw": 0.4, "alpha": 0.7, "close": True}),
    ]

    for _, row in map_df.iterrows():
        for xcol, ycol, style in layer_specs:
            xs = parse_coord_list(row.get(xcol))
            ys = parse_coord_list(row.get(ycol))
            if len(xs) < 2 or len(xs) != len(ys):
                continue
            plot_kw = {k: v for k, v in style.items() if k != "close"}
            if style.get("close") and len(xs) >= 3:
                xs = list(xs) + [xs[0]]
                ys = list(ys) + [ys[0]]
            ax.plot(xs, ys, zorder=1, **plot_kw)


def classify_vehicle_type(length_median: float, is_sdc: bool) -> str:
    """AV = SDC / ego, HV = long (>= LARGE_LENGTH_M), SV = passenger-sized."""
    if is_sdc:
        return "AV"
    if float(length_median) >= LARGE_LENGTH_M:
        return "HV"
    return "SV"


def normalize_follower_type(val) -> str:
    """Map follower type labels to AV / SV / HV (handles legacy S / L / A)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return _LEGACY_FOLLOWER_TYPE.get(s, s)


def build_vehicle_type_lookup(df: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    """Map (scenario_id, vehicle_id) -> AV / SV / HV."""
    meta = df.groupby(["scenario_id", "ID"], sort=False).agg(
        length_med=("length", "median"),
        is_sdc=("is_sdc", "any"),
    )
    return {
        (str(sc), int(vid)): classify_vehicle_type(row.length_med, bool(row.is_sdc))
        for (sc, vid), row in meta.iterrows()
    }


def load_waymo_cf_dataframe(path: str) -> pd.DataFrame:
    """Load one Waymo CSV; speed from raw velocity components, positions from center_x/y."""
    df = pd.read_csv(path)
    df = df[df["object_type"] == 1].copy()  # vehicles only
    df = df.sort_values(["scenario_id", "vehicle_id", "time"])
    df["time"] = df["time"].round(1)
    df["speed-kf"] = np.hypot(df["velocity_x"], df["velocity_y"])
    df["ID"] = df["vehicle_id"].astype(int)
    scenario_to_run = {sc: i for i, sc in enumerate(sorted(df["scenario_id"].unique()))}
    df["run-index"] = df["scenario_id"].map(scenario_to_run).astype(int)
    df["lane-kf"] = df["assigned_lane_feature_id"].astype(str)
    return df


def discover_car_following_events(df: pd.DataFrame) -> List[dict]:
    """Stable leader, no lane change, CF overlap >= MIN_CF_DURATION_S."""
    events: List[dict] = []
    for (scenario_id, follower_id), g in df.groupby(["scenario_id", "ID"], sort=False):
        g = g.sort_values("time")
        if int(g["lane_changing"].max()) != 0:
            continue

        leader_lists = g["leader_vehicle_id"].apply(parse_id_list)
        if leader_lists.apply(len).min() < 1:
            continue
        leader_id = leader_lists.iloc[0][0]
        if not leader_lists.apply(lambda xs: xs[0] if xs else -1).eq(leader_id).all():
            continue

        leader_g = df[(df["scenario_id"] == scenario_id) & (df["ID"] == leader_id)]
        if leader_g.empty:
            continue
        if float(leader_g["speed-kf"].max()) < MIN_LEADER_SPEED_MPS:
            continue

        cf_times = mutual_cf_times(g["time"].values, leader_g["time"].values)
        if len(cf_times) < 3:
            continue
        duration = float(cf_times[-1] - cf_times[0])
        if duration < MIN_CF_DURATION_S:
            continue

        length_med = float(g["length"].median())
        is_sdc = bool(g["is_sdc"].any())
        run_index = int(g["run-index"].iloc[0])

        events.append(
            {
                "follower_id": int(follower_id),
                "leader_id": int(leader_id),
                "scenario_id": str(scenario_id),
                "run_index": run_index,
                "duration": duration,
                "length_median": length_med,
                "is_sdc": is_sdc,
            }
        )
    return events


def generate_waymo_vehicle_groups(
    datasets: Dict[str, str],
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Return Waymo_{AV,SV,HV} lists of (follower_id, scenario_id).

    Grouping uses the **follower** type only (AV / SV / HV). scenario_id is the
    stable key — run-index is only unique within a single CSV file and must not
    be used when multiple Waymo files are combined.
    """
    groups: Dict[str, set] = {
        "Waymo_AV": set(),
        "Waymo_SV": set(),
        "Waymo_HV": set(),
    }
    for _tag, path in datasets.items():
        if not os.path.isfile(path):
            print(f"[skip] missing file: {path}")
            continue
        df = load_waymo_cf_dataframe(path)
        for ev in discover_car_following_events(df):
            key = (ev["follower_id"], ev["scenario_id"])
            if ev["is_sdc"]:
                groups["Waymo_AV"].add(key)
            elif ev["length_median"] >= LARGE_LENGTH_M:
                groups["Waymo_HV"].add(key)
            else:
                groups["Waymo_SV"].add(key)

    return {k: sorted(v) for k, v in groups.items()}


def extract_subject_and_leader_data(
    df: pd.DataFrame,
    follower_id: int,
    scenario_id: str,
    leader_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Optional[LaneGeom]]:
    """
    Extract follower + leader on the longest mutual CF segment (>= MIN_CF_DURATION_S).

    Longitudinal position is distance along the follower's reference lane polyline
    (mode assigned_lane_feature_id on the CF window, map CSV). Leader is projected
    onto the same polyline so gap is meaningful on curves. Falls back to per-vehicle
    path arc length when lane geometry is unavailable.
    """
    global most_leading_leader_id
    _empty: Tuple[pd.DataFrame, pd.DataFrame, float, float, None] = (
        pd.DataFrame(),
        pd.DataFrame(),
        0.0,
        0.0,
        None,
    )

    sdf = df[(df["ID"] == follower_id) & (df["scenario_id"] == scenario_id)].copy()
    if sdf.empty:
        return _empty
    if leader_id is None:
        lids = sdf["leader_vehicle_id"].apply(parse_id_list)
        leader_id = lids.iloc[lids.apply(len).argmax()][0]

    most_leading_leader_id = leader_id
    ldf = df[(df["scenario_id"] == scenario_id) & (df["ID"] == leader_id)].copy()

    cf_times = mutual_cf_times(sdf["time"].values, ldf["time"].values)
    if not cf_times:
        return _empty
    if float(cf_times[-1] - cf_times[0]) < MIN_CF_DURATION_S:
        return _empty

    ldf = ldf[ldf["time"].isin(cf_times)].sort_values("time")
    sdf = sdf[sdf["time"].isin(cf_times)].sort_values("time")
    if sdf.empty or ldf.empty:
        return _empty

    merged = pd.merge(sdf, ldf, on="time", suffixes=("_f", "_l"), how="inner")
    if merged.empty:
        return _empty

    start_time = float(merged["time"].iloc[0])
    sc_id = str(merged["scenario_id_f"].iloc[0])

    lane_geom: Optional[LaneGeom] = None
    ref_lane_id = reference_lane_feature_id(merged["assigned_lane_feature_id_f"])
    if ref_lane_id is not None:
        geom = get_lane_geometry(sc_id, ref_lane_id)
        if geom is not None:
            xs, ys, s_cum = geom
            lane_f = project_xy_array_to_lane_s(
                merged["center_x_f"].values, merged["center_y_f"].values, xs, ys, s_cum
            )
            lane_l = project_xy_array_to_lane_s(
                merged["center_x_l"].values, merged["center_y_l"].values, xs, ys, s_cum
            )
            lane_geom = (ref_lane_id, xs, ys, s_cum)
            pos_f, pos_l = lane_f, lane_l
        else:
            pos_f = arc_length_xy(merged["center_x_f"].values, merged["center_y_f"].values)
            pos_l = arc_length_xy(merged["center_x_l"].values, merged["center_y_l"].values)
    else:
        pos_f = arc_length_xy(merged["center_x_f"].values, merged["center_y_f"].values)
        pos_l = arc_length_xy(merged["center_x_l"].values, merged["center_y_l"].values)

    sdf_out = pd.DataFrame(
        {
            "time": merged["time"] - start_time,
            pos: pos_f,
            "speed-kf": merged["speed-kf_f"],
            "center_x": merged["center_x_f"],
            "center_y": merged["center_y_f"],
            "velocity_x": merged["velocity_x_f"],
            "velocity_y": merged["velocity_y_f"],
            "scenario_id": merged["scenario_id_f"],
            "length": merged["length_f"],
            "ref_lane_feature_id": ref_lane_id,
        }
    )
    ldf_out = pd.DataFrame(
        {
            "time": merged["time"] - start_time,
            pos: pos_l,
            "speed-kf": merged["speed-kf_l"],
            "center_x": merged["center_x_l"],
            "center_y": merged["center_y_l"],
            "velocity_x": merged["velocity_x_l"],
            "velocity_y": merged["velocity_y_l"],
            "length": merged["length_l"],
        }
    )
    duration = float(sdf_out["time"].iloc[-1] - sdf_out["time"].iloc[0])
    return sdf_out, ldf_out, duration, start_time, lane_geom


# ---------------------------------------------------------------------------
# IDM + GA (same structure as IDM_CF_Calibration.py)
# ---------------------------------------------------------------------------
def idm_acceleration(v, v_leader, gap):
    """IDM acceleration; gap is bumper-to-bumper spacing (m), aligned with coop_models.idm_accel."""
    max_v = 40.0
    v = max(float(v), 0.0)
    v_leader = float(v_leader)
    gap = max(float(gap), 1e-3)
    v_des = max(min(float(v0), max_v), 1e-3)
    ab = max(float(a) * float(b), 1e-6)
    dv = v - v_leader
    s_star = float(so) + v * float(T) + v * dv / (2.0 * np.sqrt(ab))
    acceleration = float(a) * (1.0 - (v / v_des) ** float(delta) - (s_star / gap) ** 2)
    if not np.isfinite(acceleration):
        acceleration = 0.0
    return float(acceleration)


def idm_bumper_gap(leader_s: float, follower_s: float) -> float:
    """Net spacing between bumpers along the lane."""
    return float(leader_s) - float(follower_s) - 0.5 * (follower_length_m + leader_length_m)


def simulate_car_following(params):
    """
    Closed-loop follower roll-out for one car-following event.

    - Follower: IDM integrates position/speed forward from the observed IC at t=0 only.
      State at step i uses sim position[i-1] and sim speed[i-1] — never reset to observed
      follower position/speed after t=0.
    - Leader: observed lane distance and speed at each step (open-loop exogenous input).
    - Gap for IDM: bumper-to-bumper spacing from lane positions and vehicle lengths.
    """
    global T, a, b, v0, so, delta
    T, a, b, v0, so, delta = clip_idm_params(params)

    num_steps = len(target_position)
    position = np.zeros(num_steps)
    speed = np.zeros(num_steps)
    acl = np.zeros(num_steps)

    # Initial conditions only — no per-step reset to observed follower state
    position[0] = float(sdf.iloc[0][pos])
    speed[0] = float(sdf.iloc[0]["speed-kf"])
    acl[0] = 0.0

    for i in range(1, num_steps):
        dt = time_step
        leader_v = leader_speed[i - 1]
        gap = idm_bumper_gap(leader_position[i - 1], position[i - 1])
        acceleration = idm_acceleration(speed[i - 1], leader_v, gap)
        acl[i] = acceleration
        speed[i] = max(0.0, speed[i - 1] + acceleration * dt)
        position[i] = position[i - 1] + speed[i - 1] * dt + 0.5 * acceleration * (dt ** 2)

    return position, speed, acl


def fitness(params):
    weight_position = 1.0
    weight_speed = 1.0

    sim_position, sim_speed, _acl = simulate_car_following(params)
    # Skip t=0 (IC fixed); compare roll-out only
    diff_position = np.asarray(sim_position[1:]) - np.asarray(target_position[1:])
    diff_speed = np.asarray(sim_speed[1:]) - np.asarray(target_speed[1:])
    if len(diff_position) == 0:
        return 0.0, {"Total Difference": float("inf"), "NRMSE": float("inf")}

    pos_span = max(float(np.ptp(target_position)), 1.0)
    spd_span = max(float(np.ptp(target_speed)), 0.5)
    rmse_position = float(np.sqrt(np.mean(diff_position ** 2)))
    rmse_speed = float(np.sqrt(np.mean(diff_speed ** 2)))
    nrmse_position = rmse_position / pos_span
    nrmse_speed = rmse_speed / spd_span
    total_diff = weight_position * nrmse_position + weight_speed * nrmse_speed
    mse_position = rmse_position ** 2
    mse_speed = rmse_speed ** 2
    mse = mse_position + mse_speed
    rmse = float(np.sqrt(mse))
    mae_position = float(np.mean(np.abs(diff_position)))
    mae_speed = float(np.mean(np.abs(diff_speed)))
    mae = mae_position + mae_speed

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_position = float(np.nanmean(np.abs(diff_position / np.asarray(target_position[1:])))) * 100
        mape_speed = float(np.nanmean(np.abs(diff_speed / np.asarray(target_speed[1:])))) * 100
    mape = (mape_position + mape_speed) / 2.0

    nrmse = total_diff / (weight_position + weight_speed)
    sse = float(np.sum(diff_position ** 2) + np.sum(diff_speed ** 2))
    ss_tot_position = float(np.sum((np.asarray(target_position[1:]) - np.mean(target_position[1:])) ** 2))
    ss_tot_speed = float(np.sum((np.asarray(target_speed[1:]) - np.mean(target_speed[1:])) ** 2))
    r2_position = 1.0 - (np.sum(diff_position ** 2) / ss_tot_position) if ss_tot_position > 0 else np.nan
    r2_speed = 1.0 - (np.sum(diff_speed ** 2) / ss_tot_speed) if ss_tot_speed > 0 else np.nan
    r2 = (r2_position + r2_speed) / 2.0

    fitness_value = 1.0 / (total_diff + 1e-5)

    error_metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "NRMSE": nrmse,
        "SSE": sse,
        "R-squared": r2,
        "Total Difference": total_diff,
    }
    return fitness_value, error_metrics


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = clip_idm_params(parent1[:crossover_point] + parent2[crossover_point:])
    child2 = clip_idm_params(parent2[:crossover_point] + parent1[crossover_point:])
    return child1, child2


def mutate(child):
    for i, (lo, hi) in enumerate(IDM_PARAM_RANGES):
        if random.random() < mutation_rate:
            span = float(hi - lo)
            child[i] += random.uniform(-0.1, 0.1) * span
    return clip_idm_params(child)


def _seed_idm_population() -> List[list]:
    """Literature-like seeds plus v0 informed by observed follower speed."""
    seeds = [
        [1.5, 1.5, 2.0, 15.0, 2.0, 4.0],
        [1.0, 1.0, 1.5, 10.0, 2.0, 4.0],
        [2.0, 2.0, 2.5, 20.0, 2.5, 4.0],
    ]
    if target_speed is not None and len(target_speed) > 1:
        v_med = float(np.median(target_speed))
        v_p90 = float(np.percentile(target_speed, 90))
        seeds.append([1.3, 1.2, 2.0, np.clip(v_med, *V0_RANGE), 2.0, 4.0])
        seeds.append([1.8, 1.8, 2.2, np.clip(v_p90, *V0_RANGE), 2.0, 4.0])
    return [clip_idm_params(s) for s in seeds]


def genetic_algorithm():
    population = _seed_idm_population()
    while len(population) < population_size:
        population.append(
            clip_idm_params(
                [random.uniform(*r) for r in IDM_PARAM_RANGES]
            )
        )

    best_error = float("inf")
    best_individual = None
    best_metrics = None

    for _generation in range(num_generations):
        fitness_and_errors = [fitness(ind) for ind in population]
        population_sorted = sorted(zip(population, fitness_and_errors), key=lambda x: x[1][0], reverse=True)
        population = [ind for ind, _ in population_sorted]

        current_best_error = population_sorted[0][1][1]["Total Difference"]
        if current_best_error < best_error:
            best_error = current_best_error
            best_individual = population_sorted[0][0]
            best_metrics = population_sorted[0][1][1]

        parents = population[: len(population) // 2]
        children = []
        while len(children) < (population_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([mutate(child1), mutate(child2)])
        population = parents + children[: population_size - len(parents)]

    return clip_idm_params(best_individual), best_error, best_metrics


def plot_simulation(
    timex_,
    leader_position_,
    target_position_,
    sim_position_,
    leader_speed_,
    target_speed_,
    sim_speed_,
    follower_id,
    leader_id,
    scenario_id,
    save_dir,
    outname,
    sdf_obs: Optional[pd.DataFrame] = None,
    ldf_obs: Optional[pd.DataFrame] = None,
    lane_geom: Optional[LaneGeom] = None,
):
    """
    Validation plots after closed-loop follower roll-out.

    Lane-distance and speed panels show the actual IDM state variable vs observed data.
    X–Y / center_x / center_y for the simulated follower use the reference lane polyline
    when available; otherwise headings-based arc reconstruction (visualization only).
    """
    sc_tag = str(scenario_id)[:8]
    fig_title = f"FID: {follower_id}, LID: {leader_id}, scenario: {sc_tag}"
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(fig_title, fontsize=12)
    ax_xy, ax_arc, ax_spd, ax_tx, ax_ty, ax_err = axes.ravel()
    t = np.asarray(timex_)
    sim_s = np.asarray(sim_position_)
    obs_s = np.asarray(target_position_)
    on_lane = lane_geom is not None
    s_label = "lane distance (m)" if on_lane else "path arc length (m)"
    s_title = "Lane distance vs time" if on_lane else "Arc length vs time"
    err_title = "Follower lane-s error" if on_lane else "Follower arc error"

    ax_arc.plot(t, leader_position_, label="Leader (obs)")
    ax_arc.plot(t, obs_s, label="Follower (obs)")
    ax_arc.plot(t, sim_s, "--", label="Follower (sim, closed-loop)")
    ax_arc.set_xlabel("time (s)")
    ax_arc.set_ylabel(s_label)
    ax_arc.set_title(s_title)
    ax_arc.legend(fontsize=8)
    ax_arc.grid(True)

    ax_err.plot(t, sim_s - obs_s, color="C3", label="error (sim − obs)")
    ax_err.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel(f"{'lane-s' if on_lane else 'arc'} error (m)")
    ax_err.set_title(err_title)
    ax_err.legend(fontsize=8)
    ax_err.grid(True)

    if sdf_obs is not None and ldf_obs is not None and not sdf_obs.empty and not ldf_obs.empty:
        obs_fx = sdf_obs["center_x"].to_numpy()
        obs_fy = sdf_obs["center_y"].to_numpy()
        obs_lx = ldf_obs["center_x"].to_numpy()
        obs_ly = ldf_obs["center_y"].to_numpy()
        if on_lane:
            _ref_id, xs, ys, s_cum = lane_geom
            sim_fx, sim_fy = lane_s_to_xy(sim_s, xs, ys, s_cum)
            sim_label = "Follower (sim, on lane)"
        else:
            sim_fx, sim_fy = sim_xy_from_arc(
                sim_s,
                obs_fx,
                obs_fy,
                sdf_obs["velocity_x"].to_numpy(),
                sdf_obs["velocity_y"].to_numpy(),
            )
            sim_label = "Follower (sim, from arc)"

        plot_map_on_ax(ax_xy, get_map_features_for_scenario(scenario_id))
        ax_xy.plot(obs_lx, obs_ly, label="Leader (obs)", zorder=3, lw=2.0)
        ax_xy.plot(obs_fx, obs_fy, label="Follower (obs)", zorder=3, lw=2.0)
        ax_xy.plot(sim_fx, sim_fy, "--", label=sim_label, zorder=4, lw=2.0)
        ax_xy.set_xlabel("center_x (m)")
        ax_xy.set_ylabel("center_y (m)")
        ax_xy.set_title("X–Y plan view")
        ax_xy.axis("equal")
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True)

        ax_tx.plot(t, obs_lx, label="Leader (obs)")
        ax_tx.plot(t, obs_fx, label="Follower (obs)")
        ax_tx.plot(t, sim_fx, "--", label=sim_label)
        ax_tx.set_xlabel("time (s)")
        ax_tx.set_ylabel("center_x (m)")
        ax_tx.set_title("center_x vs time")
        ax_tx.margins(y=0.5)
        ax_tx.legend(fontsize=8)
        ax_tx.grid(True)

        ax_ty.plot(t, obs_ly, label="Leader (obs)")
        ax_ty.plot(t, obs_fy, label="Follower (obs)")
        ax_ty.plot(t, sim_fy, "--", label=sim_label)
        ax_ty.set_xlabel("time (s)")
        ax_ty.set_ylabel("center_y (m)")
        ax_ty.set_title("center_y vs time")
        ax_ty.margins(y=0.5)
        ax_ty.legend(fontsize=8)
        ax_ty.grid(True)
    else:
        for ax, name in zip(
            (ax_xy, ax_tx, ax_ty),
            ("X–Y plan view", "center_x vs time", "center_y vs time"),
        ):
            ax.text(0.5, 0.5, "No center_x/center_y in observation data", ha="center", va="center")
            ax.set_title(name)

    ax_spd.plot(t, leader_speed_, label="Leader (obs)")
    ax_spd.plot(t, target_speed_, label="Follower (obs)")
    ax_spd.plot(t, sim_speed_, "--", label="Follower (sim, closed-loop)")
    ax_spd.set_xlabel("time (s)")
    ax_spd.set_ylabel("speed (m/s)")
    ax_spd.set_title("Speed vs time")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True)

    plot_filename = os.path.join(
        save_dir,
        f"{outname}_FID_{follower_id}_LID_{leader_id}_sc_{sc_tag}.png",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table (follower-leader type means)
# ---------------------------------------------------------------------------
def _load_param_frames(param_files: Optional[List[str]] = None) -> pd.DataFrame:
    if param_files is None:
        pattern = os.path.join(RESULTS_DIR, "IDM_Params_Waymo_*.csv")
        param_files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(p) for p in param_files if os.path.isfile(p)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _enrich_params_types_from_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill follower/leader types from motion CSVs when missing (summary-only)."""
    if df.empty:
        return df
    out = df.copy()
    needs_follower = "follower_type" not in out.columns
    needs_leader = "leader_type" not in out.columns

    if needs_follower or needs_leader:
        combined, _, _ = build_combined_dataframe(refresh_waymo_datasets())
        lookup = build_vehicle_type_lookup(combined) if not combined.empty else {}
        if needs_follower and "Follower_ID" in out.columns and "scenario_id" in out.columns:
            out["follower_type"] = out.apply(
                lambda r: lookup.get((str(r["scenario_id"]), int(r["Follower_ID"])), np.nan),
                axis=1,
            )
        elif needs_follower and "source" in out.columns:
            out["follower_type"] = out["source"].map(SOURCE_TO_FOLLOWER_TYPE)

        if needs_leader and "Leader_ID" in out.columns and "scenario_id" in out.columns:
            out["leader_type"] = out.apply(
                lambda r: lookup.get((str(r["scenario_id"]), int(r["Leader_ID"])), np.nan),
                axis=1,
            )

    if "follower_type" in out.columns:
        out["follower_type"] = out["follower_type"].map(normalize_follower_type)
    if "leader_type" in out.columns:
        out["leader_type"] = out["leader_type"].map(normalize_follower_type)
    return out


def _prepare_params_for_summary(all_params: pd.DataFrame) -> pd.DataFrame:
    """Ensure follower_type column exists (AV / SV / HV)."""
    df = _enrich_params_types_from_trajectories(all_params.copy())
    if "follower_type" in df.columns:
        df["follower_type"] = df["follower_type"].map(normalize_follower_type)
    elif "fl_combo" in df.columns:
        # Legacy per-event files: use follower half of S-S style combos
        df["follower_type"] = df["fl_combo"].astype(str).str.split("-").str[0].map(normalize_follower_type)
    elif "source" in df.columns:
        df["follower_type"] = df["source"].map(SOURCE_TO_FOLLOWER_TYPE)
    return df


def _summary_combo_value(subset: pd.DataFrame, param_col: str) -> object:
    if subset.empty or param_col not in subset.columns:
        return ""
    return round(float(subset[param_col].mean()), 2)


def build_idm_summary_table(param_files: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build IDM summary with mean parameters per follower type (AV, SV, HV)
    plus a pooled Vehicle-Vehicle column.
    """
    summary_cols = ["Model", "Parameter", "Range"] + FOLLOWER_TYPE_ORDER + ["Vehicle-Vehicle"]
    all_params = _prepare_params_for_summary(_load_param_frames(param_files))
    if all_params.empty:
        return pd.DataFrame(columns=summary_cols)

    rows: List[dict] = []
    for col, label, rng in IDM_PARAM_SUMMARY:
        row: dict = {
            "Model": "IDM",
            "Parameter": label,
            "Range": _format_range(rng),
        }
        for ftype in FOLLOWER_TYPE_ORDER:
            sub = all_params[all_params["follower_type"] == ftype]
            row[ftype] = _summary_combo_value(sub, col)
        row["Vehicle-Vehicle"] = round(float(all_params[col].mean()), 2)
        rows.append(row)

    count_row: dict = {
        "Model": "IDM",
        "Parameter": "count",
        "Range": "-",
        "Vehicle-Vehicle": int(len(all_params)),
    }
    for ftype in FOLLOWER_TYPE_ORDER:
        count_row[ftype] = int((all_params["follower_type"] == ftype).sum())
    rows.append(count_row)
    return pd.DataFrame(rows, columns=summary_cols)

EventSpec = Tuple[int, int, str]  # follower_id, leader_id, scenario_id prefix


def parse_event_specs(spec: str) -> List[EventSpec]:
    """Parse 'FID:LID:scenario_prefix' comma-separated event selectors."""
    out: List[EventSpec] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(
                f"Invalid --events entry {part!r}; expected FID:LID:scenario_prefix"
            )
        out.append((int(bits[0]), int(bits[1]), bits[2].strip()))
    return out


def resolve_event_specs(
    specs: List[EventSpec],
    combined_df: pd.DataFrame,
    groups: Dict[str, List[Tuple[int, str]]],
) -> List[Tuple[str, int, int, str]]:
    """Map each spec to (group_name, follower_id, leader_id, full scenario_id)."""
    key_to_group: Dict[Tuple[int, str], str] = {}
    for gname, keys in groups.items():
        for fid, sc in keys:
            key_to_group[(int(fid), str(sc))] = gname

    resolved: List[Tuple[str, int, int, str]] = []
    for fid, lid, sc_prefix in specs:
        sc_matches = combined_df.loc[
            (combined_df["ID"] == fid)
            & (combined_df["scenario_id"].astype(str).str.startswith(sc_prefix)),
            "scenario_id",
        ].unique()
        if len(sc_matches) == 0:
            print(f"[warn] No match for FID={fid} scenario prefix {sc_prefix!r}")
            continue
        if len(sc_matches) > 1:
            print(
                f"[warn] Multiple scenarios for FID={fid} prefix {sc_prefix!r}; "
                f"using {sc_matches[0]}"
            )
        scenario_id = str(sc_matches[0])
        group = key_to_group.get((fid, scenario_id))
        if group is None:
            print(
                f"[warn] FID={fid} scenario={scenario_id[:8]} is not a discovered "
                f"car-following event; skipping"
            )
            continue
        resolved.append((group, fid, lid, scenario_id))
    return resolved


def _merge_event_csv(path: str, new_df: pd.DataFrame, key_cols: Tuple[str, ...]) -> None:
    """Replace matching rows in an existing CSV, or create the file."""
    if new_df.empty:
        return
    if os.path.isfile(path):
        old = pd.read_csv(path)
        for _, row in new_df.iterrows():
            mask = np.ones(len(old), dtype=bool)
            for col in key_cols:
                mask &= old[col] == row[col]
            old = old.loc[~mask]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)


def cap_events_total(
    groups: Dict[str, List[Tuple[int, str]]],
    max_total: int,
) -> Dict[str, List[Tuple[int, str]]]:
    """Limit total calibration events across AV / SV / HV follower groups."""
    capped: Dict[str, List[Tuple[int, str]]] = {}
    remaining = int(max_total)
    for gname in GROUP_ORDER:
        items = groups.get(gname, [])
        if remaining <= 0:
            capped[gname] = []
            continue
        take = min(len(items), remaining)
        capped[gname] = items[:take]
        remaining -= take
    return capped


def _format_range(rng: Tuple[float, float]) -> str:
    lo, hi = rng
    if float(lo).is_integer() and float(hi).is_integer():
        return f"({int(lo)}, {int(hi)})"
    return f"({lo}, {hi})"


def write_idm_summary_table(
    save_dir: str = RESULTS_DIR,
    param_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Save IDM_Summary_Waymo.csv and print the table."""
    if param_files is None:
        pattern = os.path.join(save_dir, "IDM_Params_Waymo_*.csv")
        param_files = sorted(glob.glob(pattern))

    summary = build_idm_summary_table(param_files)
    out_path = os.path.join(save_dir, "IDM_Summary_Waymo.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary table: {out_path}")
    if not summary.empty:
        console = summary.copy()
        console["Parameter"] = console["Parameter"].str.replace("δ", "delta")
        print(console.to_string(index=False))
    else:
        print("No calibrated parameter files found for summary.")
    return summary


def plot_idm_param_distributions(
    save_dir: str = RESULTS_DIR,
    param_files: Optional[List[str]] = None,
) -> Optional[str]:
    """Histograms of calibrated IDM parameters (AV / SV / HV) from parameter CSVs."""
    all_params = _prepare_params_for_summary(_load_param_frames(param_files))
    if all_params.empty:
        print("No parameter data for distribution plots.")
        return None

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    colors = {"AV": "#e53935", "SV": "#1e88e5", "HV": "#43a047"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, label, _rng) in zip(axes.ravel(), IDM_PARAM_SUMMARY):
        any_data = False
        for ftype in FOLLOWER_TYPE_ORDER:
            sub = all_params.loc[all_params["follower_type"] == ftype, col].dropna()
            if sub.empty:
                continue
            any_data = True
            bins = min(25, max(5, len(sub) // 2))
            ax.hist(
                sub,
                bins=bins,
                alpha=0.55,
                label=f"{ftype} (n={len(sub)})",
                color=colors[ftype],
                density=True,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"IDM {label.replace('δ', 'delta')}")
        if any_data:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibrated IDM parameter distributions by follower type", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(plots_dir, "IDM_Params_Waymo_distribution.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter distribution plot: {out_path}")
    return out_path


def write_idm_summary_and_plots(
    save_dir: str = RESULTS_DIR,
    param_files: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Rebuild summary CSV and parameter distribution figure."""
    summary = write_idm_summary_table(save_dir=save_dir, param_files=param_files)
    dist_path = plot_idm_param_distributions(save_dir=save_dir, param_files=param_files)
    return summary, dist_path


# ---------------------------------------------------------------------------
# Extraction diagnostics (no GA)
# ---------------------------------------------------------------------------
def run_extraction_diagnostics(
    datasets: Optional[Dict[str, str]] = None,
    save_dir: str = PLOTS_DIR,
    max_events_per_group: int = 5,
) -> None:
    """Plot observed leader/follower x,y and lane-distance kinematics without running GA."""
    datasets = datasets or refresh_waymo_datasets()
    if not datasets:
        print(f"No Waymo trajectory CSVs found in {DATASETS_DIR}")
        return
    _print_waymo_dataset_summary(datasets)
    os.makedirs(save_dir, exist_ok=True)

    combined_df, _scenario_src, _global_run = build_combined_dataframe(datasets)
    groups = generate_waymo_vehicle_groups(datasets)

    for group_name, event_keys in groups.items():
        for follower_id, scenario_id in event_keys[:max_events_per_group]:
            sdf, ldf, duration, _start_time, lane_geom = extract_subject_and_leader_data(
                combined_df, follower_id, scenario_id
            )
            leader_id = int(most_leading_leader_id) if most_leading_leader_id is not None else -1
            if sdf.empty or ldf.empty or duration < MIN_CF_DURATION_S:
                continue

            time_step = TIME_STEP_S
            total_time = len(sdf) * time_step
            num_steps = round(total_time / time_step)
            timex = np.linspace(0, total_time, num_steps)

            plot_simulation(
                timex,
                ldf[pos].tolist(),
                sdf[pos].tolist(),
                sdf[pos].tolist(),  # obs arc length as pseudo-sim
                ldf["speed-kf"].tolist(),
                sdf["speed-kf"].tolist(),
                sdf["speed-kf"].tolist(),
                follower_id,
                leader_id,
                scenario_id,
                save_dir,
                f"Diag_{group_name}",
                sdf_obs=sdf,
                ldf_obs=ldf,
                lane_geom=lane_geom,
            )
            print(
                f"Diagnostic plot: {group_name} FID={follower_id} "
                f"LID={leader_id} scenario={scenario_id[:8]}"
            )


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------
def run_calibration(
    datasets: Optional[Dict[str, str]] = None,
    save_dir: str = RESULTS_DIR,
    max_events_per_group: Optional[int] = None,
    max_events_total: Optional[int] = None,
    skip_existing: bool = True,
    save_plots: bool = True,
    event_specs: Optional[List[EventSpec]] = None,
) -> None:
    global sdf, ldf, total_time, time_step, timex
    global leader_position, leader_speed, target_position, target_speed
    global follower_length_m, leader_length_m

    datasets = datasets or refresh_waymo_datasets()
    if not datasets:
        print(f"No Waymo trajectory CSVs found in {DATASETS_DIR}")
        return
    _print_waymo_dataset_summary(datasets)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    groups = generate_waymo_vehicle_groups(datasets)
    combined_df, _scenario_src, global_run = build_combined_dataframe(datasets)
    type_lookup = build_vehicle_type_lookup(combined_df)

    selected_events: Optional[Dict[str, List[Tuple[int, int, str]]]] = None
    if event_specs:
        resolved = resolve_event_specs(event_specs, combined_df, groups)
        if not resolved:
            print("No matching car-following events for --events")
            return
        selected_events = {}
        for group_name, follower_id, leader_id, scenario_id in resolved:
            selected_events.setdefault(group_name, []).append(
                (follower_id, leader_id, scenario_id)
            )
        print(f"Scheduled selective calibration: {len(resolved)} event(s)")
        for group_name, items in selected_events.items():
            for follower_id, leader_id, scenario_id in items:
                print(
                    f"  {group_name} FID={follower_id} LID={leader_id} "
                    f"scenario={scenario_id[:8]}"
                )
    else:
        print(
            f"Waymo car-following events (overlap >= {MIN_CF_DURATION_S}s):"
        )
        for gname, items in groups.items():
            print(f"  {gname}: {len(items)} follower events")

        if max_events_total is not None:
            groups = cap_events_total(groups, max_events_total)
            n_scheduled = sum(len(v) for v in groups.values())
            print(f"Scheduled for calibration (max-total={max_events_total}): {n_scheduled} events")
            for gname in GROUP_ORDER:
                if groups.get(gname):
                    print(f"  {gname}: {len(groups[gname])}")

    loop_groups = selected_events if selected_events is not None else groups

    for group_name, event_keys in loop_groups.items():
        if not event_keys:
            print(f"[skip] {group_name}: no events")
            continue

        outname = f"IDM_Params_{group_name}"
        output_csv_path = os.path.join(save_dir, f"{outname}.csv")
        if selected_events is None and skip_existing and os.path.exists(output_csv_path):
            print(f"[skip] {output_csv_path} already exists")
            continue

        if selected_events is None and max_events_per_group is not None:
            event_keys = event_keys[: max_events_per_group]

        params_list: List[list] = []
        all_simulations_list: List[pd.DataFrame] = []
        best_metrics = None

        for event in event_keys:
            if selected_events is not None:
                follower_id, leader_id, scenario_id = event
            else:
                follower_id, scenario_id = event
                leader_id = None
            sdf, ldf, duration, start_time, lane_geom = extract_subject_and_leader_data(
                combined_df, follower_id, scenario_id, leader_id=leader_id
            )
            if leader_id is None:
                leader_id = int(most_leading_leader_id) if most_leading_leader_id is not None else -1
            run_index = int(global_run[scenario_id])
            print(
                f"Processing {group_name} FID={follower_id} LID={leader_id} "
                f"scenario={scenario_id[:8]} run={run_index}"
            )

            if sdf.empty or ldf.empty or duration < MIN_CF_DURATION_S:
                print(f"  -> CF overlap < {MIN_CF_DURATION_S}s; skip")
                continue

            if len(sdf) < 3:
                print("  -> too few aligned steps; skip")
                continue
            time_step = TIME_STEP_S
            total_time = len(sdf) * time_step
            num_steps = round(total_time / time_step)
            timex = np.linspace(0, total_time, num_steps)
            leader_position = ldf[pos].tolist()
            leader_speed = ldf["speed-kf"].tolist()
            target_position = sdf[pos].tolist()
            target_speed = sdf["speed-kf"].tolist()
            follower_length_m = float(sdf["length"].median())
            leader_length_m = float(ldf["length"].median())

            best_params, best_error, best_metrics = genetic_algorithm()
            if best_params is None or best_metrics is None:
                print("  -> GA failed; skip")
                continue

            follower_type = type_lookup.get((scenario_id, follower_id), "SV")
            leader_type = type_lookup.get((scenario_id, leader_id), "SV")

            params_list.append(
                [
                    follower_id,
                    leader_id,
                    run_index,
                    scenario_id,
                    duration,
                    follower_type,
                    leader_type,
                ]
                + list(best_params)
                + [best_error]
                + list(best_metrics.values())
            )

            sim_position, sim_speed, acl = simulate_car_following(best_params)
            if save_plots:
                plot_simulation(
                    timex,
                    leader_position,
                    target_position,
                    sim_position,
                    leader_speed,
                    target_speed,
                    sim_speed,
                    follower_id,
                    leader_id,
                    scenario_id,
                    PLOTS_DIR,
                    outname,
                    sdf_obs=sdf,
                    ldf_obs=ldf,
                    lane_geom=lane_geom,
                )

            sim_df = pd.DataFrame(
                {
                    "ID": follower_id,
                    "run-index": run_index,
                    "scenario_id": scenario_id,
                    "Leader_ID": leader_id,
                    "time": np.round(timex + start_time, 1),
                    pos: sim_position,
                    "speed-kf": sim_speed,
                    "sim_acceleration": acl,
                    "source_group": group_name,
                }
            )
            all_simulations_list.append(sim_df)

        if params_list and best_metrics is not None:
            metrics_names = list(best_metrics.keys())
            columns = (
                [
                    "Follower_ID",
                    "Leader_ID",
                    "Run_Index",
                    "scenario_id",
                    "Duration",
                    "follower_type",
                    "leader_type",
                    "T",
                    "a",
                    "b",
                    "v0",
                    "so",
                    "delta",
                    "Error",
                ]
                + metrics_names
            )
            params_df = pd.DataFrame(params_list, columns=columns)
            params_df["source"] = group_name
            if selected_events is not None:
                _merge_event_csv(
                    output_csv_path,
                    params_df,
                    ("Follower_ID", "Leader_ID", "scenario_id"),
                )
            else:
                params_df.to_csv(output_csv_path, index=False)
            print(f"Saved parameters: {output_csv_path}")

        if all_simulations_list:
            sim_path = os.path.join(save_dir, f"IDM_Simulated_{group_name}.csv")
            sim_df = pd.concat(all_simulations_list, ignore_index=True)
            if selected_events is not None:
                _merge_event_csv(
                    sim_path,
                    sim_df,
                    ("ID", "Leader_ID", "scenario_id"),
                )
            else:
                sim_df.to_csv(sim_path, index=False)
            print(f"Saved simulated trajectories: {sim_path}")

    write_idm_summary_and_plots(save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Waymo IDM car-following calibration")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap events per follower-type group (Waymo_AV, Waymo_SV, Waymo_HV each)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Cap total events across all groups (e.g. 100)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override GA generations (default 80)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if output CSV exists",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help=(
            "Comma-separated FID:LID:scenario_prefix selectors "
            "(e.g. 122:10:edb3253f,160:108:6d19bfcf). Merges into existing CSVs."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot observed x/y and arc-length kinematics (no GA calibration)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-event PNG plots during calibration (faster for large runs)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild IDM_Summary_Waymo.csv from existing parameter CSVs",
    )
    args = parser.parse_args()

    global num_generations
    if args.generations is not None:
        num_generations = int(args.generations)

    if args.summary_only:
        write_idm_summary_and_plots()
    elif args.plot_only:
        run_extraction_diagnostics(max_events_per_group=args.max_events or 5)
    else:
        event_specs = parse_event_specs(args.events) if args.events else None
        run_calibration(
            max_events_per_group=args.max_events,
            max_events_total=args.max_total,
            skip_existing=not args.overwrite,
            save_plots=not args.no_plots,
            event_specs=event_specs,
        )


if __name__ == "__main__":
    main()
