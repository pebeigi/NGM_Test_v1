"""
Waymo Open Motion — IDM car-following calibration (GA).

Mirrors the workflow in ../1.1.2 - Car-Following Parametric Analysis/IDM_CF_Calibration.py
but reads pre-processed Waymo leader–follower CSVs from ``0 - Datasets/`` at the repo root.

Input columns (per file):
  scenario_id, time, vehicle_id, object_type, center_x/y, velocity_x/y, length,
  lane_changing, leader_vehicle_id, is_sdc, assigned_lane_feature_id

Calibration: skip cases where stable leader–follower overlap lasts less than MIN_CF_DURATION_S.
Kinematics: speed from velocity components; 1D position = distance along the follower's
  assigned lane polyline (map CSV + assigned_lane_feature_id). Leader is projected onto
  the same lane. Falls back to path arc length if map geometry is missing.
Simulation: closed-loop follower roll-out (IC at t=0 only; IDM integrates forward).
  Leader trajectory stays observed (open-loop).

Vehicle groups (same S / L / A convention as freeway calibration):
  Waymo_S — passenger-sized vehicles (length < LARGE_LENGTH_M)
  Waymo_L — long vehicles (length >= LARGE_LENGTH_M)
  Waymo_A — ego / SDC tracks (is_sdc == True)

Outputs under ./Results/:
  IDM_Params_Waymo_{S,L,A}.csv
  IDM_Simulated_Waymo_{S,L,A}.csv
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
from ngm_paths import DATASETS_DIR, REPO_ROOT

RESULTS_DIR = os.path.join(THIS_DIR, "Results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

DATA_FILES = {
    "Waymo_196": os.path.join(
        DATASETS_DIR, "March2023waymo_scenario_lane_leader_follower_assigned_196_data.csv"
    ),
    "Waymo_197": os.path.join(
        DATASETS_DIR, "March2023waymo_scenario_lane_leader_follower_assigned_197_data.csv"
    ),
    "Waymo_198": os.path.join(
        DATASETS_DIR, "March2023waymo_scenario_lane_leader_follower_assigned_198_data.csv"
    ),
    "Waymo_199": os.path.join(
        DATASETS_DIR, "March2023waymo_scenario_lane_leader_follower_assigned_199_data.csv"
    ),
}

MAP_FILES = {
    "Waymo_196": os.path.join(DATASETS_DIR, "March2023waymo_map_features_196_data.csv"),
    "Waymo_197": os.path.join(DATASETS_DIR, "March2023waymo_map_features_197_data.csv"),
    "Waymo_198": os.path.join(DATASETS_DIR, "March2023waymo_map_features_198_data.csv"),
    "Waymo_199": os.path.join(DATASETS_DIR, "March2023waymo_map_features_199_data.csv"),
}

_scenario_source: Dict[str, str] = {}
_map_scenario_cache: Dict[str, pd.DataFrame] = {}
_lane_geom_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_loaded_map_tags: set = set()

# Returned by extract_subject_and_leader_data when lane projection succeeds:
# (ref_lane_feature_id, polyline_x, polyline_y, cumulative_s_along_polyline)
LaneGeom = Tuple[int, np.ndarray, np.ndarray, np.ndarray]

population_size = 40
num_generations = 80
mutation_rate = 0.1

MIN_CF_DURATION_S = 15.0  # skip if stable leader–follower overlap is shorter than this (s)
MIN_LEADER_SPEED_MPS = 1.0
LARGE_LENGTH_M = 6.0
TIME_STEP_S = 0.1
CONTIGUOUS_DT_MAX_S = 0.2

# IDM GA parameter ranges (same as IDM_CF_Calibration.py, wider v0 lower bound)
T_RANGE = (0.5, 2.5)
A_RANGE = (0.3, 3.0)
B_RANGE = (0.5, 3.0)
V0_RANGE = (1.0, 35.0)  # desired speed: 2–35 m/s (reference uses 5–35)
SO_RANGE = (1.0, 5.0)
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
GROUP_ORDER = ["Waymo_S", "Waymo_L", "Waymo_A"]
SOURCE_TO_TYPE = {"Waymo_S": "S", "Waymo_L": "L", "Waymo_A": "A"}
# Follower-leader type pairs (same layout as freeway IDM summary table)
FL_COMBO_ORDER = [
    "S-S", "S-L", "S-A",
    "L-S", "L-L", "L-A",
    "A-S", "A-L", "A-A",
]

# Simulation globals (set per event before GA, same pattern as IDM_CF_Calibration.py)
pos = "lane-s"  # distance along reference lane polyline (m)
T = a = b = v0 = so = delta = None
most_leading_leader_id = None
sdf = ldf = None
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
    """S = passenger-sized, L = long (>= LARGE_LENGTH_M), A = SDC / ego."""
    if is_sdc:
        return "A"
    if float(length_median) >= LARGE_LENGTH_M:
        return "L"
    return "S"


def build_vehicle_type_lookup(df: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    """Map (scenario_id, vehicle_id) -> S / L / A."""
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
    Return Waymo_{S,L,A} lists of (follower_id, scenario_id).

    scenario_id is the stable key — run-index is only unique within a single CSV file and
    must not be used when the four Waymo files are combined.
    """
    groups: Dict[str, set] = {
        "Waymo_S": set(),
        "Waymo_L": set(),
        "Waymo_A": set(),
    }
    for _tag, path in datasets.items():
        if not os.path.isfile(path):
            print(f"[skip] missing file: {path}")
            continue
        df = load_waymo_cf_dataframe(path)
        for ev in discover_car_following_events(df):
            key = (ev["follower_id"], ev["scenario_id"])
            if ev["is_sdc"]:
                groups["Waymo_A"].add(key)
            elif ev["length_median"] >= LARGE_LENGTH_M:
                groups["Waymo_L"].add(key)
            else:
                groups["Waymo_S"].add(key)

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
def idm_acceleration(v, v_leader, s):
    max_v = 40.0
    max_s = 1000.0
    v = max(float(v), 0.0)
    v_leader = float(v_leader)
    s = max(float(s), 0.5)
    v_des = max(min(float(v0), max_v), 1e-3)
    ab_prod = max(float(a) * float(b), 1e-12)
    interaction = v * float(T) + (v * (v - v_leader)) / (2.0 * np.sqrt(ab_prod))
    s_star = float(so) + max(0.0, interaction)
    acceleration = float(a) * (
        1.0 - (v / v_des) ** float(delta) - (s_star / min(s, max_s)) ** 2
    )
    if not np.isfinite(acceleration):
        acceleration = 0.0
    return float(acceleration)


def simulate_car_following(params):
    """
    Closed-loop follower roll-out for one car-following event.

    - Follower: IDM integrates position/speed forward from the observed IC at t=0 only.
      State at step i uses sim position[i-1] and sim speed[i-1] — never reset to observed
      follower position/speed after t=0.
    - Leader: observed lane distance and speed at each step (open-loop exogenous input).
    - Gap for IDM: leader_lane_s[i-1] - follower_sim_lane_s[i-1].
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
        gap = leader_position[i - 1] - position[i - 1]
        acceleration = idm_acceleration(speed[i - 1], leader_v, gap)
        acl[i] = acceleration
        speed[i] = max(0.0, speed[i - 1] + acceleration * dt)
        position[i] = position[i - 1] + speed[i - 1] * dt + 0.5 * acceleration * (dt ** 2)

    return position, speed, acl


def fitness(params):
    weight_position = 1.0
    weight_speed = 0.5

    sim_position, sim_speed, _acl = simulate_car_following(params)
    diff_position = np.array(sim_position) - np.array(target_position)
    diff_speed = np.array(sim_speed) - np.array(target_speed)

    mse_position = np.mean(diff_position ** 2) * weight_position
    mse_speed = np.mean(diff_speed ** 2) * weight_speed
    mse = mse_position + mse_speed

    rmse_position = np.sqrt(mse_position)
    rmse_speed = np.sqrt(mse_speed)
    rmse = np.sqrt(mse)

    mae_position = np.mean(np.abs(diff_position)) * weight_position
    mae_speed = np.mean(np.abs(diff_speed)) * weight_speed
    mae = mae_position + mae_speed

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_position = np.nanmean(np.abs(diff_position / np.array(target_position))) * 100 * weight_position
        mape_speed = np.nanmean(np.abs(diff_speed / np.array(target_speed))) * 100 * weight_speed
    mape = (mape_position + mape_speed) / 2.0

    pos_span = np.max(target_position) - np.min(target_position)
    spd_span = np.max(target_speed) - np.min(target_speed)
    nrmse_position = rmse_position / pos_span if pos_span > 1e-6 else np.nan
    nrmse_speed = rmse_speed / spd_span if spd_span > 1e-6 else np.nan
    nrmse = (nrmse_position * weight_position + nrmse_speed * weight_speed) / (
        weight_position + weight_speed
    )

    sse_position = np.sum(diff_position ** 2) * weight_position
    sse_speed = np.sum(diff_speed ** 2) * weight_speed
    sse = sse_position + sse_speed

    ss_tot_position = np.sum((np.array(target_position) - np.mean(target_position)) ** 2)
    ss_tot_speed = np.sum((np.array(target_speed) - np.mean(target_speed)) ** 2)
    r2_position = 1.0 - (sse_position / ss_tot_position) if ss_tot_position > 0 else np.nan
    r2_speed = 1.0 - (sse_speed / ss_tot_speed) if ss_tot_speed > 0 else np.nan
    r2 = (r2_position * weight_position + r2_speed * weight_speed) / (weight_position + weight_speed)

    total_diff = np.sum(np.abs(diff_position)) * weight_position + np.sum(np.abs(diff_speed)) * weight_speed
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
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += random.uniform(-0.1, 0.1)
    return clip_idm_params(child)


def genetic_algorithm():
    population = [
        [
            random.uniform(*r)
            for r in (T_RANGE, A_RANGE, B_RANGE, V0_RANGE, SO_RANGE, DELTA_RANGE)
        ]
        for _ in range(population_size)
    ]

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
    title_suffix = f"FID: {follower_id}, LID: {leader_id}, scenario: {sc_tag}"
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
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
    ax_arc.set_title(f"{s_title} — {title_suffix}")
    ax_arc.legend(fontsize=8)
    ax_arc.grid(True)

    ax_err.plot(t, sim_s - obs_s, color="C3", label="error (sim − obs)")
    ax_err.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel(f"{'lane-s' if on_lane else 'arc'} error (m)")
    ax_err.set_title(f"{err_title} — {title_suffix}")
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
        if on_lane:
            ax_xy.plot(xs, ys, color="#ff9800", lw=2.5, alpha=0.85, zorder=2, label=f"Ref lane {_ref_id}")
        ax_xy.plot(obs_lx, obs_ly, label="Leader (obs)", zorder=3, lw=2.0)
        ax_xy.plot(obs_fx, obs_fy, label="Follower (obs)", zorder=3, lw=2.0)
        ax_xy.plot(sim_fx, sim_fy, "--", label=sim_label, zorder=4, lw=2.0)
        ax_xy.set_xlabel("center_x (m)")
        ax_xy.set_ylabel("center_y (m)")
        ax_xy.set_title(f"X–Y plan view — {title_suffix}")
        ax_xy.axis("equal")
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True)

        ax_tx.plot(t, obs_lx, label="Leader (obs)")
        ax_tx.plot(t, obs_fx, label="Follower (obs)")
        ax_tx.plot(t, sim_fx, "--", label=sim_label)
        ax_tx.set_xlabel("time (s)")
        ax_tx.set_ylabel("center_x (m)")
        ax_tx.set_title(f"center_x vs time — {title_suffix}")
        ax_tx.margins(y=0.5)
        ax_tx.legend(fontsize=8)
        ax_tx.grid(True)

        ax_ty.plot(t, obs_ly, label="Leader (obs)")
        ax_ty.plot(t, obs_fy, label="Follower (obs)")
        ax_ty.plot(t, sim_fy, "--", label=sim_label)
        ax_ty.set_xlabel("time (s)")
        ax_ty.set_ylabel("center_y (m)")
        ax_ty.set_title(f"center_y vs time — {title_suffix}")
        ax_ty.margins(y=0.5)
        ax_ty.legend(fontsize=8)
        ax_ty.grid(True)
    else:
        for ax, name in zip(
            (ax_xy, ax_tx, ax_ty),
            ("X–Y plan view", "center_x vs time", "center_y vs time"),
        ):
            ax.text(0.5, 0.5, "No center_x/center_y in observation data", ha="center", va="center")
            ax.set_title(f"{name} — {title_suffix}")

    ax_spd.plot(t, leader_speed_, label="Leader (obs)")
    ax_spd.plot(t, target_speed_, label="Follower (obs)")
    ax_spd.plot(t, sim_speed_, "--", label="Follower (sim, closed-loop)")
    ax_spd.set_xlabel("time (s)")
    ax_spd.set_ylabel("speed (m/s)")
    ax_spd.set_title(f"Speed vs time — {title_suffix}")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True)

    plot_filename = os.path.join(
        save_dir,
        f"{outname}_FID_{follower_id}_LID_{leader_id}_sc_{sc_tag}.png",
    )
    fig.tight_layout()
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
    needs_follower = "follower_type" not in df.columns
    needs_leader = "leader_type" not in df.columns
    if not needs_follower and not needs_leader:
        return df

    combined, _, _ = build_combined_dataframe(DATA_FILES)
    if combined.empty:
        return df
    lookup = build_vehicle_type_lookup(combined)

    out = df.copy()
    if needs_follower and "Follower_ID" in out.columns and "scenario_id" in out.columns:
        out["follower_type"] = out.apply(
            lambda r: lookup.get((str(r["scenario_id"]), int(r["Follower_ID"])), np.nan),
            axis=1,
        )
    elif needs_follower and "source" in out.columns:
        out["follower_type"] = out["source"].map(SOURCE_TO_TYPE)

    if needs_leader and "Leader_ID" in out.columns and "scenario_id" in out.columns:
        out["leader_type"] = out.apply(
            lambda r: lookup.get((str(r["scenario_id"]), int(r["Leader_ID"])), np.nan),
            axis=1,
        )
    return out


def _prepare_params_for_summary(all_params: pd.DataFrame) -> pd.DataFrame:
    """Ensure follower_type, leader_type, fl_combo columns exist."""
    df = _enrich_params_types_from_trajectories(all_params.copy())
    if "fl_combo" not in df.columns:
        if "follower_type" in df.columns and "leader_type" in df.columns:
            valid = df["follower_type"].notna() & df["leader_type"].notna()
            df["fl_combo"] = np.nan
            df.loc[valid, "fl_combo"] = (
                df.loc[valid, "follower_type"].astype(str)
                + "-"
                + df.loc[valid, "leader_type"].astype(str)
            )
        elif "follower_type" in df.columns:
            df["fl_combo"] = np.nan
    return df


def _summary_combo_value(subset: pd.DataFrame, param_col: str) -> object:
    if subset.empty or param_col not in subset.columns:
        return ""
    return round(float(subset[param_col].mean()), 2)


def build_idm_summary_table(param_files: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build IDM summary with mean parameters per follower-leader type pair (S-S, S-L, …)
    plus a pooled Vehicle-Vehicle column.
    """
    summary_cols = ["Model", "Parameter", "Range"] + FL_COMBO_ORDER + ["Vehicle-Vehicle"]
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
        for combo in FL_COMBO_ORDER:
            sub = all_params[all_params["fl_combo"] == combo]
            row[combo] = _summary_combo_value(sub, col)
        row["Vehicle-Vehicle"] = round(float(all_params[col].mean()), 2)
        rows.append(row)

    count_row: dict = {
        "Model": "IDM",
        "Parameter": "count",
        "Range": "-",
        "Vehicle-Vehicle": int(len(all_params)),
    }
    for combo in FL_COMBO_ORDER:
        count_row[combo] = int((all_params["fl_combo"] == combo).sum())
    rows.append(count_row)
    return pd.DataFrame(rows, columns=summary_cols)

def cap_events_total(
    groups: Dict[str, List[Tuple[int, str]]],
    max_total: int,
) -> Dict[str, List[Tuple[int, str]]]:
    """Limit total calibration events across S / L / A groups."""
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


# ---------------------------------------------------------------------------
# Extraction diagnostics (no GA)
# ---------------------------------------------------------------------------
def run_extraction_diagnostics(
    datasets: Optional[Dict[str, str]] = None,
    save_dir: str = PLOTS_DIR,
    max_events_per_group: int = 5,
) -> None:
    """Plot observed leader/follower x,y and lane-distance kinematics without running GA."""
    datasets = datasets or DATA_FILES
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
) -> None:
    global sdf, ldf, total_time, time_step, timex
    global leader_position, leader_speed, target_position, target_speed

    datasets = datasets or DATA_FILES
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    groups = generate_waymo_vehicle_groups(datasets)
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

    combined_df, _scenario_src, global_run = build_combined_dataframe(datasets)
    type_lookup = build_vehicle_type_lookup(combined_df)

    for group_name, event_keys in groups.items():
        if not event_keys:
            print(f"[skip] {group_name}: no events")
            continue

        outname = f"IDM_Params_{group_name}"
        output_csv_path = os.path.join(save_dir, f"{outname}.csv")
        if skip_existing and os.path.exists(output_csv_path):
            print(f"[skip] {output_csv_path} already exists")
            continue

        if max_events_per_group is not None:
            event_keys = event_keys[: max_events_per_group]

        params_list: List[list] = []
        all_simulations_list: List[pd.DataFrame] = []
        best_metrics = None

        for follower_id, scenario_id in event_keys:
            sdf, ldf, duration, start_time, lane_geom = extract_subject_and_leader_data(
                combined_df, follower_id, scenario_id
            )
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

            best_params, best_error, best_metrics = genetic_algorithm()
            if best_params is None or best_metrics is None:
                print("  -> GA failed; skip")
                continue

            follower_type = type_lookup.get((scenario_id, follower_id), "S")
            leader_type = type_lookup.get((scenario_id, leader_id), "S")
            fl_combo = f"{follower_type}-{leader_type}"

            params_list.append(
                [
                    follower_id,
                    leader_id,
                    run_index,
                    scenario_id,
                    duration,
                    follower_type,
                    leader_type,
                    fl_combo,
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
                    "fl_combo",
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
            params_df.to_csv(output_csv_path, index=False)
            print(f"Saved parameters: {output_csv_path}")

        if all_simulations_list:
            sim_path = os.path.join(save_dir, f"IDM_Simulated_{group_name}.csv")
            pd.concat(all_simulations_list, ignore_index=True).to_csv(sim_path, index=False)
            print(f"Saved simulated trajectories: {sim_path}")

    write_idm_summary_table(save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Waymo IDM car-following calibration")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap events per group (Waymo_S, Waymo_L, Waymo_A each)",
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
        write_idm_summary_table()
    elif args.plot_only:
        run_extraction_diagnostics(max_events_per_group=args.max_events or 5)
    else:
        run_calibration(
            max_events_per_group=args.max_events,
            max_events_total=args.max_total,
            skip_existing=not args.overwrite,
            save_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
