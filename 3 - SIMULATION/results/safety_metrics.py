"""Trajectory-based surrogate safety metrics for CS3 (and any CSV with the same schema).

Schema expected (per row): time, id, type, x, y, v, theta, length[, lane, lane_pos]

Methods
-------
V2V   : Split by location (SUMO ``lane`` id):
        * **Link / approach (not intersection):** 1-D same-lane TTC — excludes
          internal junction lanes (ids starting with ``:``).
        * **Intersection (internal lanes):** 2-D predictive motor–motor TTC
          when at least one vehicle is on an internal ``:`` lane at that time.
        Per (follower_id, leader_id) -> min TTC over time in each regime.

V2P,
V2B   : Predictive 2-D TTC (constant-velocity, directional).
        Solve the smallest positive ``t`` of
              |Δp + Δv t|² = r_sum²,
        with Δp = p_vru - p_veh, Δv = v_vru - v_veh (both 2-D vectors,
        so Δv is directional — a bike travelling alongside a car at the
        same velocity yields Δv ≈ 0 and hence no TTC).  Agents that
        merely pass close to each other (future paths do not come within
        r_sum of each other) correctly return no TTC, avoiding the
        over-triggering that a purely radial "d / closing_speed"
        formulation produces at lateral near-misses.

        Aggregation (per-VRU critical exposure):
            1. Each timestep, for every VRU, compute TTC to every vehicle
               and keep the single vehicle that yields the minimum TTC.
            2. Across timesteps, keep one row per vru_id: the globally
               smallest TTC and the vehicle id responsible for it.
        (SSAM / Hayward-style predictive 2-D TTC.)

PET   : Post-Encroachment Time (SSAM-style grid).
        Discretize the map into cells, record each (cell, id, type, time).
        For any cell visited by two different agents at different times,
        PET = t_second_enter - t_first_leave (bounded below by 0).
        Per (id_a, id_b) -> min PET over all shared cells.

Type codes (single_inter_sim.py):
   -1 : pedestrian,  -2 : bicycle
    0/1/2 : human-driven (SV / AV / HV),  3/4 : connected (CAV / CAHV)

Typical usage
-------------
    python safety_metrics.py --input 3.csv --label CS3_50SV_50CAV

    # or from a notebook / module
    from safety_metrics import run
    run(input_csv="3.csv", label="CS3_50SV_50CAV")
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
HUMAN_TYPES = {"0", "1", "2"}   # SV, AV, HV (PT / DDM in sim)
CAV_TYPES = {"3", "4"}          # CAV, CAHV (C-IDM / C-MOBIL)
PED_TYPE, BIKE_TYPE = -1, -2

# All disks approximate *lateral* extent (half-width), never half-length.
# Using half-length would over-inflate the footprint and would flag
# side-by-side traffic (e.g. a bike in an adjacent lane at 2 m lateral
# clearance) as "already overlapping".
R_PED_DEFAULT = 0.30          # pedestrian shoulder half-width
R_BIKE_DEFAULT = 0.40         # bicycle + rider half-width
R_VEH_HALF_WIDTH = 0.90       # half of a typical 1.8 m passenger-car width
R_VEH_BUFFER = 0.10           # small safety pad around the disk
R_VEH_GAP_FALLBACK = 2.5      # only for the follower/leader gap in 1-D V2V

PREFILTER_RANGE_M = 50.0
TTC_HORIZON_S = 10.0
TTC_CRITICAL_S = 1.5
TTC_MAX_PLOT = 15.0

# Minimum relative speed (m/s) required to accept a V2VRU TTC.  Prevents
# quasi-static drift encounters (SUMO's 0.5 s step can produce sub-cm
# position jitter that would otherwise yield spuriously low TTCs).
TTC_MIN_REL_SPEED_MS = 0.5

PET_GRID_M = 1.0
PET_CRITICAL_S = 2.0

# 508-friendly (Wong)
_C_HUMAN, _C_CAV = "#000000", "#0072B2"
_HATCH_HUMAN, _HATCH_CAV = "///", "\\\\\\"


# ============================================================== IO / cleanup
def load_trajectory_csv(path: str) -> pd.DataFrame:
    """Load trajectory CSV, coerce numeric columns, normalize ``type`` to str int."""
    df = pd.read_csv(path)
    req = {"time", "id", "type", "x", "y", "v", "theta"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    df["type"] = df["type"].astype(str).str.split(".").str[0]
    for col in ("time", "id", "v", "x", "y", "theta"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("lane_pos", "length"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "length" not in df.columns:
        df["length"] = np.nan
    return df.dropna(subset=["time", "id", "v", "x", "y", "theta"]).reset_index(drop=True)


def _type_int(s) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 999


def _is_motor(t) -> bool:
    return _type_int(t) >= 0


def _is_ped(t) -> bool:
    return _type_int(t) == PED_TYPE


def _is_bike(t) -> bool:
    return _type_int(t) == BIKE_TYPE


def _sumo_angle_to_rad(sumo_deg: np.ndarray) -> np.ndarray:
    """Convert SUMO angle (0 deg = north, CW) to standard math radians (CCW from +x)."""
    phi = 90.0 - np.asarray(sumo_deg, dtype=float)
    phi = np.fmod(phi + 360.0, 360.0)
    return np.radians(phi)


def _velocity_xy(v: np.ndarray, theta_sumo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = _sumo_angle_to_rad(theta_sumo)
    return v * np.cos(rad), v * np.sin(rad)


def _radius(row_type: str, length: float) -> float:
    """Effective circular radius for the radial V2VRU TTC.

    Vehicles use half-width (not half-length).  A disk that circumscribes the
    whole vehicle (length/2) grossly inflates the lateral footprint and would
    flag pedestrians on adjacent sidewalks as conflicts; half-width is the
    standard compromise for VRU analyses.
    """
    if _is_ped(row_type):
        return R_PED_DEFAULT
    if _is_bike(row_type):
        return R_BIKE_DEFAULT
    return R_VEH_HALF_WIDTH + R_VEH_BUFFER


def _group(t: str) -> str:
    """Map single_inter_sim agent type to summary group (0=SV … 4=CAHV)."""
    t = str(t).split(".")[0]
    if t in {"0", "2"}:
        return "Human"
    if t == "1":
        return "AV"
    if t in {"3", "4"}:
        return "CAV"
    return "Other"


# ================================================================= V2V
_V2V_COLS = ["follower_id", "leader_id", "ttc", "follower_type"]


def _lane_is_internal(lane) -> bool:
    """True on SUMO junction / intersection internal lanes (lane id starts with ':')."""
    return str(lane).startswith(":")


def _empty_v2v() -> pd.DataFrame:
    return pd.DataFrame(columns=_V2V_COLS)


# ------------------------------------------------------------------ V2V 1-D (link)
def compute_ttc_v2v_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Same-lane 1-D TTC on **non-intersection** lanes; min per (follower, leader)."""
    need = {"lane", "lane_pos", "length"}
    if not need.issubset(df.columns):
        return _empty_v2v()

    d = df[df["type"].apply(_is_motor)].copy()
    d = d.dropna(subset=["lane", "lane_pos"])
    if d.empty:
        return _empty_v2v()

    d = d[~d["lane"].map(_lane_is_internal)].copy()
    d = d.sort_values(["time", "lane", "lane_pos"]).reset_index(drop=True)
    g = d.groupby(["time", "lane"], sort=False)

    d["leader_id"] = g["id"].shift(-1)
    d["lead_pos"] = g["lane_pos"].shift(-1)
    d["lead_v"] = g["v"].shift(-1)
    d["lead_len"] = g["length"].shift(-1).fillna(R_VEH_GAP_FALLBACK * 2)

    d = d.dropna(subset=["leader_id", "lead_pos"]).copy()
    if d.empty:
        return _empty_v2v()
    d["gap"] = d["lead_pos"] - d["lane_pos"] - d["lead_len"]
    d["dv"] = d["v"] - d["lead_v"]
    d = d[(d["gap"] > 0) & (d["dv"] > 0)].copy()
    d["ttc"] = d["gap"] / d["dv"]

    d = d.rename(columns={"id": "follower_id", "type": "follower_type"})
    return (
        d.groupby(["follower_id", "leader_id"], as_index=False)
        .agg(ttc=("ttc", "min"), follower_type=("follower_type", "first"))
    )


def compute_ttc_v2v(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for :func:`compute_ttc_v2v_1d` (link / same-lane, off-intersection)."""
    return compute_ttc_v2v_1d(df)


def diagnose_ttc_v2v_1d(df: pd.DataFrame) -> Dict[str, int]:
    """Counts explaining why same-lane 1-D V2V on links may be empty."""
    out = {
        "motor_rows": 0,
        "after_internal_lane_drop": 0,
        "timesteps_2plus_per_lane": 0,
        "adjacent_lane_pairs": 0,
        "closing_pairs": 0,
    }
    need = {"lane", "lane_pos", "length"}
    if not need.issubset(df.columns):
        return out
    d = df[df["type"].apply(_is_motor)].copy()
    out["motor_rows"] = len(d)
    d = d[~d["lane"].map(_lane_is_internal)]
    d = d.dropna(subset=["lane", "lane_pos"])
    out["after_internal_lane_drop"] = len(d)
    if d.empty:
        return out
    d = d.sort_values(["time", "lane", "lane_pos"]).reset_index(drop=True)
    g = d.groupby(["time", "lane"], sort=False)
    sizes = g.size()
    out["timesteps_2plus_per_lane"] = int((sizes >= 2).sum())
    d["leader_id"] = g["id"].shift(-1)
    d["lead_pos"] = g["lane_pos"].shift(-1)
    d["lead_v"] = g["v"].shift(-1)
    d["lead_len"] = g["length"].shift(-1).fillna(R_VEH_GAP_FALLBACK * 2)
    d = d.dropna(subset=["leader_id", "lead_pos"])
    out["adjacent_lane_pairs"] = len(d)
    d["gap"] = d["lead_pos"] - d["lane_pos"] - d["lead_len"]
    d["dv"] = d["v"] - d["lead_v"]
    out["closing_pairs"] = int(((d["gap"] > 0) & (d["dv"] > 0)).sum())
    return out


def compute_ttc_v2v_2d(
    df: pd.DataFrame,
    *,
    horizon: float = TTC_HORIZON_S,
    range_m: float = PREFILTER_RANGE_M,
    intersection_only: bool = False,
) -> pd.DataFrame:
    """Predictive 2-D motor–motor TTC; min over time per (follower_id, leader_id).

    If ``intersection_only`` is True, only timesteps where **at least one** vehicle
    is on a SUMO internal junction lane (``lane`` id starts with ``:``) are used.
    """
    if df.empty or not {"x", "y", "theta"}.issubset(df.columns):
        return _empty_v2v()

    w = df[df["type"].apply(_is_motor)].dropna(subset=["x", "y", "theta", "time", "id"]).copy()
    if w.empty:
        return _empty_v2v()
    if intersection_only and "lane" not in w.columns:
        return _empty_v2v()

    w["vx"], w["vy"] = _velocity_xy(w["v"].values, w["theta"].values)
    w["radius"] = [
        _radius(t, l) for t, l in zip(w["type"].values, w["length"].values)
    ]

    rows: list[tuple] = []
    r2 = float(range_m) ** 2
    for _t, g in w.groupby("time", sort=False):
        V = g.reset_index(drop=True)
        n = len(V)
        if n < 2:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ai, aj = V.iloc[i], V.iloc[j]
                if intersection_only:
                    if not (
                        _lane_is_internal(ai.get("lane", ""))
                        or _lane_is_internal(aj.get("lane", ""))
                    ):
                        continue
                dx = float(aj["x"] - ai["x"])
                dy = float(aj["y"] - ai["y"])
                if dx * dx + dy * dy > r2:
                    continue
                dvx = float(aj["vx"] - ai["vx"])
                dvy = float(aj["vy"] - ai["vy"])
                r_sum = float(ai["radius"] + aj["radius"])
                ttc = float(
                    _predictive_ttc_vectorized(
                        np.array([dx]), np.array([dy]),
                        np.array([dvx]), np.array([dvy]),
                        np.array([r_sum]),
                        horizon,
                    )[0]
                )
                if np.isfinite(ttc):
                    rows.append(
                        (int(ai["id"]), int(aj["id"]), ttc, str(ai["type"]))
                    )

    if not rows:
        return _empty_v2v()
    raw = pd.DataFrame(rows, columns=_V2V_COLS)
    return raw.groupby(["follower_id", "leader_id"], as_index=False).agg(
        ttc=("ttc", "min"),
        follower_type=("follower_type", "first"),
    )


def compute_ttc_v2v_2d_intersection(
    df: pd.DataFrame,
    *,
    horizon: float = TTC_HORIZON_S,
    range_m: float = PREFILTER_RANGE_M,
) -> pd.DataFrame:
    """2-D V2V TTC restricted to intersection (internal ``:``) lane timesteps."""
    return compute_ttc_v2v_2d(
        df, horizon=horizon, range_m=range_m, intersection_only=True
    )


def compute_ttc_v2v_split(
    df: pd.DataFrame,
    *,
    horizon: float = TTC_HORIZON_S,
    range_m: float = PREFILTER_RANGE_M,
) -> Dict[str, pd.DataFrame]:
    """Return ``{\"V2V_1D_link\", \"V2V_2D_intersection\"}`` TTC tables."""
    return {
        "V2V_1D_link": compute_ttc_v2v_1d(df),
        "V2V_2D_intersection": compute_ttc_v2v_2d_intersection(
            df, horizon=horizon, range_m=range_m
        ),
    }


# =========================================== V2VRU predictive 2-D TTC
def _predictive_ttc_vectorized(
    dx: np.ndarray,
    dy: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    r_sum: np.ndarray,
    horizon: float,
    min_rel_speed: float = TTC_MIN_REL_SPEED_MS,
) -> np.ndarray:
    """
    Trajectory-based 2-D TTC under constant-velocity assumption.

    Find the smallest positive ``t`` solving
        |Δp + Δv·t|² = r_sum²,
    with Δp = p_vru - p_veh and Δv = v_vru - v_veh (both 2-D vectors;
    Δv is therefore directional — a bike moving alongside a car at the
    same velocity gives Δv ≈ 0 and no TTC).

    Expanding yields  a t² + b t + c = 0  with
        a = |Δv|²,                (relative speed squared)
        b = 2 Δp · Δv,
        c = |Δp|² - r_sum²        (positive when currently separated).

    A valid TTC exists iff:
        * c > 0            (not already overlapping; spawn/sidewalk artefact
                            samples are dropped rather than reported as 0),
        * b < 0            (currently approaching — equivalent to "closing
                            speed along line-of-sight is positive"),
        * a > min_rel_speed² (non-trivial relative motion; filters quasi-
                            static encounters arising from sub-cm jitter),
        * b² - 4ac ≥ 0     (the two future paths actually bring the disks
                            into contact; agents that merely pass near each
                            other miss this test and correctly return NaN),
        * the smallest positive root lies inside (0, horizon].
    """
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    c = dx * dx + dy * dy - r_sum * r_sum

    min_a = float(min_rel_speed) ** 2
    valid = (c > 0.0) & (b < 0.0) & (a > min_a)
    disc = b * b - 4.0 * a * c
    valid &= (disc >= 0.0)

    ttc = np.full_like(a, np.nan, dtype=float)
    if not valid.any():
        return ttc

    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(np.where(valid, disc, 0.0))
        t1 = np.where(valid, (-b - sq) / (2.0 * a), np.nan)

    ok = (t1 > 0.0) & (t1 <= horizon)
    return np.where(ok, t1, np.nan)


def compute_ttc_v2vru(
    df: pd.DataFrame, horizon: float = TTC_HORIZON_S
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predictive 2-D TTC (constant-velocity, directional Δv).

    Per-VRU worst-case exposure:
        * each timestep, for every VRU, compute TTC to every vehicle and
          keep the *single* vehicle that gives the minimum TTC at that step;
        * across all timesteps, keep *one* row per VRU with the overall
          smallest TTC and the vehicle id responsible for it.

    Returns (V2P, V2B) dataframes with columns
        vru_id, veh_id, ttc, time, vru_class, veh_type.
    """
    cols = ["vru_id", "veh_id", "ttc", "time", "vru_class", "veh_type"]
    if df.empty:
        empty = pd.DataFrame(columns=cols)
        return empty, empty.copy()

    w = df.copy()
    w["vx"], w["vy"] = _velocity_xy(w["v"].values, w["theta"].values)
    w["radius"] = [
        _radius(t, l) for t, l in zip(w["type"].values, w["length"].values)
    ]

    per_step: list[tuple] = []
    for t_val, g in w.groupby("time", sort=False):
        V = g[g["type"].apply(_is_motor)]
        if V.empty:
            continue
        P = g[g["type"].apply(_is_ped)]
        B = g[g["type"].apply(_is_bike)]
        if P.empty and B.empty:
            continue

        vx = V["x"].values[:, None]
        vy = V["y"].values[:, None]
        vvx = V["vx"].values[:, None]
        vvy = V["vy"].values[:, None]
        vr = V["radius"].values[:, None]
        v_ids = V["id"].values.astype(int)
        v_types = V["type"].values

        for S, vclass in ((P, "ped"), (B, "bike")):
            if S.empty:
                continue
            dx = S["x"].values[None, :] - vx
            dy = S["y"].values[None, :] - vy
            near = (dx * dx + dy * dy) <= PREFILTER_RANGE_M * PREFILTER_RANGE_M
            if not near.any():
                continue

            dvx = S["vx"].values[None, :] - vvx
            dvy = S["vy"].values[None, :] - vvy
            r_sum = vr + S["radius"].values[None, :]

            ttc = _predictive_ttc_vectorized(dx, dy, dvx, dvy, r_sum, horizon)
            ttc = np.where(near, ttc, np.nan)

            # Per VRU column: which vehicle gives the smallest TTC?
            mask_finite = np.isfinite(ttc)
            col_has_any = mask_finite.any(axis=0)
            if not col_has_any.any():
                continue

            filled = np.where(mask_finite, ttc, np.inf)
            idx_veh = np.argmin(filled, axis=0)
            min_ttc = filled[idx_veh, np.arange(filled.shape[1])]

            keep = np.isfinite(min_ttc) & col_has_any
            if not keep.any():
                continue

            vru_ids = S["id"].values[keep].astype(int)
            best_veh = v_ids[idx_veh[keep]]
            best_types = v_types[idx_veh[keep]]
            best_ttc = min_ttc[keep].astype(float)

            per_step.extend(
                (int(vi), int(ci), float(tt), float(t_val), vclass, str(vt))
                for vi, ci, tt, vt in zip(vru_ids, best_veh, best_ttc, best_types)
            )

    if not per_step:
        empty = pd.DataFrame(columns=cols)
        return empty, empty.copy()

    raw = pd.DataFrame(per_step, columns=cols)
    # Global min per VRU: keep the single timestep that defines the worst case.
    raw_sorted = raw.sort_values("ttc", kind="mergesort")
    final = raw_sorted.drop_duplicates(subset=["vru_class", "vru_id"], keep="first")
    final = final.reset_index(drop=True)

    ped = final[final["vru_class"] == "ped"].reset_index(drop=True)
    bike = final[final["vru_class"] == "bike"].reset_index(drop=True)
    return ped, bike


# =========================================================== PET (grid-based)
def _agent_class(t: str) -> str:
    if _is_ped(t):
        return "ped"
    if _is_bike(t):
        return "bike"
    if _is_motor(t):
        return "veh"
    return "other"


def compute_pet(
    df: pd.DataFrame,
    grid: float = PET_GRID_M,
) -> Dict[str, pd.DataFrame]:
    """
    SSAM-style PET.  For each grid cell, keep per-agent (enter, leave) windows
    based on time samples in that cell.  PET between agents a, b in one cell
    is max(0, t_b_enter - t_a_leave) when a leaves before b enters; take min
    across shared cells per pair.

    Returns dict with keys 'V2V', 'V2P', 'V2B'.  Columns:
        id_a, type_a, id_b, type_b, pet
    """
    if df.empty:
        empty = pd.DataFrame(columns=["id_a", "type_a", "id_b", "type_b", "pet"])
        return {"V2V": empty.copy(), "V2P": empty.copy(), "V2B": empty.copy()}

    w = df.copy()
    w["cx"] = np.floor(w["x"].values / grid).astype(np.int64)
    w["cy"] = np.floor(w["y"].values / grid).astype(np.int64)
    w["class"] = w["type"].apply(_agent_class)
    w = w[w["class"].isin({"veh", "ped", "bike"})]
    if w.empty:
        empty = pd.DataFrame(columns=["id_a", "type_a", "id_b", "type_b", "pet"])
        return {"V2V": empty.copy(), "V2P": empty.copy(), "V2B": empty.copy()}

    visits = (
        w.groupby(["cx", "cy", "id"], as_index=False)
        .agg(
            enter=("time", "min"),
            leave=("time", "max"),
            type=("type", "first"),
            cls=("class", "first"),
        )
    )

    pet_rows: list[tuple] = []
    for _cell, g in visits.groupby(["cx", "cy"], sort=False):
        if len(g) < 2:
            continue
        rec = g.sort_values("enter").to_records(index=False)
        n = len(rec)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = rec[i], rec[j]
                if a["id"] == b["id"] or a["cls"] == b["cls"] == "ped":
                    pass
                if a["id"] == b["id"]:
                    continue
                pet = max(0.0, float(b["enter"]) - float(a["leave"]))
                pet_rows.append(
                    (int(a["id"]), str(a["type"]), a["cls"],
                     int(b["id"]), str(b["type"]), b["cls"], pet)
                )

    if not pet_rows:
        empty = pd.DataFrame(columns=["id_a", "type_a", "id_b", "type_b", "pet"])
        return {"V2V": empty.copy(), "V2P": empty.copy(), "V2B": empty.copy()}

    raw = pd.DataFrame(
        pet_rows,
        columns=["id_a", "type_a", "cls_a", "id_b", "type_b", "cls_b", "pet"],
    )
    pair_min = raw.groupby(
        ["id_a", "id_b"], as_index=False
    ).agg(
        pet=("pet", "min"),
        type_a=("type_a", "first"),
        type_b=("type_b", "first"),
        cls_a=("cls_a", "first"),
        cls_b=("cls_b", "first"),
    )

    def _slice(kind: str) -> pd.DataFrame:
        if kind == "V2V":
            sel = (pair_min["cls_a"] == "veh") & (pair_min["cls_b"] == "veh")
        elif kind == "V2P":
            sel = (
                ((pair_min["cls_a"] == "veh") & (pair_min["cls_b"] == "ped"))
                | ((pair_min["cls_a"] == "ped") & (pair_min["cls_b"] == "veh"))
            )
        else:
            sel = (
                ((pair_min["cls_a"] == "veh") & (pair_min["cls_b"] == "bike"))
                | ((pair_min["cls_a"] == "bike") & (pair_min["cls_b"] == "veh"))
            )
        return pair_min.loc[sel, ["id_a", "type_a", "id_b", "type_b", "pet"]].copy()

    return {"V2V": _slice("V2V"), "V2P": _slice("V2P"), "V2B": _slice("V2B")}


# =================================================================== summary
@dataclass
class InteractionSlice:
    name: str
    kind: str
    human: np.ndarray
    cav: np.ndarray


def _summarize(slices: Iterable[InteractionSlice], critical: float) -> pd.DataFrame:
    rows = []
    for s in slices:
        for grp, series in (("Human", s.human), ("CAV", s.cav)):
            series = np.asarray(series, dtype=float)
            series = series[np.isfinite(series)]
            n = int(series.size)
            if n == 0:
                rows.append({
                    "Interaction": s.name, "Metric": s.kind, "Group": grp,
                    "Pairs": 0, "Median": np.nan, "P05": np.nan,
                    "Threshold_s": critical, "Critical_count": 0,
                    "Critical_rate_%": np.nan,
                })
                continue
            crit = int((series < critical).sum())
            rows.append({
                "Interaction": s.name,
                "Metric": s.kind,
                "Group": grp,
                "Pairs": n,
                "Median": round(float(np.median(series)), 3),
                "P05": round(float(np.quantile(series, 0.05)), 3),
                "Threshold_s": critical,
                "Critical_count": crit,
                "Critical_rate_%": round(100.0 * crit / n, 3),
            })
    return pd.DataFrame(rows)


# =================================================================== plotting
def _split_by_group(df: pd.DataFrame, type_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Default two-way split: human-driven (0,2) vs connected (3,4). AV (1) is excluded."""
    if df is None or df.empty:
        return np.array([]), np.array([])
    g = df[type_col].astype(str).apply(_group)
    h = df.loc[g == "Human", value_col].values.astype(float)
    c = df.loc[g == "CAV", value_col].values.astype(float)
    return h[np.isfinite(h)], c[np.isfinite(c)]


def _plot_distribution(
    human: np.ndarray,
    cav: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    critical: float,
    max_plot: float,
    out_path: str,
) -> None:
    if human.size == 0 and cav.size == 0:
        return
    bins = np.linspace(0, max_plot, 31)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if human.size:
        ax.hist(np.clip(human, 0, max_plot), bins=bins, alpha=0.55, color=_C_HUMAN,
                edgecolor="black", hatch=_HATCH_HUMAN, label=f"Human (n={human.size})")
    if cav.size:
        ax.hist(np.clip(cav, 0, max_plot), bins=bins, alpha=0.55, color=_C_CAV,
                edgecolor="black", hatch=_HATCH_CAV, label=f"CAV (n={cav.size})")
    ax.axvline(critical, color="#D55E00", linestyle="--", linewidth=1.7,
               label=f"Critical = {critical} s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_plot)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, edgecolor="0.35")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_cdf(
    human: np.ndarray,
    cav: np.ndarray,
    title: str,
    xlabel: str,
    critical: float,
    max_plot: float,
    out_path: str,
) -> None:
    if human.size == 0 and cav.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lbl, color, ls, series in [
        ("Human", _C_HUMAN, "-", human),
        ("CAV", _C_CAV, "--", cav),
    ]:
        if series.size == 0:
            continue
        srt = np.sort(series)
        cdf = np.arange(1, srt.size + 1) / srt.size
        ax.plot(srt, cdf, color=color, linestyle=ls, linewidth=2.2, label=lbl)
    ax.axvline(critical, color="#D55E00", linestyle="--", linewidth=1.7,
               label=f"Critical = {critical} s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{title} — CDF", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_plot)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, edgecolor="0.35")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================= run
def run(
    input_csv: str,
    label: str = "scenario",
    out_root: str = "processed_outputs_safety",
    horizon: float = TTC_HORIZON_S,
    grid: float = PET_GRID_M,
) -> Dict[str, pd.DataFrame]:
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)

    df = load_trajectory_csv(input_csv)
    print(f"[{label}] rows={len(df)}, unique ids={df['id'].nunique()}, "
          f"types={sorted(df['type'].unique())}")

    v2v_split = compute_ttc_v2v_split(df, horizon=horizon)
    v2v_1d = v2v_split["V2V_1D_link"]
    v2v_2d = v2v_split["V2V_2D_intersection"]
    v2p, v2b = compute_ttc_v2vru(df, horizon=horizon)
    pet = compute_pet(df, grid=grid)

    v2v_1d.to_csv(os.path.join(out_dir, "V2V_1D_link_TTC.csv"), index=False)
    v2v_2d.to_csv(os.path.join(out_dir, "V2V_2D_intersection_TTC.csv"), index=False)
    v2p.to_csv(os.path.join(out_dir, "V2P_TTC.csv"), index=False)
    v2b.to_csv(os.path.join(out_dir, "V2B_TTC.csv"), index=False)
    for k, df_pet in pet.items():
        df_pet.to_csv(os.path.join(out_dir, f"{k}_PET.csv"), index=False)

    def _plot_pair(df_in, type_col, name, kind, ylabel, critical, max_plot):
        h, c = _split_by_group(df_in, type_col, value_col="ttc" if kind == "TTC" else "pet")
        title = f"{name} {kind} — {label}"
        xlabel = "Time-to-Collision (s)" if kind == "TTC" else "Post-Encroachment Time (s)"
        _plot_distribution(h, c, title, xlabel, ylabel, critical, max_plot,
                           os.path.join(out_dir, f"{name}_{kind}_distribution.png"))
        _plot_cdf(h, c, title, xlabel, critical, max_plot,
                  os.path.join(out_dir, f"{name}_{kind}_CDF.png"))
        return InteractionSlice(name=name, kind=kind, human=h, cav=c)

    yl_pair = "Count of follower–leader pairs (min TTC per pair)"
    yl_vru_p = "Count of pedestrians (global-min TTC per pedestrian)"
    yl_vru_b = "Count of bicycles (global-min TTC per bicycle)"
    yl_pet = "Count of id–id pairs (min PET per pair)"
    slices = [
        _plot_pair(v2v_1d, "follower_type", "V2V-1D (link)", "TTC", yl_pair, TTC_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(v2v_2d, "follower_type", "V2V-2D (intersection)", "TTC", yl_pair, TTC_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(v2p, "veh_type",      "V2P", "TTC", yl_vru_p, TTC_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(v2b, "veh_type",      "V2B", "TTC", yl_vru_b, TTC_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(pet["V2V"], "type_a", "V2V", "PET", yl_pet,   PET_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(pet["V2P"], "type_a", "V2P", "PET", yl_pet,   PET_CRITICAL_S, TTC_MAX_PLOT),
        _plot_pair(pet["V2B"], "type_a", "V2B", "PET", yl_pet,   PET_CRITICAL_S, TTC_MAX_PLOT),
    ]

    summary = pd.DataFrame()
    if slices:
        summary = pd.concat([
            _summarize([s], critical=TTC_CRITICAL_S if s.kind == "TTC" else PET_CRITICAL_S)
            for s in slices
        ], ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"\n[{label}] Summary:")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"\nArtifacts written to: {out_dir}")

    return {
        "V2V_1D_link_TTC": v2v_1d,
        "V2V_2D_intersection_TTC": v2v_2d,
        "V2P_TTC": v2p,
        "V2B_TTC": v2b,
        "V2V_PET": pet["V2V"],
        "V2P_PET": pet["V2P"],
        "V2B_PET": pet["V2B"],
        "summary": summary,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Trajectory CSV")
    p.add_argument("--label", default="scenario", help="Run label used for output folder & titles")
    p.add_argument("--out-dir", default="processed_outputs_safety", help="Root output directory")
    p.add_argument("--horizon", type=float, default=TTC_HORIZON_S, help="Max TTC kept (s)")
    p.add_argument("--grid", type=float, default=PET_GRID_M, help="PET grid cell size (m)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        input_csv=args.input,
        label=args.label,
        out_root=args.out_dir,
        horizon=args.horizon,
        grid=args.grid,
    )
