"""
Post-simulation plotting from trajectory CSV (same schema as simulation export).

Expected columns (minimum): time, id, x, y, v, lane, lane_pos
Optional: type, road, theta, length

Called after the main results CSV is written (e.g. from GUI.run_simulation).
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sanitize_lane_name(lane: str) -> str:
    s = str(lane).strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:120] if len(s) > 120 else s


def _load_trajectory_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time", "id", "x", "y", "lane", "lane_pos", "v"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df.dropna(subset=["time", "id", "lane"])
    for c in ("time", "x", "y", "v", "lane_pos"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "id"])
    return df


def plot_trajectory_xy(df: pd.DataFrame, out_path: str, title: str | None = None) -> None:
    """
    Single figure: all agents' (x, y) samples in one axes.
    Colors cycle by vehicle id (hashed for stability).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ids = df["id"].unique()
    cmap = plt.get_cmap("tab20")
    for j, vid in enumerate(ids):
        sub = df[df["id"] == vid]
        color = cmap(j % 20)
        ax.scatter(sub["x"], sub["y"], s=2, c=[color], alpha=0.5, linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or "Trajectory plot (all agents)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_space_per_lane(
    df: pd.DataFrame,
    out_dir: str,
    base_name: str = "time_space",
) -> list[str]:
    """
    For each lane, one PNG: time (s) vs lane_pos (m), one scatter series per vehicle id.
    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    lanes = sorted(df["lane"].dropna().unique(), key=lambda x: str(x))
    cmap = plt.get_cmap("tab20")

    for lane in lanes:
        sub = df[df["lane"] == lane].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        lane_ids = sub["id"].unique()
        for j, vid in enumerate(lane_ids):
            g = sub[sub["id"] == vid].sort_values("time")
            color = cmap(j % 20)
            ax.scatter(
                g["time"],
                g["lane_pos"],
                color=color,
                s=8,
                alpha=0.65,
                linewidths=0,
                edgecolors="none",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Lane position (m)")
        safe = _sanitize_lane_name(lane)
        ax.set_title(f"Time vs space — lane {lane}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{base_name}_lane_{safe}.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def plot_flow_density_per_lane(
    df: pd.DataFrame,
    out_dir: str,
    base_name: str = "flow_density",
    time_bin_s: float = 30.0,
) -> list[str]:
    """
    Per lane: estimate density (veh/km) and flow (veh/h) in time bins using
    mean occupancy and mean speed in the bin, then q ≈ ρ × v (space-mean, km/h).

    Lane length (km) is approximated from observed lane_pos span on that lane.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    t_min, t_max = float(df["time"].min()), float(df["time"].max())
    if t_max <= t_min:
        return written

    bins = np.arange(t_min, t_max + time_bin_s, time_bin_s)
    lanes = sorted(df["lane"].dropna().unique(), key=lambda x: str(x))

    for lane in lanes:
        sub = df[df["lane"] == lane].copy()
        if sub.empty:
            continue
        pos_span = float(sub["lane_pos"].max() - sub["lane_pos"].min())
        lane_len_km = max(pos_span / 1000.0, 1e-6)

        rhos: list[float] = []
        qs: list[float] = []

        for i in range(len(bins) - 1):
            t0, t1 = bins[i], bins[i + 1]
            win = sub[(sub["time"] >= t0) & (sub["time"] < t1)]
            if win.empty:
                continue
            # Average number of distinct vehicles present in window / lane length
            n_ids = win["id"].nunique()
            rho = n_ids / lane_len_km
            v_mean = float(win["v"].mean())
            if np.isnan(v_mean):
                continue
            v_kmh = v_mean * 3.6
            q = rho * v_kmh

            rhos.append(rho)
            qs.append(q)

        if len(rhos) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(rhos, qs, c="steelblue", s=20, alpha=0.7)
        ax.set_xlabel("Density ρ (veh/km, approx.)")
        ax.set_ylabel("Flow q (veh/h, approx.)")
        safe = _sanitize_lane_name(lane)
        ax.set_title(f"Flow vs density — lane {lane}\n(aggregation Δt = {time_bin_s:g} s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{base_name}_lane_{safe}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


_VALID_MODES = frozenset({"trajectory_xy", "time_space", "flow_density"})


def run_post_sim_plots(
    csv_path: str,
    modes: str | Iterable[str],
    output_dir: str | None = None,
    flow_density_time_bin_s: float = 30.0,
) -> list[str]:
    """
    Run the selected visualization(s) on the saved trajectory CSV.

    Parameters
    ----------
    csv_path : str
        Path to the main CSV (e.g. test_sim.csv).
    modes : str or iterable of str
        One or more of: 'trajectory_xy', 'time_space', 'flow_density'.
        A single string is treated as one mode.
    output_dir : str, optional
        Directory for figures. Defaults to the CSV directory.
    flow_density_time_bin_s : float, optional
        Time window (seconds) for binning when computing flow/density pairs.
        Ignored unless ``flow_density`` is in ``modes``.

    Returns
    -------
    list of str
        Paths to written PNG files.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    if isinstance(modes, str):
        mode_list = [modes]
    else:
        mode_list = list(modes)

    if not mode_list:
        return []

    for m in mode_list:
        if m not in _VALID_MODES:
            raise ValueError(f"Unknown post-sim plot mode: {m!r}")

    df = _load_trajectory_df(csv_path)
    if df.empty:
        return []

    out_base = output_dir or os.path.dirname(os.path.abspath(csv_path))
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    written: list[str] = []

    for mode in mode_list:
        if mode == "trajectory_xy":
            out_png = os.path.join(out_base, f"{stem}_trajectory_xy.png")
            plot_trajectory_xy(df, out_png)
            written.append(out_png)
        elif mode == "time_space":
            subdir = os.path.join(out_base, f"{stem}_time_space_plots")
            written.extend(plot_time_space_per_lane(df, subdir, base_name=stem))
        elif mode == "flow_density":
            subdir = os.path.join(out_base, f"{stem}_flow_density_plots")
            tb = float(flow_density_time_bin_s)
            if tb <= 0:
                tb = 30.0
            written.extend(
                plot_flow_density_per_lane(
                    df, subdir, base_name=stem, time_bin_s=tb
                )
            )

    return written


__all__: Iterable[str] = (
    "run_post_sim_plots",
    "plot_trajectory_xy",
    "plot_time_space_per_lane",
    "plot_flow_density_per_lane",
)
