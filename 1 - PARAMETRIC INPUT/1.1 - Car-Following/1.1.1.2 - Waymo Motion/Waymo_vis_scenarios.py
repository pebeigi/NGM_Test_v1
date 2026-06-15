"""
Waymo Open Motion — scenario visualization from pre-processed CSVs.

Replaces the TFRecord-based notebooks. Plots agent trajectories with or without
map geometry. No TensorFlow or waymo-open-dataset required.

Inputs (under ``0 - Datasets/`` at repo root):
  March2023waymo_scenario_lane_leader_follower_assigned_{196–199}_data.csv
  March2023waymo_map_features_{196–199}_data.csv

Outputs (default ``Results/plots_scenarios/``):
  {scenario_id}_map.png      — trajectories over lane/road map
  {scenario_id}_nomap.png    — trajectories only (color per agent)
"""

from __future__ import annotations

import argparse
import ast
import gc
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

_NGM_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import DATASETS_DIR, REPO_ROOT

DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "Results", "plots_scenarios")

CF_GLOB = "March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv"
MAP_GLOB = "March2023waymo_map_features_*_data.csv"

TYPE_VEHICLE = 1
AGENT_LINE_STYLES = (
    "-", "--", "-.", ":",
    (0, (6, 4)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2, 1, 2)),
)


def parse_coord_list(val) -> List[float]:
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


def discover_csv_pairs(datasets_dir: str) -> List[Tuple[str, str, str]]:
    """Return (tag, cf_path, map_path) for each numbered Waymo CSV pair."""
    pairs: List[Tuple[str, str, str]] = []
    for cf_path in sorted(glob_paths(datasets_dir, CF_GLOB)):
        suffix = cf_path.split("_assigned_")[-1].replace("_data.csv", "")
        map_name = f"March2023waymo_map_features_{suffix}_data.csv"
        map_path = os.path.join(datasets_dir, map_name)
        if os.path.isfile(map_path):
            pairs.append((f"Waymo_{suffix}", cf_path, map_path))
    return pairs


def glob_paths(directory: str, pattern: str) -> List[str]:
    import glob

    return sorted(glob.glob(os.path.join(directory, pattern)))


def plot_map_on_ax(ax, map_df: pd.DataFrame, label_lanes: bool = False) -> None:
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

        if label_lanes:
            xs = parse_coord_list(row.get("lane_polyline_x"))
            ys = parse_coord_list(row.get("lane_polyline_y"))
            if len(xs) >= 2:
                mid = len(xs) // 2
                feature_id = row.get("feature_id", "")
                ax.text(
                    xs[mid] + 0.5, ys[mid] + 0.5, f"L:{feature_id}",
                    color="black", fontsize=6, ha="center", va="bottom", zorder=3, alpha=0.85,
                )


def extract_tracks(scenario_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Build (states, masks, object_types, vehicle_ids) arrays for one scenario."""
    tracks = []
    masks = []
    types = []
    vehicle_ids: List[int] = []

    for vehicle_id, g in scenario_df.groupby("vehicle_id", sort=False):
        g = g.sort_values("time")
        states = np.column_stack([g["center_x"].to_numpy(), g["center_y"].to_numpy()])
        valid = g["valid"].astype(bool).to_numpy() if "valid" in g.columns else np.ones(len(g), dtype=bool)
        if valid.sum() == 0:
            continue
        tracks.append(states)
        masks.append(valid)
        types.append(int(g["object_type"].iloc[0]))
        vehicle_ids.append(int(vehicle_id))

    if not tracks:
        return (
            np.zeros((0, 0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
            np.zeros(0, dtype=int),
            [],
        )

    max_steps = max(t.shape[0] for t in tracks)
    n = len(tracks)
    all_states = np.zeros((n, max_steps, 2), dtype=np.float32)
    all_masks = np.zeros((n, max_steps), dtype=bool)
    for i, (states, valid) in enumerate(zip(tracks, masks)):
        all_states[i, : states.shape[0]] = states
        all_masks[i, : valid.shape[0]] = valid

    return all_states, all_masks, np.array(types, dtype=int), vehicle_ids


def get_viewport(states: np.ndarray, masks: np.ndarray) -> Tuple[float, float, float]:
    valid = states[masks]
    if valid.size == 0:
        return 0.0, 0.0, 10.0
    xs = valid[:, 0]
    ys = valid[:, 1]
    center_x = (xs.max() + xs.min()) / 2
    center_y = (ys.max() + ys.min()) / 2
    width = max(float(np.ptp(xs)), float(np.ptp(ys)), 10.0)
    return center_y, center_x, width


def agent_linestyle(agent_idx: int):
    return AGENT_LINE_STYLES[agent_idx % len(AGENT_LINE_STYLES)]


def agent_colors(num_agents: int) -> np.ndarray:
    cmap = plt.get_cmap("jet", max(num_agents, 1))
    colors = cmap(np.arange(num_agents))
    np.random.default_rng(0).shuffle(colors)
    return colors


def plot_scenario(
    scenario_id: str,
    scenario_df: pd.DataFrame,
    map_df: Optional[pd.DataFrame],
    *,
    with_map: bool,
    time_index: int,
    size_pixels: int = 1000,
    label_lanes: bool = False,
) -> Optional[plt.Figure]:
    states, masks, object_types, _ = extract_tracks(scenario_df)
    if states.shape[0] == 0:
        return None

    total_steps = states.shape[1]
    t_idx = min(time_index, total_steps - 1)

    past_states = states[:, :t_idx, :]
    past_masks = masks[:, :t_idx]
    current_states = states[:, t_idx : t_idx + 1, :]
    current_masks = masks[:, t_idx : t_idx + 1]
    future_states = states[:, t_idx + 1 :, :]
    future_masks = masks[:, t_idx + 1 :]

    center_y, center_x, width = get_viewport(states, masks)

    dpi = 100
    fig, ax = plt.subplots(figsize=(size_pixels / dpi, size_pixels / dpi), dpi=dpi)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)

    if with_map and map_df is not None and not map_df.empty:
        plot_map_on_ax(ax, map_df, label_lanes=label_lanes)

    num_agents = states.shape[0]
    colors = agent_colors(num_agents)

    for agent_idx in range(num_agents):
        is_vehicle = object_types[agent_idx] == TYPE_VEHICLE
        marker = "o" if is_vehicle else "*"
        marker_size = 20 if is_vehicle else 40

        past_x = past_states[agent_idx, :, 0][past_masks[agent_idx, :]]
        past_y = past_states[agent_idx, :, 1][past_masks[agent_idx, :]]
        future_x = future_states[agent_idx, :, 0][future_masks[agent_idx, :]]
        future_y = future_states[agent_idx, :, 1][future_masks[agent_idx, :]]

        if with_map:
            ls = agent_linestyle(agent_idx)
            if past_x.size >= 2:
                ax.plot(past_x, past_y, color="black", linestyle=ls, linewidth=0.9, zorder=4)
            if future_x.size >= 2:
                ax.plot(future_x, future_y, color="black", linestyle=ls, linewidth=0.9, zorder=4)
            if current_masks[agent_idx, 0]:
                ax.scatter(
                    current_states[agent_idx, 0, 0], current_states[agent_idx, 0, 1],
                    marker=marker, s=marker_size * 1.5, zorder=5,
                    facecolors="white", edgecolors="black", linewidths=1.0,
                )
        else:
            color = colors[agent_idx]
            if past_x.size:
                ax.scatter(past_x, past_y, marker=marker, s=marker_size, color=color, alpha=0.7, zorder=4)
            if future_x.size:
                ax.scatter(future_x, future_y, marker=marker, s=marker_size, color=color, alpha=0.7, zorder=4)
            if current_masks[agent_idx, 0]:
                ax.scatter(
                    current_states[agent_idx, 0, 0], current_states[agent_idx, 0, 1],
                    marker=marker, s=marker_size * 1.5, color=color, alpha=0.9, zorder=5,
                )

    mode_label = "with map" if with_map else "trajectories only"
    ax.set_title(f"Scenario {scenario_id} — {mode_label} (T={t_idx})")
    plot_range = max(10.0, width * 1.05)
    ax.set_xlim(center_x - plot_range / 2, center_x + plot_range / 2)
    ax.set_ylim(center_y - plot_range / 2, center_y + plot_range / 2)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def process_file(
    cf_path: str,
    map_path: str,
    output_dir: str,
    *,
    modes: Sequence[str],
    scenario_ids: Optional[Sequence[str]],
    max_scenarios: Optional[int],
    time_index: int,
    label_lanes: bool,
) -> int:
    print(f"Loading {os.path.basename(cf_path)} ...")
    cf_df = pd.read_csv(cf_path)
    map_df = pd.read_csv(map_path)
    map_by_scenario = {str(k): v for k, v in map_df.groupby("scenario_id", sort=False)}

    scenario_list = sorted(cf_df["scenario_id"].astype(str).unique())
    if scenario_ids:
        wanted = {str(s) for s in scenario_ids}
        scenario_list = [s for s in scenario_list if s in wanted]

    images_saved = 0
    scenarios_plotted = 0
    for i, scenario_id in enumerate(scenario_list):
        if max_scenarios is not None and scenarios_plotted >= max_scenarios:
            break

        scenario_df = cf_df[cf_df["scenario_id"].astype(str) == scenario_id]
        scenario_map = map_by_scenario.get(scenario_id, pd.DataFrame())
        scenario_saved = False

        for mode in modes:
            with_map = mode == "map"
            fig = plot_scenario(
                scenario_id,
                scenario_df,
                scenario_map,
                with_map=with_map,
                time_index=time_index,
                label_lanes=label_lanes and with_map,
            )
            if fig is None:
                print(f"  [{i + 1}] {scenario_id} ({mode}): no valid tracks, skipped")
                continue

            out_path = os.path.join(output_dir, f"{scenario_id}_{mode}.png")
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{i + 1}] saved {out_path}")
            images_saved += 1
            scenario_saved = True

        if scenario_saved:
            scenarios_plotted += 1

        if (i + 1) % 10 == 0:
            gc.collect()

    return images_saved


def parse_modes(mode_arg: str) -> List[str]:
    if mode_arg == "both":
        return ["map", "nomap"]
    if mode_arg in ("map", "nomap"):
        return [mode_arg]
    raise argparse.ArgumentTypeError("mode must be map, nomap, or both")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Waymo scenarios from pre-processed CSVs (no TFRecords)."
    )
    parser.add_argument(
        "--datasets-dir",
        default=DATASETS_DIR,
        help=f"Folder with Waymo CSVs (default: {DATASETS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder for PNGs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mode",
        default="both",
        type=parse_modes,
        help="Plot style: map (with geometry), nomap (trajectories only), or both (default)",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="Plot only this scenario ID (repeatable)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Cap the number of scenarios plotted per CSV file",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=99,
        help="Timestep index used as the current/pivot point (default: 99)",
    )
    parser.add_argument(
        "--label-lanes",
        action="store_true",
        help="Annotate lane feature IDs on map plots",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    pairs = discover_csv_pairs(args.datasets_dir)
    if not pairs:
        raise SystemExit(
            f"No Waymo CSV pairs found in {args.datasets_dir}. "
            "Expected March2023waymo_scenario_lane_leader_follower_assigned_*_data.csv "
            "and matching map_features files."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0
    for _tag, cf_path, map_path in pairs:
        total += process_file(
            cf_path,
            map_path,
            args.output_dir,
            modes=args.mode,
            scenario_ids=args.scenario_ids,
            max_scenarios=args.max_scenarios,
            time_index=args.time_index,
            label_lanes=args.label_lanes,
        )

    print(f"\nDone. Saved {total} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
