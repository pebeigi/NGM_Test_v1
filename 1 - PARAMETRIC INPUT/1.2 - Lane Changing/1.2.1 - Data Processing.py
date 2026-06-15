import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NGM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import (
    ALL_LANE_CHANGES_CSV,
    dataset_path,
    lane_change_dataset_paths,
    load_tgsim_csv,
)

input_files = lane_change_dataset_paths()

all_lane_changes = []


def get_direction(run_group, dataset_name):
    if dataset_name.startswith("I395"):
        return "yloc-kf", +1
    movement_signs = []
    for _, veh in run_group.groupby("ID"):
        dx = veh["xloc-kf"].diff().dropna()
        if not dx.empty:
            movement_signs.append(dx.mean())
    avg_sign = sum(movement_signs) / len(movement_signs) if movement_signs else 0
    direction = +1 if avg_sign >= 0 else -1
    return "xloc-kf", direction


def get_nearest(cars, pos, axis, direction):
    ahead = cars[cars[axis] * direction > pos * direction].sort_values(axis).head(1)
    behind = cars[cars[axis] * direction < pos * direction].sort_values(axis, ascending=False).head(1)
    front_id = ahead["ID"].values[0] if not ahead.empty else None
    back_id = behind["ID"].values[0] if not behind.empty else None
    return back_id, front_id


for dataset_name, file_path in input_files.items():
    df = load_tgsim_csv(file_path)
    df = df.sort_values(by=["run-index", "ID", "time"]).reset_index(drop=True)

    for run_index, run_group in df.groupby("run-index"):
        axis, direction = get_direction(run_group, dataset_name)

        for veh_id, group in run_group.groupby("ID"):
            group = group.sort_values("time")
            prev_lane = None

            for _, row in group.iterrows():
                current_lane = row["lane-kf"]
                if prev_lane is None:
                    prev_lane = current_lane
                    continue

                if current_lane != prev_lane:
                    change_time = row["time"]
                    pos = row[axis]
                    car_type = row["type-most-common"]

                    frame = run_group[run_group["time"] == change_time]

                    orig_lane_cars = frame[(frame["lane-kf"] == prev_lane) & (frame["ID"] != veh_id)]
                    new_lane_cars = frame[(frame["lane-kf"] == current_lane) & (frame["ID"] != veh_id)]

                    orig_back, orig_front = get_nearest(orig_lane_cars, pos, axis, direction)
                    new_back, new_front = get_nearest(new_lane_cars, pos, axis, direction)

                    all_lane_changes.append(
                        {
                            "dataset": dataset_name,
                            "run-index": run_index,
                            "time": change_time,
                            "vehicle_id": veh_id,
                            "vehicle_type": car_type,
                            "lane_from": prev_lane,
                            "lane_to": current_lane,
                            "orig_lane_behind_id": orig_back,
                            "orig_lane_front_id": orig_front,
                            "new_lane_behind_id": new_back,
                            "new_lane_front_id": new_front,
                        }
                    )

                    prev_lane = current_lane

lane_changes_df = pd.DataFrame(all_lane_changes)
output_path = dataset_path(ALL_LANE_CHANGES_CSV)
lane_changes_df.to_csv(output_path, index=False)

print(f" Lane change analysis complete.\nSaved to:\n{output_path}")

plot_dir = os.path.join(_SCRIPT_DIR, "lane_change_plots")
os.makedirs(plot_dir, exist_ok=True)

all_data = []
for dataset_name, file_path in input_files.items():
    df = load_tgsim_csv(file_path)
    df["dataset"] = dataset_name
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

for dataset_name in input_files:
    dataset_df = combined_df[combined_df["dataset"] == dataset_name]
    if dataset_df.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    for veh_id, group in dataset_df.groupby("ID"):
        ax.scatter(group["xloc-kf"], group["yloc-kf"], s=5, alpha=0.5, label=f"ID {veh_id}")
    ax.set_title(f"Lane-change scatter — {dataset_name}")
    ax.set_xlabel("xloc-kf")
    ax.set_ylabel("yloc-kf")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(os.path.join(plot_dir, f"{dataset_name}_lane_changes.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

print(f"Plots saved to: {plot_dir}")
