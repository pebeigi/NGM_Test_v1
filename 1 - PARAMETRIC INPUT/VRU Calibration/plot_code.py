# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:00:14 2025

@author: TBP
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Read the CSV files
df_pred = pd.read_csv(os.path.join(_THIS_DIR, "SF_calibration_predictions.csv"))
df_id = pd.read_csv(os.path.join(_THIS_DIR, "SF_calibrated_params.csv"))

# 2) Merge predictions with calibration per ID using the "id" column
df = pd.merge(df_pred, df_id, on="id", how="left")

# Define a dictionary for type names (assuming 0: pedestrian, 1: bicycle)
type_names = {0: "pedestrian", 1: "bicycle"}

# 3) Loop through each type and produce the corresponding plots
for type_val, type_label in type_names.items():
    # Filter the merged data for the current type
    df_type = df[df["type_most_common"] == type_val]
    
    if df_type.empty:
        print(f"No data for type: {type_label}")
        continue

    # 3a) Find the ID with the highest number of rows (presence) within this type
    counts = df_type.groupby("id").size()
    max_id = counts.idxmax()

    # Optionally, pick a random ID (if you want to use it instead of max_id)
    unique_ids = df_type["id"].unique()
    rand_id = random.choice(unique_ids)

    # Uncomment below to prompt the user or select based on some logic.
    # choice = input("Enter 'max' to pick the ID with highest presence or 'random' to pick a random ID: ").strip().lower()
    # selected_id = max_id if choice == 'max' else rand_id

    # selected_id = max_id  # Currently, always selecting the ID with the highest presence
    selected_id = max_id
    print(f"The selected ID for {type_label} is: {selected_id}")

    # 3b) Filter the data for the selected ID and sort by time
    # df_plot = df_type[df_type["id"] == selected_id].copy()
    df_plot = df[df["id"] == selected_id].copy()
    df_plot.sort_values(by="time", inplace=True)
    
    # Smooth predicted y and vy using Savitzky-Golay filter with window_length=11 and polyorder=2
    #x_pred_smooth = savgol_filter(df_plot["x_pred"], window_length=2, polyorder=1)
    #vx_pred_smooth = savgol_filter(df_plot["vx_pred"], window_length=2, polyorder=1)
    #y_pred_smooth = savgol_filter(df_plot["y_pred"], window_length=2, polyorder=1)
    #vy_pred_smooth = savgol_filter(df_plot["vy_pred"], window_length=2, polyorder=1)

    # Define colors for observations and predictions
    color_obs = "#2ca02c"   # green
    color_pred = "#ff7f0e"  # orange

    # 3c) Create subplots for x, y, vx, vy
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Type: {type_label.capitalize()} | ID: {selected_id}", fontsize=16)

    # Top-left: X vs Time
    axes[0, 0].plot(df_plot["time"], df_plot["x_obs"], color=color_obs, linestyle='-', label="x_obs")
    axes[0, 0].plot(df_plot["time"], df_plot["x_pred"], color=color_pred, linestyle='--', label="x_pred")
    axes[0, 0].set_title("X over Time")
    axes[0, 0].set_xlabel("Time (sec)")
    axes[0, 0].set_ylabel("X (m)")
    #axes[0, 0].set_xlim(5580, 5620)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Top-right: Y vs Time
    axes[0, 1].plot(df_plot["time"], df_plot["y_obs"], color=color_obs, linestyle='-', label="y_obs")
    axes[0, 1].plot(df_plot["time"], df_plot["y_pred"], color=color_pred, linestyle='--', label="y_pred")
    axes[0, 1].set_title("Y over Time")
    axes[0, 1].set_xlabel("Time (sec)")
    axes[0, 1].set_ylabel("Y (m)")
    #axes[0, 1].set_xlim(5580, 5620)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Bottom-left: Vx vs Time
    axes[1, 0].plot(df_plot["time"], df_plot["vx_obs"], color=color_obs, linestyle='-', label="vx_obs")
    axes[1, 0].plot(df_plot["time"], df_plot["vx_pred"], color=color_pred, linestyle='--', label="vx_pred")
    axes[1, 0].set_title("$V_x$ over Time")
    axes[1, 0].set_xlabel("Time (sec)")
    axes[1, 0].set_ylabel("$V_x$ (m/s)")
    #axes[1, 0].set_xlim(5580, 5620)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Bottom-right: Vy vs Time
    axes[1, 1].plot(df_plot["time"], df_plot["vy_obs"], color=color_obs, linestyle='-', label="vy_obs")
    axes[1, 1].plot(df_plot["time"], df_plot["vy_pred"], color=color_pred, linestyle='--', label="vy_pred")
    axes[1, 1].set_title("$V_y$ over Time")
    axes[1, 1].set_xlabel("Time (sec)")
    axes[1, 1].set_ylabel("$V_y$ (m/s)")
    #axes[1, 1].set_xlim(5580, 5620)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Set the background (facecolor) of each subplot to white
    #for ax in axes.flatten():
    #    ax.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
