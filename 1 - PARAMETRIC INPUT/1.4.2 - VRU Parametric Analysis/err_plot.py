# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:43:32 2025

@author: TBP
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Switch: True = separate peds & bicycles, False = combine them in one plot
split_by_type = True

# 1) Load the data
df = pd.read_csv(os.path.join(_THIS_DIR, "SF_calibrated_params.csv"))

# 2) Define the metrics and colors
metrics = ["RMSE", "MAE", "R-squared"]
colors = ["blue", "green", "red"]  # one color per metric

if split_by_type:
    # =======================================================================
    # CASE 1: SPLIT PEDESTRIANS AND BICYCLES
    # =======================================================================
    # Split into pedestrians (0) and bicycles (1)
    df_peds = df[df["type_most_common"] == 0].copy()
    df_bicy = df[df["type_most_common"] == 1].copy()
    
    # Create a 2 (rows) × 3 (columns) figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    fig.suptitle("Distribution of Error Metrics for Pedestrians and Bicycles", 
                 fontsize=16, y=1.03)
    
    # Plot each metric for pedestrians (top row) and bicycles (bottom row)
    for i, metric in enumerate(metrics):
        # Filter data for pedestrians based on the metric
        if metric == "R-squared":
            data_peds = df_peds[df_peds[metric].between(0.6, 1)]
        elif metric == "MAE":
            data_peds = df_peds[df_peds[metric].between(0, 5)]
        elif metric == "RMSE":
            data_peds = df_peds[df_peds[metric].between(0, 10)]
        else:
            data_peds = df_peds
        
        sns.histplot(
            data=data_peds,
            x=metric,
            ax=axes[0, i],
            kde=True,
            color=colors[i],
            bins=20,
            edgecolor="black",
            alpha=0.7
        )
        axes[0, i].set_title(f"Distribution of {metric} (Peds)", fontsize=12)
        axes[0, i].set_xlabel(metric)
        axes[0, i].set_ylabel("Count")
        
        # Filter data for bicycles based on the metric
        if metric == "R-squared":
            data_bicy = df_bicy[df_bicy[metric].between(0, 1)]
        elif metric == "MAE":
            data_bicy = df_bicy[df_bicy[metric].between(0, 15)]
        elif metric == "RMSE":
            data_bicy = df_bicy[df_bicy[metric].between(0, 20)]
        else:
            data_bicy = df_bicy
        
        sns.histplot(
            data=data_bicy,
            x=metric,
            ax=axes[1, i],
            kde=True,
            color=colors[i],
            bins=20,
            edgecolor="black",
            alpha=0.7
        )
        axes[1, i].set_title(f"Distribution of {metric} (Bicycles)", fontsize=12)
        axes[1, i].set_xlabel(metric)
        axes[1, i].set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()

else:
    # =======================================================================
    # CASE 2: COMBINE ALL DATA (NO SPLIT)
    # =======================================================================
    # Create a single row of 3 subplots for the entire dataset
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    fig.suptitle("Distribution of Error Metrics (All Data Combined)", 
                 fontsize=16, y=1.03)
    
    for i, metric in enumerate(metrics):
        if metric == "R-squared":
            data_all = df[df[metric].between(0, 1)]
        elif metric == "MAE":
            data_all = df[df[metric].between(0, 15)]
        elif metric == "RMSE":
            data_all = df[df[metric].between(0, 20)]
        else:
            data_all = df
        
        sns.histplot(
            data=data_all,
            x=metric,
            ax=axes[i],
            kde=True,
            color=colors[i],
            bins=20,
            edgecolor="black",
            alpha=0.7
        )
        axes[i].set_title(f"Distribution of {metric} (All)", fontsize=12)
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()
