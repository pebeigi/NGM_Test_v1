# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:43:32 2025

@author: TBP
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Load data
df = pd.read_csv(os.path.join(_THIS_DIR, "SF_calibrated_params.csv"))

# 2) Separate data by type_most_common
df_ped = df[df["type_most_common"] == 0].copy()  # Pedestrians
df_bic = df[df["type_most_common"] == 1].copy()  # Bicycles

# Define parameter subsets
ped_params = ["A_pp", "B_pp", "A_wall", "B_wall"]
bic_params = ["eps_m", "A_w", "B_w", "A_s", "B_s"]

# Drop rows with missing data in the relevant columns
df_ped = df_ped[ped_params].dropna()
df_bic = df_bic[bic_params].dropna()

# 3) Helper function to plot correlation coefficient + color in upper triangle
def corrfunc(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.annotate(f"{r:.2f}",
                xy=(0.5, 0.5),
                xycoords=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold")
    cmap = plt.cm.get_cmap("coolwarm")
    # Map correlation -1..1 to 0..1 for color
    c = (r + 1) / 2
    ax.set_facecolor(cmap(c))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

###############################################################################
# 4) PLOT 1: PEDESTRIANS
###############################################################################
if not df_ped.empty:
    g_ped = sns.PairGrid(df_ped, vars=ped_params, diag_sharey=False)
    
    # Lower triangle: 2D KDE
    g_ped.map_lower(
        sns.kdeplot,
        cmap="Blues",
        fill=True,
        thresh=0,
        levels=10
    )
    
    # Diagonal: 1D histogram (with KDE)
    g_ped.map_diag(
        sns.histplot,
        kde=True,
        color="skyblue",
        edgecolor="black"
    )
    
    # Upper triangle: correlation coefficient
    g_ped.map_upper(corrfunc)
    
    # Optional: adjust layout
    plt.suptitle("Pedestrian Parameters — Pairwise Relationships", 
                 fontsize=14, y=1.02)
    plt.show()
    
    # Separate correlation heatmap for pedestrians
    corr_matrix_ped = df_ped.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix_ped,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        fmt=".2f"
    )
    plt.title("Correlation Matrix Heatmap (Pedestrians)", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("No pedestrian data found (type_most_common == 0).")

###############################################################################
# 5) PLOT 2: BICYCLES
###############################################################################
if not df_bic.empty:
    g_bic = sns.PairGrid(df_bic, vars=bic_params, diag_sharey=False)
    
    # Lower triangle: 2D KDE
    g_bic.map_lower(
        sns.kdeplot,
        cmap="Blues",
        fill=True,
        thresh=0,
        levels=10
    )
    
    # Diagonal: 1D histogram (with KDE)
    g_bic.map_diag(
        sns.histplot,
        kde=True,
        color="skyblue",
        edgecolor="black"
    )
    
    # Upper triangle: correlation coefficient
    g_bic.map_upper(corrfunc)
    
    # Optional: adjust layout
    plt.suptitle("Bicycle Parameters — Pairwise Relationships", 
                 fontsize=14, y=1.02)
    plt.show()
    
    # Separate correlation heatmap for bicycles
    corr_matrix_bic = df_bic.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix_bic,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        fmt=".2f"
    )
    plt.title("Correlation Matrix Heatmap (Bicycles)", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("No bicycle data found (type_most_common == 1).")
