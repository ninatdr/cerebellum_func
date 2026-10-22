#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:34:55 2026

@author: ninatreder
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cerebellar gradient analysis
────────────────────────────
Intra-cerebellar (IC) and cerebro-cerebellar (CC) gradients,
network-level visualisations, and age (PMA) effects.

Network 0 (Misc) is excluded from all plots and stats.
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import SUITPy.flatmap as flatmap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr
from statsmodels.stats.multitest import fdrcorrection

# ── Paths ──────────────────────────────────────────────────────────────────
FOLDER        = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/03_gradients"
SUBJECT_LIST  = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/subject_list/subjects_37_to_42_grad"
DB_PATH       = "/Users/ninatreder/Desktop/dHCP/DB_with_cerebellum_coverage.csv"
ATLAS_PATH    = "/path/to/templates/MNI_transform/atl-Buckner7_space-MNI_dseg_actualMNI.nii.gz"
LAMBDAS_PATH  = os.path.join(FOLDER, "CC_group_lambdas.txt")
OUTPUT_FOLDER = FOLDER

# Buckner 7-network atlas volumes (for supplementary flatmap check)
LOOSE_MASK_PATH = "/path/to/Buckner_JNeurophysiol11_MNI152/Buckner2011_7Networks_MNI152_FreeSurferConformed1mm_LooseMask.nii.gz"
TIGHT_MASK_PATH = "/path/to/Buckner_JNeurophysiol11_MNI152/Buckner2011_7Networks_MNI152_FreeSurferConformed1mm_TightMask.nii.gz"

# ── Subject IDs ────────────────────────────────────────────────────────────
with open(SUBJECT_LIST, 'r') as f:
    subject_IDs = [s.strip() for s in f.readlines()]

# ── Load gradients ─────────────────────────────────────────────────────────
crbl_gradient      = np.load(os.path.join(FOLDER, "CC_group_gradients_vert.npy"))
intracrbl_gradient = np.load(os.path.join(FOLDER, "IC_group_gradients_vert.npy"))

# Flip gradient I of intra-cerebellar (sign is arbitrary from decomposition;
# flipped here for consistent interpretation with cerebro-cerebellar gradients)
intracrbl_gradient = intracrbl_gradient.copy()
intracrbl_gradient[:, 0] *= -1

# ── Atlas flatmap ──────────────────────────────────────────────────────────
atlas         = nib.load(ATLAS_PATH)
atlas_flatmap = flatmap.vol_to_surf(atlas, space='FSL', stats='mode').flatten()

# ── Network definitions (network 0 / Misc excluded from analysis) ─────────
ALL_NETWORKS  = [0, 1, 2, 3, 4, 5, 6, 7]
networks      = [1, 2, 3, 4, 5, 6, 7]  # Misc excluded
network_names = ["Misc", "Visual", "Somatomotor", "Ventral Attention",
                  "Dorsal Attention", "Limbic", "Frontoparietal", "Default Mode"]
hex_colours   = ["#000000", "#890086", "#2c84b8", "#d528ff",
                  "#007800", "#d4f998", "#f49000", "#df2c4a"]

network_map      = dict(zip(ALL_NETWORKS, hex_colours))
network_name_map = dict(zip(ALL_NETWORKS, network_names))
gradient_labels  = ["I", "II", "III", "IV"]

# ── Mask (exclude background AND network 0) ────────────────────────────────
atlas_flatmap_1d = atlas_flatmap.flatten()
mask = (atlas_flatmap_1d >= 0) & (atlas_flatmap_1d != 0)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def make_legend(network_map, network_name_map, networks):
    return [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=network_map[n], markersize=4,
               label=network_name_map[n])
        for n in networks
    ]


def scatter_gradients(ax, x, y, atlas_flatmap, network_map, xlabel, ylabel):
    df = pd.DataFrame({"x": x, "y": y, "network": atlas_flatmap}).sample(frac=1, random_state=42)
    sns.scatterplot(data=df, x="x", y="y", hue="network", palette=network_map,
                     s=10, alpha=1, linewidth=0, legend=False, ax=ax)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_legend_and_show(fig, network_map, network_name_map, networks):
    fig.legend(handles=make_legend(network_map, network_name_map, networks),
               loc='center right', ncol=1, fontsize=20, frameon=False, markerscale=4)
    plt.tight_layout(rect=[0, 0, 0.85, 1], pad=2.5)
    plt.show()


def scatter_cross(ax, x, y, networks_masked, network_map, xlabel, ylabel, r):
    df = pd.DataFrame({"x": x, "y": y, "network": networks_masked}).sample(frac=1, random_state=42)
    sns.scatterplot(data=df, x="x", y="y", hue="network", palette=network_map,
                     s=10, alpha=1, linewidth=0, legend=False, ax=ax)
    ax.text(0.03, 0.99, f"r = {r:.3f}", transform=ax.transAxes, fontsize=24,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis='both', labelsize=22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_bottom_legend(fig, network_map, network_name_map, networks):
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=network_map[n],
               markersize=8, label=network_name_map[n])
        for n in networks
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(networks),
               fontsize=24, frameon=False, markerscale=2)


def plot_ridges_v1_median(gradient_data, atlas_flatmap_1d, mask,
                           network_map, network_name_map,
                           networks, gradient_labels,
                           title_prefix, sorted_networks,
                           n_gradients=4):
    """Ridge plot with median line per network (fixed ordering, Misc excluded)."""
    plt.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(1, n_gradients, figsize=(24, 6))
    y_positions = {n: i + 1 for i, n in enumerate(sorted_networks)}

    for i, ax in enumerate(axes):
        g_m   = gradient_data[:, i][mask]
        net_m = atlas_flatmap_1d[mask].astype(int)
        df    = pd.DataFrame({"gradient": g_m, "network": net_m})
        x_abs_max = np.abs(df["gradient"]).max()
        x_range   = np.linspace(-x_abs_max, x_abs_max, 300)

        for n in sorted_networks:
            vals = df[df["network"] == n]["gradient"].values
            if len(vals) < 5:
                continue
            y_pos = y_positions[n]
            color = network_map[n]
            kde   = gaussian_kde(vals, bw_method=0.3)
            kde_vals = kde(x_range) / kde(x_range).max() * 0.7
            ax.fill_between(x_range, y_pos, y_pos + kde_vals, color=color, alpha=0.8)
            ax.plot(x_range, y_pos + kde_vals, color=color, linewidth=0.8)
            median        = np.median(vals)
            median_height = kde(np.array([median]))[0] / kde(x_range).max() * 0.7
            ax.vlines(median, y_pos, y_pos + median_height, color="white", linewidth=2)

        ax.set_xlim(-x_abs_max, x_abs_max)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"{title_prefix} Gradient {gradient_labels[i]}", fontsize=24)
        ax.set_yticks(range(1, len(sorted_networks) + 1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=18)

        # Only show y-tick labels on the first subplot
        if i == 0:
            ax.set_yticklabels([network_name_map[n] for n in sorted_networks], fontsize=24)
        else:
            ax.set_yticklabels([])

    plt.tight_layout(w_pad=3)
    plt.show()


def plot_heatmap_v5(gradient_data, atlas_flatmap_1d, mask, network_map, network_name_map,
                     networks, gradient_labels, title_prefix, sorted_networks, n_gradients=4):
    """Heatmap of median gradient value per network (Misc excluded)."""
    medians = np.array([
        [np.median(gradient_data[:, i][mask][atlas_flatmap_1d[mask].astype(int) == n])
         for i in range(n_gradients)]
        for n in sorted_networks
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(medians,
                xticklabels=[f"Gradient {l}" for l in gradient_labels],
                yticklabels=[network_name_map[n] for n in sorted_networks],
                cmap='coolwarm', center=0, ax=ax, annot=True, fmt='.2f',
                linewidths=0.5, linecolor='white')
    ax.tick_params(labelsize=14)
    ax.set_title(f"{title_prefix} — Median gradient value per network", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_heatmap_zscored(gradient_data, atlas_flatmap_1d, mask, network_map, network_name_map,
                          networks, gradient_labels, title_prefix, sorted_networks, n_gradients=4):
    """Heatmap of z-scored median gradient value per network (Misc excluded)."""
    medians = np.array([
        [np.median(gradient_data[:, i][mask][atlas_flatmap_1d[mask].astype(int) == n])
         for i in range(n_gradients)]
        for n in sorted_networks
    ])
    # z-score across networks (per gradient column)
    medians_z = (medians - medians.mean(axis=0)) / medians.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(medians_z,
                xticklabels=[f"Gradient {l}" for l in gradient_labels],
                yticklabels=[network_name_map[n] for n in sorted_networks],
                cmap='coolwarm', center=0, ax=ax, annot=True, fmt='.2f',
                linewidths=0.5, linecolor='white')
    ax.tick_params(labelsize=14)
    ax.set_title(f"{title_prefix} — Z-scored median gradient value per network", fontsize=18)
    plt.tight_layout()
    plt.show()


def format_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — VARIANCE EXPLAINED (scree / cumulative)
# ═══════════════════════════════════════════════════════════════════════════

with open(LAMBDAS_PATH, 'r') as f:
    lambdas_arr = np.array([s.strip() for s in f.readlines()]).astype(float)

perc_explained  = lambdas_arr / np.sum(lambdas_arr) * 100
cumulative      = np.cumsum(perc_explained)
cumulative_plot = np.concatenate([[0], cumulative])
x_range_cum     = list(range(len(cumulative_plot)))

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.lineplot(x=x_range_cum, y=cumulative_plot, color='black', marker='o', ax=ax)
ax.set_xticks(x_range_cum)
ax.tick_params(labelsize=12)
ax.set_xlim(-0.5, 10)
ax.set_ylim(0, None)
ax.set_xlabel("Gradients", fontsize=20)
ax.set_ylabel("Cumulative variance\nexplained (%)", fontsize=20)
ax.axvspan(0.5, 4.5, color='red', alpha=0.3)
for x, y in zip(x_range_cum, cumulative_plot):
    if x == 0:
        continue
    ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(-10, 6),
                textcoords='offset points', fontsize=12, ha='center', va='bottom')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCATTERPLOTS (within-gradient-type pairs)
# ═══════════════════════════════════════════════════════════════════════════

sns.set_style("white")
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

# Intra-cerebellar: all pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (i, j) in zip(axes.flatten(), pairs):
    scatter_gradients(ax, intracrbl_gradient[:, i], intracrbl_gradient[:, j],
                       atlas_flatmap, network_map,
                       f"Intra-cerebellar gradient {gradient_labels[i]}",
                       f"Intra-cerebellar gradient {gradient_labels[j]}")
add_legend_and_show(fig, network_map, network_name_map, networks)

# Cerebro-cerebellar: all pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (i, j) in zip(axes.flatten(), pairs):
    scatter_gradients(ax, crbl_gradient[:, i], crbl_gradient[:, j],
                       atlas_flatmap, network_map,
                       f"Cerebro-cerebellar gradient {gradient_labels[i]}",
                       f"Cerebro-cerebellar gradient {gradient_labels[j]}")
add_legend_and_show(fig, network_map, network_name_map, networks)

# Intra-cerebellar: gradient I vs II, III, IV
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 5))
for ax, j in zip(axes, [1, 2, 3]):
    scatter_gradients(ax, intracrbl_gradient[:, j], intracrbl_gradient[:, 0],
                       atlas_flatmap, network_map,
                       f"Intra-cerebellar gradient {gradient_labels[j]}",
                       "Intra-cerebellar gradient I")
add_legend_and_show(fig, network_map, network_name_map, networks)

# Cerebro-cerebellar: gradient I vs II, III, IV
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 5))
for ax, j in zip(axes, [1, 2, 3]):
    scatter_gradients(ax, crbl_gradient[:, j], crbl_gradient[:, 0],
                       atlas_flatmap, network_map,
                       f"Cerebro-cerebellar gradient {gradient_labels[j]}",
                       "Cerebro-cerebellar gradient I")
add_legend_and_show(fig, network_map, network_name_map, networks)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CROSS-GRADIENT SCATTERPLOTS (IC vs CC)
# ═══════════════════════════════════════════════════════════════════════════

# 4×4 grid: all IC vs all CC
# NB: sign flips below correct for arbitrary gradient polarity from the
# decomposition so that cross-gradient relationships are interpretable.
flip_plots = [1, 2, 4, 7, 8, 10, 15]

fig, axes = plt.subplots(4, 4, figsize=(24, 24))
axes = axes.flatten()

for plot_idx, (j, i) in enumerate((j, i) for j in range(4) for i in range(4)):
    x = intracrbl_gradient[:, i].copy()
    y = crbl_gradient[:, j]
    if plot_idx in flip_plots:
        x = -x
    x_m   = x[mask]
    y_m   = y[mask]
    net_m = atlas_flatmap_1d[mask].astype(int)
    r, _  = spearmanr(x_m, y_m)
    scatter_cross(axes[plot_idx], x_m, y_m, net_m, network_map,
                  f"Intra-cerebellar Gradient {gradient_labels[i]}",
                  f"Cerebro-cerebellar Gradient {gradient_labels[j]}", r)

add_bottom_legend(fig, network_map, network_name_map, networks)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# IC Gradient I vs all CC
flip_ic_vs_cc = [1, 2]

fig, axes = plt.subplots(4, 1, figsize=(6, 24))
for j in range(4):
    x = intracrbl_gradient[:, 0].copy()
    y = crbl_gradient[:, j]
    if j in flip_ic_vs_cc:
        x = -x
    x_m, y_m = x[mask], y[mask]
    net_m    = atlas_flatmap_1d[mask].astype(int)
    r, _     = spearmanr(x_m, y_m)
    xlabel = "Intra-cerebellar Gradient I" if j == 3 else ""
    scatter_cross(axes[j], x_m, y_m, net_m, network_map,
                  xlabel, f"Cerebro-cerebellar \n Gradient {gradient_labels[j]}", r)
plt.subplots_adjust(hspace=0.3)
plt.show()

# CC Gradient I vs all IC
flip_cc_vs_ic = [1, 2]

fig, axes = plt.subplots(4, 1, figsize=(6, 24))
for i in range(4):
    x = crbl_gradient[:, 0].copy()
    y = intracrbl_gradient[:, i]
    if i in flip_cc_vs_ic:
        x = -x
    x_m, y_m = x[mask], y[mask]
    net_m    = atlas_flatmap_1d[mask].astype(int)
    r, _     = spearmanr(x_m, y_m)
    xlabel = "Cerebro-cerebellar Gradient I" if i == 3 else ""
    scatter_cross(axes[i], x_m, y_m, net_m, network_map,
                  xlabel, f"Intra-cerebellar \n Gradient {gradient_labels[i]}", r)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — RIDGE & HEATMAP PLOTS
# ═══════════════════════════════════════════════════════════════════════════

# Derive sorted network order from IC gradient I (Misc already excluded by mask)
g1    = intracrbl_gradient[:, 0][mask]
net_m = atlas_flatmap_1d[mask].astype(int)
df_g1 = pd.DataFrame({"gradient": g1, "network": net_m})

network_medians = {
    n: np.median(df_g1[df_g1["network"] == n]["gradient"].values)
    for n in networks
    if len(df_g1[df_g1["network"] == n]) >= 5
}

sorted_networks = sorted(network_medians, key=lambda n: network_medians[n], reverse=False)

for gradient_data, title_prefix in [
    (intracrbl_gradient, "Intra-cerebellar"),
    (crbl_gradient,      "Cerebro-cerebellar"),
]:
    plot_ridges_v1_median(gradient_data, atlas_flatmap_1d, mask, network_map,
                           network_name_map, networks, gradient_labels,
                           title_prefix, sorted_networks)

    plot_heatmap_zscored(gradient_data, atlas_flatmap_1d, mask, network_map,
                          network_name_map, networks, gradient_labels,
                          title_prefix, sorted_networks)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PMA (AGE) EFFECTS
# ═══════════════════════════════════════════════════════════════════════════

GRAD  = 'CC'
SPACE = 'volume'

corr_surf_test = pd.read_csv(os.path.join(FOLDER, f"{GRAD}_corr_{SPACE}.csv"), index_col=0)
db             = pd.read_csv(DB_PATH, index_col=0)

db_merged = pd.merge(corr_surf_test[["g1", "g2", "g3", "g4"]], db,
                      how="left", left_index=True, right_index=True)
test_db   = pd.merge(pd.DataFrame(subject_IDs), db_merged,
                      how="left", left_on=0, right_on="subject")

test_db['pnw'] = test_db['nscan_ga_at_scan_weeks'] - test_db['baby_ga_at_birth_weeks']

results = []
for i in range(1, 5):
    y_var = f"g{i}"
    pcorr = pg.partial_corr(data=test_db, x=y_var, y="nscan_ga_at_scan_weeks",
                             covar=['fd_outliers', 'baby_gender'], method='spearman')
    r         = pcorr['r'].values[0]
    p_partial = pcorr['p-val'].values[0]

    X     = sm.add_constant(test_db[["nscan_ga_at_scan_weeks", "fd_outliers", "baby_gender"]])
    model = sm.OLS(test_db[y_var], X, missing="drop").fit()

    results.append({
        "Gradient":  i,
        "Partial_r": r,
        "Partial_p": p_partial,
        "Beta_PMA":  model.params["nscan_ga_at_scan_weeks"],
        "P_PMA":     model.pvalues["nscan_ga_at_scan_weeks"],
        "R2":        model.rsquared,
    })

df_results = pd.DataFrame(results)
_, p_fdr = fdrcorrection(df_results["P_PMA"])
df_results["P_PMA_FDR"] = p_fdr
_, p_fdr = fdrcorrection(df_results["Partial_p"])
df_results["Partial_p_FDR"] = p_fdr

print(df_results)
df_results.to_csv(os.path.join(OUTPUT_FOLDER, f"{SPACE}_results_pma_{GRAD}.csv"))

# Plot PMA regression
fig, axes = plt.subplots(2, 2, sharey=True, figsize=(10, 9))
axes = axes.flatten()

for i in range(1, 5):
    ax = axes[i - 1]

    sns.regplot(data=test_db, x="nscan_ga_at_scan_weeks", y=f"g{i}",
                scatter_kws=dict(color='black', alpha=0.8),
                line_kws=dict(color="blue", lw=2), ax=ax)

    ax.set_title(f"Intra-cerebellar gradient {gradient_labels[i - 1]}",
                 fontsize=20, pad=30, weight="bold")

    p_unc = float(df_results['Partial_p'][i - 1])
    p_fdr = float(df_results['Partial_p_FDR'][i - 1])

    ax.text(
        0.5, 1.02,
        rf"$r={float(df_results['Partial_r'][i - 1]):.3f}"
        rf"\;|\;P_{{\mathrm{{UNC}}}} {format_p(p_unc)}"
        rf"\;|\;P_{{\mathrm{{FDR}}}} {format_p(p_fdr)}$",
        transform=ax.transAxes, ha='center', va='bottom', fontsize=16,
    )

    ax.tick_params(labelsize=16)
    ax.set_xlabel("PMA at scan (weeks)", fontsize=20)

    # Only left column gets y-label
    ax.set_ylabel("Gradient similarity" if i in [1, 3] else "")
    ax.grid(False)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# PNW + GA model

results = []
for i in range(1, 5):

    y_var = f"g{i}"

    # Partial correlation: PNW controlling for GA, motion, sex
    pcorr_pnw = pg.partial_corr(
        data=test_db,
        x=y_var,
        y="pnw",
        covar=["baby_ga_at_birth_weeks", "fd_outliers", "baby_gender"],
        method="spearman",
    )

    # Partial correlation: GA controlling for PNW, motion, sex
    pcorr_ga = pg.partial_corr(
        data=test_db,
        x=y_var,
        y="baby_ga_at_birth_weeks",
        covar=["pnw", "fd_outliers", "baby_gender"],
        method="spearman",
    )

    # Full regression model
    X = sm.add_constant(
        test_db[
            [
                "pnw",
                "baby_ga_at_birth_weeks",
                "fd_outliers",
                "baby_gender",
            ]
        ]
    )

    model = sm.OLS(
        test_db[y_var],
        X,
        missing="drop"
    ).fit()

    results.append({
        "Gradient": i,

        "PNW_partial_r": pcorr_pnw["r"].iloc[0],
        "PNW_partial_p": pcorr_pnw["p-val"].iloc[0],

        "GA_partial_r": pcorr_ga["r"].iloc[0],
        "GA_partial_p": pcorr_ga["p-val"].iloc[0],

        "R2": model.rsquared,
    })

df_results = pd.DataFrame(results)

# FDR correction for PNW effects
_, p_fdr = fdrcorrection(df_results["PNW_partial_p"])
df_results["PNW_partial_p_FDR"] = p_fdr

# FDR correction for GA effects
_, p_fdr = fdrcorrection(df_results["GA_partial_p"])
df_results["GA_partial_p_FDR"] = p_fdr

print(df_results)

df_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        f"{SPACE}_results_pnw_ga_{GRAD}.csv"
    ),
    index=False,
)

results = []

for i in range(1, 5):

    y_var = f"g{i}"

    # PNW effect controlling for GA + covariates
    pcorr_pnw = pg.partial_corr(
        data=test_db,
        x=y_var,
        y="pnw",
        covar=["baby_ga_at_birth_weeks", "fd_outliers", "baby_gender"],
        method="spearman",
    )

    # GA effect controlling for PNW + covariates
    pcorr_ga = pg.partial_corr(
        data=test_db,
        x=y_var,
        y="baby_ga_at_birth_weeks",
        covar=["pnw", "fd_outliers", "baby_gender"],
        method="spearman",
    )

    X = sm.add_constant(
        test_db[
            ["pnw", "baby_ga_at_birth_weeks", "fd_outliers", "baby_gender"]
        ]
    )

    model = sm.OLS(test_db[y_var], X, missing="drop").fit()

    results.append({
        "Gradient": i,
        "PNW_Partial_r": pcorr_pnw["r"].iloc[0],
        "PNW_Partial_p": pcorr_pnw["p-val"].iloc[0],
        "GA_Partial_r": pcorr_ga["r"].iloc[0],
        "GA_Partial_p": pcorr_ga["p-val"].iloc[0],
        "R2": model.rsquared,
    })

df_results = pd.DataFrame(results)

_, p_fdr = fdrcorrection(df_results["PNW_Partial_p"])
df_results["PNW_Partial_p_FDR"] = p_fdr

_, p_fdr = fdrcorrection(df_results["GA_Partial_p"])
df_results["GA_Partial_p_FDR"] = p_fdr

print(df_results)

df_results.to_csv(
    os.path.join(OUTPUT_FOLDER, f"{SPACE}_results_pnw_ga_{GRAD}.csv"),
    index=False,
)


