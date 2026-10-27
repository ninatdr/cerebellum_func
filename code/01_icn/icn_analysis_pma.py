#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Network Strength (CNS) Analysis
=====================================

Analyses the relationship between core network strength (CNS) of dual-regression
derived resting-state networks and age at scan in term-born infants (dHCP dataset),
and produces the accompanying publication figures.

CNS values are computed beforehand with `core_network_strength.py`.

Pipeline
--------
1. Load CNS values and the clinical/demographic database; merge on subject ID.
2. PMA model: for each component, CNS ~ PMA at scan + sex + motion outliers.
   Effect of PMA tested with a partial Spearman correlation (controlling for sex and
   motion outliers); p-values FDR-corrected (Benjamini-Hochberg) across components.
3. Network flatmaps: cerebellar flatmap of each independent component network, plus a
   winner-take-all "atlas" map showing the dominant network per vertex.
4. Component maps: flatmap of a single component's thresholded GLM/dual-regression
   map, and the same map split out by postmenstrual week.
5. Scatterplots of CNS (residualised for nuisance covariates) vs. age at scan for
   FDR-significant components, annotated with partial Spearman correlations.

Usage
-----
Edit the CONFIG section below (paths, parameters and the RUN_* switches), then run:

    python core_network_strength_analysis.py

Requirements
------------
pandas, numpy, matplotlib, seaborn, statsmodels, pingouin, nibabel, SUITPy
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from SUITPy import flatmap

# --------------------------------------------------------------------------- #
# CONFIG - edit these paths/parameters before running
# --------------------------------------------------------------------------- #

# Optional steps (set to False to skip)
RUN_PMA_ANALYSIS = True      # PMA-at-scan statistics + scatterplots of significant components
RUN_NETWORK_MAPS = True      # network flatmaps + winner-take-all atlas
RUN_COMPONENT_MAPS = True    # single-component map + by-week maps

# Raw inputs used to build the merged CNS/clinical dataframe
CNS_CSV = "data/core_network_strengths.csv"   # per-subject CNS values (output of core_network_strength.py)
DATABASE_CSV = "data/db_outcome_imd.csv"      # clinical/demographic data

# Component indices to exclude from the CNS statistical analysis (noise/artefact ICs)
COMPONENTS_TO_DROP = [4, 8, 9, 11, 14, 15, 16, 20]

# Columns to keep from the clinical database
DATABASE_COLUMNS = [
    "id",
    "crbl",
    "redcap_repeat_instance",
    "baby_ga_at_birth_weeks",
    "nscan_ga_at_scan_weeks",
    "baby_gender",
    "radiology_score",
    "mean_fd",
    "fd_outliers",
    "pctl99_20pc",
]

ALPHA = 0.05
OUTPUT_DIR = "results"

# --- Network flatmaps ------------------------------------------------------
MELODIC_IC_MAP = "networks_to_surface/melodic_in_MNI_og.nii.gz"  # all-component IC map (MNI, surface-mapped)
NETWORK_COMPONENTS_TO_EXCLUDE = [4, 7, 8, 10, 13, 14, 15, 19]  # 0-indexed columns to drop from the IC map
NETWORK_LABELS = [
    "Left Crus I/II", "Right Crus I/II", "Left Crus I/II", "Right Crus I/II",
    "Right VI", "Left VI", "Bilateral V + Vermis", "Vermis VI-VIII",
    "Medial Crus I/II", "Right VIIb", "Left VIIIa", "Left V-VI",
]
NETWORK_CMAP = "hot"
NETWORK_VMIN, NETWORK_VMAX = 3, 10

# --- Single-component GLM map + by-week maps --------------------------------
DUAL_REGRESSION_DIR = "new_dr_feb2026/dual_regression"
SINGLE_COMPONENT_MAP = (
    "dual_regression_all_IC20_0-2hz_perm.dr/dr_stage3_ic0002_tstat1_thrSig_MNILin6aAsym.nii.gz"
)
SINGLE_MAP_VMIN, SINGLE_MAP_VMAX = 1, 4
WEEKS_TO_PLOT = [39, 40, 41, 42, 43]
WEEK_MAP_TEMPLATE = "comp2_group_maps/{week}wks_group_map_MNILin6aAsym.nii.gz"
WEEK_MAP_VMIN, WEEK_MAP_VMAX = 3, 12


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #

def load_and_merge_data(cns_csv, database_csv, components_to_drop, database_columns):
    """Load CNS and clinical data, drop excluded components, and merge on subject ID."""
    cns = pd.read_csv(cns_csv)
    for i in components_to_drop:
        col = f"Component_{i}"
        if col in cns.columns:
            cns = cns.drop(columns=col)

    db = pd.read_csv(database_csv)
    db_subset = db[database_columns].copy()

    merged = pd.merge(cns, db_subset, how="left", left_on="subject_id", right_on="id")
    merged["sex_bin"] = merged["baby_gender"].map({2: 1, 1: 0})
    merged["pnw"] = merged["nscan_ga_at_scan_weeks"] - merged["baby_ga_at_birth_weeks"]
    return merged


def get_component_columns(df):
    return df.columns[pd.Series(df.columns).str.startswith("Component")].tolist()


# --------------------------------------------------------------------------- #
# Statistics: PMA model
# --------------------------------------------------------------------------- #

def _extract_term_stats(model, term):
    """Pull beta, SE, t, p, and 95% CI for a single term from a fitted model."""
    conf = model.conf_int().loc[term]
    return {
        "beta": model.params[term],
        "se": model.bse[term],
        "t": model.tvalues[term],
        "p": model.pvalues[term],
        "ci_low": conf[0],
        "ci_high": conf[1],
    }


def fit_pma_models(df, comp_cols):
    """Fit CNS ~ PMA + sex + motion_outliers and calculate partial Spearman correlations."""
    results = {}
    for col in comp_cols:
        formula = f"{col} ~ nscan_ga_at_scan_weeks + sex_bin + fd_outliers"
        results[col] = smf.ols(formula, data=df).fit()

    term = "nscan_ga_at_scan_weeks"
    stats = {col: _extract_term_stats(model, term) for col, model in results.items()}

    rows = []
    for col in comp_cols:
        pcorr = pg.partial_corr(
            data=df,
            x=col,
            y="nscan_ga_at_scan_weeks",
            covar=["fd_outliers", "sex_bin"],
            method="spearman",
        )

        rows.append({
            "col": col,
            "beta": stats[col]["beta"],
            "se": stats[col]["se"],
            "t": stats[col]["t"],
            "ci_low": stats[col]["ci_low"],
            "ci_high": stats[col]["ci_high"],
            "partial_r": pcorr["r"].iloc[0],
            "partial_p": pcorr["p-val"].iloc[0],
        })

    pval_df = pd.DataFrame(rows)

    reject, p_fdr, _, _ = multipletests(
        pval_df["partial_p"],
        alpha=ALPHA,
        method="fdr_bh",
    )

    pval_df["partial_p_fdr"] = p_fdr
    pval_df["significant"] = reject

    return results, pval_df


def write_model_report(results, pval_df, out_path, csv_path=None):
    """Write corrected p-values and full OLS summaries to a text file, and the table to CSV."""
    with open(out_path, "w") as f:
        f.write("=== Corrected p-values ===\n")
        f.write(pval_df.to_string(index=False))
        f.write("\n\n=== Model summaries ===\n")
        for col, model in results.items():
            f.write(f"\n--- {col} ---\n")
            f.write(model.summary().as_text())
            f.write("\n")

    if csv_path is None:
        csv_path = str(out_path).rsplit(".", 1)[0] + ".csv"

    pval_df.to_csv(csv_path, index=False)

    return csv_path


# --------------------------------------------------------------------------- #
# Network flatmaps + winner-take-all atlas
# --------------------------------------------------------------------------- #

def plot_network_flatmaps(melodic_ic_map, exclude_idx, labels, cmap, vmin, vmax, out_path):
    """Plot each independent-component network as a separate cerebellar flatmap panel."""
    func_data = flatmap.vol_to_surf(melodic_ic_map, space="FSL")
    func_data_filtered = np.delete(func_data, exclude_idx, axis=1)
    n_components = func_data_filtered.shape[1]

    n_cols = 5
    n_rows = int(np.ceil(n_components / n_cols))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for i in range(n_components):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        plt.sca(ax)
        flatmap.plot(
            data=func_data_filtered[:, i], cmap=cmap, cscale=[vmin, vmax],
            threshold=vmin, new_figure=False, colorbar=False,
        )
        label = labels[i] if i < len(labels) else ""
        ax.set_title(f"ICN{i + 1:02d}: {label}", fontsize=20, pad=0)

    for j in range(n_components, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, j + 1).set_visible(False)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable, ax=fig.get_axes(), location="bottom", anchor=(0.8, 0.2),
        orientation="horizontal", ticks=[vmin, vmax], fraction=0.025, pad=0.02,
    )
    cbar.set_label("Z-score", fontsize=20, fontweight="bold")
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return func_data_filtered, n_components


def plot_winner_take_all_atlas(func_data_filtered, n_components, vmin, out_path):
    """Assign each surface vertex to its dominant network and plot as a labelled atlas."""
    func_thresh = func_data_filtered.copy()
    func_thresh[func_thresh < vmin] = 0

    winner = np.argmax(func_thresh, axis=1).astype(float)
    max_vals = np.max(func_thresh, axis=1)
    winner += 1
    winner[max_vals == 0] = np.nan  # no network exceeds threshold -> masked

    flatmap.plot(
        data=winner, overlay_type="label", cmap="tab20", cscale=[1, n_components],
        new_figure=True, colorbar=False, render="matplotlib",
    )

    cmap_cat = plt.get_cmap("tab20")
    colors = [cmap_cat(i + 1) for i in range(n_components)]
    cmap_custom = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, n_components + 1.5), ncolors=n_components)
    mappable = cm.ScalarMappable(cmap=cmap_custom, norm=norm)
    mappable.set_array([])

    labels = [f"ICN{i + 1:02d}" for i in range(n_components)]
    cbar = plt.colorbar(mappable, ax=plt.gca(), orientation="vertical", fraction=0.04, pad=0.04)
    cbar.set_ticks(np.arange(1, n_components + 1))
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Single-component GLM map + by-week maps
# --------------------------------------------------------------------------- #

def plot_single_component_map(map_path, cmap, vmin, vmax, out_path):
    """Plot one component's thresholded GLM/dual-regression map on the cerebellar flatmap."""
    img = nib.load(map_path)
    surf = flatmap.vol_to_surf(img)

    fig = plt.figure(figsize=(4, 4))
    flatmap.plot(data=surf, cmap=cmap, cscale=[vmin, vmax], threshold=vmin, new_figure=False, colorbar=False)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    cbar = fig.colorbar(
        mappable, ax=fig.get_axes(), location="right", orientation="vertical",
        ticks=[vmin, vmax], fraction=0.025, pad=0.02,
    )
    cbar.set_label("Z-score", fontsize=20, fontweight="bold")
    cbar.ax.tick_params(labelsize=20)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_maps(dual_regression_dir, week_map_template, weeks, cmap, vmin, vmax, out_path):
    """Plot the group map of a single component for each postmenstrual week."""
    n_cols = len(weeks)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

    for i, week in enumerate(weeks):
        map_path = os.path.join(dual_regression_dir, week_map_template.format(week=week))
        img = nib.load(map_path)
        surf = flatmap.vol_to_surf(img, space="FSL")

        plt.sca(axes[i])
        flatmap.plot(data=surf, cmap=cmap, cscale=[vmin, vmax], threshold=vmin, new_figure=False, colorbar=False)
        axes[i].set_title(f"Week {week}", fontsize=20, pad=0)
        axes[i].axis("off")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    cbar = fig.colorbar(
        mappable, ax=fig.get_axes(), location="right", orientation="vertical",
        ticks=[vmin, vmax], fraction=0.025, pad=0.02,
    )
    cbar.set_label("Z-score", fontsize=20, fontweight="bold")
    cbar.ax.tick_params(labelsize=20)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Significant-component scatterplots with partial correlation
# --------------------------------------------------------------------------- #

def plot_significant_components(df, results, pval_df, out_path):
    """Plot CNS (residualised for sex and motion outliers) vs. PMA at scan for
    FDR-significant components, annotated with the partial Spearman correlation and
    both raw and FDR-corrected p-values."""
    sig_cols = pval_df.loc[pval_df["significant"], "col"].tolist()
    if not sig_cols:
        print("No components survived FDR correction (PMA model).")
        return
    sig_nums = [col.split("_")[1] for col in sig_cols]

    sns.set_style("dark")
    fig, axes = plt.subplots(1, len(sig_cols), figsize=(5 * len(sig_cols), 6))
    if len(sig_cols) == 1:
        axes = [axes]

    for ax, col, num in zip(axes, sig_cols, sig_nums):
        x_cov = sm.add_constant(df[["fd_outliers", "sex_bin"]])
        y = df[col]
        cov_model = sm.OLS(y, x_cov).fit()

        df_plot = df.copy()
        df_plot[f"{col}_resid_cov"] = cov_model.resid

        pcorr = pg.partial_corr(
            data=df, x=col, y="nscan_ga_at_scan_weeks",
            covar=["fd_outliers", "sex_bin"], method="spearman",
        )
        r = pcorr["r"].values[0]
        p_uncorr = pval_df.loc[pval_df["col"] == col, "partial_p"].values[0]
        p_corr = pval_df.loc[pval_df["col"] == col, "partial_p_fdr"].values[0]

        sns.regplot(
            ax=ax, data=df_plot, x="nscan_ga_at_scan_weeks", y=f"{col}_resid_cov",
            scatter_kws=dict(color="black", alpha=0.8), line_kws=dict(color="blue", lw=2),
        )

        ax.set_title(f"ICN{int(num):02d}", fontsize=18, pad=25, weight="bold")
        ax.text(
            0.5, 1.02,
            f"r = {r:.3f}   |   p (uncorr.) = {p_uncorr:.3f}   |   p (FDR) = {p_corr:.3f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12, color="black",
        )
        ax.set_xlabel("PMA at scan (weeks)", fontsize=16)
        ax.set_ylabel("Core network strength | motion outliers, sex", fontsize=14)
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data prep -----------------------------------------------------------

    cns_db = load_and_merge_data(CNS_CSV, DATABASE_CSV, COMPONENTS_TO_DROP, DATABASE_COLUMNS)
    comp_cols = get_component_columns(cns_db)

    # --- PMA at scan: statistics + scatterplots -------------------------------
    if RUN_PMA_ANALYSIS:
        results_pma, pval_pma = fit_pma_models(cns_db, comp_cols)
        write_model_report(results_pma, pval_pma, os.path.join(OUTPUT_DIR, "ols_results_pma.txt"))
        print("=== PMA model: FDR-corrected p-values ===")
        print(pval_pma)

        plot_significant_components(
            cns_db, results_pma, pval_pma,
            out_path=os.path.join(OUTPUT_DIR, "fig2c_cns_vs_pma_significant.png"),
        )


    # --- Network flatmaps + atlas ---------------------------------------------
    if RUN_NETWORK_MAPS:
        func_data_filtered, n_components = plot_network_flatmaps(
            MELODIC_IC_MAP, NETWORK_COMPONENTS_TO_EXCLUDE, NETWORK_LABELS,
            NETWORK_CMAP, NETWORK_VMIN, NETWORK_VMAX,
            out_path=os.path.join(OUTPUT_DIR, "fig1a_networks.png"),
        )
        plot_winner_take_all_atlas(
            func_data_filtered, n_components, NETWORK_VMIN,
            out_path=os.path.join(OUTPUT_DIR, "fig1a_atlas.png"),
        )

    # --- Single-component map + by-week maps ----------------------------------
    if RUN_COMPONENT_MAPS:
        plot_single_component_map(
            os.path.join(DUAL_REGRESSION_DIR, SINGLE_COMPONENT_MAP),
            NETWORK_CMAP, SINGLE_MAP_VMIN, SINGLE_MAP_VMAX,
            out_path=os.path.join(OUTPUT_DIR, "fig2b_component_map.png"),
        )
        plot_weekly_maps(
            DUAL_REGRESSION_DIR, WEEK_MAP_TEMPLATE, WEEKS_TO_PLOT,
            NETWORK_CMAP, WEEK_MAP_VMIN, WEEK_MAP_VMAX,
            out_path=os.path.join(OUTPUT_DIR, "fig2b_weekly_maps.png"),
        )


if __name__ == "__main__":
    main()