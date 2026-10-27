#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster strength vs. age analysis
==================================

Tests how the connectivity strength of retained ICA-derived cerebellar clusters
relates to age in infants (dHCP dataset), and plots the FDR-significant effects.

Pipeline
--------
1. Load the clinical database and the per-subject cluster strengths for the retained
   (selected) components.
2. For every retained cluster, fit two models (OLS for effect sizes; partial Spearman
   correlation for inference):
     - PMA model:    strength ~ PMA at scan + motion outliers + sex
     - PNW/GA model: strength ~ postnatal weeks (PNW) + GA at birth + motion outliers + sex
       (partial correlations for PNW and for GA, each controlling for the other predictor)
3. FDR-correct (Benjamini-Hochberg) partial-correlation p-values across clusters,
   separately for each tested effect (PMA, PNW, GA).
4. Save the results tables and plot, for each FDR-significant cluster, strength
   (residualised for the covariates) against the predictor of interest.

Usage
-----
Edit the CONFIG section below (paths and the RUN_* switches), then run:

    python cluster_strength_pma_analysis.py

Requirements
------------
pandas, seaborn, matplotlib, statsmodels, pingouin (tested with 0.5.5)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------- #
# CONFIG - edit these paths/parameters before running
# --------------------------------------------------------------------------- #

# Optional steps (set to False to skip)
RUN_PMA_PLOTS = True       # scatterplots for FDR-significant PMA effects
RUN_PNW_GA_PLOTS = True    # scatterplots for FDR-significant PNW and GA effects
SHOW_PLOTS = True          # display figures interactively (figures are always saved)

# Inputs
DATABASE_CSV = "data/demographics.csv"        # clinical/demographic database
SUBJECT_LIST = "subject_list/subjects_subset2"                       # subjects to analyse, one ID per line
SELECTION_CSV = "data/02_crb_crbl/ica_component_selection.csv"           # columns: component, label, selected
CLUSTER_STRENGTH_CSV = "data/02_crb_crbl/d20_all_cluster_strengths.csv"  # columns: subject_id, cluster_01_mean ... cluster_20_mean

# Outputs
OUTPUT_DIR = "results"
PMA_RESULTS_CSV = "results_pma_retained_clusters_FDR.csv"

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


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_database(database_csv, columns):
    """Load the clinical database and derive binary sex and postnatal weeks (PNW)."""
    db = pd.read_csv(database_csv)
    db_subset = db[columns].copy()

    db_subset["sex_bin"] = db_subset["baby_gender"].map({2: 1, 1: 0})
    db_subset["pnw"] = (
        db_subset["nscan_ga_at_scan_weeks"]
        - db_subset["baby_ga_at_birth_weeks"]
    )
    return db_subset


def load_cluster_data(selection_csv, cluster_strength_csv, subject_list):
    """Return {component label: DataFrame(subject_id, tot_strength)} for retained components."""
    with open(subject_list, "r") as f:
        subject_ids = [s.strip() for s in f.readlines()]

    selection = pd.read_csv(selection_csv)
    retained = selection[selection["selected"]].copy()
    component_labels = dict(zip(retained["component"], retained["label"]))

    strengths = pd.read_csv(cluster_strength_csv)

    # restrict to subjects actually being analysed
    strengths = strengths[strengths["subject_id"].isin(subject_ids)].copy()

    cluster_data = {}
    for _, row in retained.iterrows():
        comp = int(row["component"])

        # component 0-19 --> cluster_01-20
        col = f"cluster_{comp + 1:02d}_mean"

        if col not in strengths.columns:
            print(f"Missing column: {col}")
            continue

        cluster_df = strengths[["subject_id", col]].rename(columns={col: "tot_strength"})
        label = component_labels.get(comp, f"component_{comp:02d}")
        cluster_data[label] = cluster_df

    print(f"{len(cluster_data)} retained components loaded")
    return cluster_data


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def fit_models(cluster_data, db_subset):
    """Fit the PMA and PNW/GA models for every retained cluster."""
    pma_results = []
    pnwga_results = []

    for cluster_name, cluster_df in cluster_data.items():

        cluster_db = pd.merge(
            cluster_df,
            db_subset,
            how="left",
            left_on="subject_id",
            right_on="id",
        )

        # --- PMA model -------------------------------------------------------
        ols_pma = smf.ols(
            "tot_strength ~ nscan_ga_at_scan_weeks + fd_outliers + sex_bin",
            data=cluster_db,
        ).fit()

        pcorr_pma = pg.partial_corr(
            data=cluster_db,
            x="tot_strength",
            y="nscan_ga_at_scan_weeks",
            covar=["fd_outliers", "sex_bin"],
            method="spearman",
        )

        pma_results.append({
            "cluster": cluster_name,
            "Partial_r": pcorr_pma["r"].values[0],
            "Partial_p": pcorr_pma["p-val"].values[0],
            "Beta_PMA": ols_pma.params["nscan_ga_at_scan_weeks"],
            "P_PMA": ols_pma.pvalues["nscan_ga_at_scan_weeks"],
            "R2": ols_pma.rsquared,
            "_ols": ols_pma,
            "_cluster_db": cluster_db,
        })

        
    return pd.DataFrame(pma_results)


def add_fdr_correction(df_pma):
    """Benjamini-Hochberg FDR across clusters, separately for PMA, PNW and GA effects."""
    reject, p_fdr, _, _ = multipletests(df_pma["Partial_p"], alpha=ALPHA, method="fdr_bh")
    df_pma["Partial_p_FDR"] = p_fdr
    df_pma["FDR_sig"] = reject

   
    return df_pma


def save_results(df_pma, df_pnwga, output_dir):
    df_pma.drop(columns=["_ols", "_cluster_db"]).to_csv(
        os.path.join(output_dir, PMA_RESULTS_CSV), index=False
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def format_pval_latex(p):
    if p < 0.001:
        return r"<0.001"
    return f"={p:.3f}"


def plot_partial_effect(cluster_db, x_var, covariates, resid_col, title, xlabel, ylabel,
                        ylabel_size, r, p_unc, p_fdr, out_path):
    """Scatterplot of cluster strength (residualised for `covariates`) vs. `x_var`,
    annotated with the partial correlation and uncorrected / FDR-corrected p-values."""
    x_cov = sm.add_constant(cluster_db[covariates])
    resid_model = sm.OLS(cluster_db["tot_strength"], x_cov, missing="drop").fit()
    cluster_db = cluster_db.copy()
    cluster_db[resid_col] = resid_model.resid

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.regplot(
        ax=ax, data=cluster_db,
        x=x_var, y=resid_col,
        scatter_kws=dict(color="black", alpha=0.8),
        line_kws=dict(color="blue", lw=2),
    )
    ax.set_title(title, fontsize=20, pad=30, weight="bold")
    ax.text(
        0.5, 1.02,
        rf"$r={float(r):.3f}"
        rf"\;|\;P_{{\mathrm{{UNC}}}}{format_pval_latex(float(p_unc))}"
        rf"\;|\;P_{{\mathrm{{FDR}}}}{format_pval_latex(float(p_fdr))}$",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=16, color="black",
    )
    ax.tick_params(labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_pma_effects(df_pma, output_dir):
    """PMA model: only clusters significant after FDR correction."""
    for _, row in df_pma[df_pma["FDR_sig"]].iterrows():
        name = row["cluster"]
        plot_partial_effect(
            row["_cluster_db"],
            x_var="nscan_ga_at_scan_weeks",
            covariates=["fd_outliers", "sex_bin"],
            resid_col="strength_resid",
            title=name,
            xlabel="PMA at scan (weeks)",
            ylabel="Cluster strength | motion outliers, sex",
            ylabel_size=16,
            r=row["Partial_r"], p_unc=row["Partial_p"], p_fdr=row["Partial_p_FDR"],
            out_path=os.path.join(output_dir, f"plot_{name}_pma.png"),
        )

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    db_subset = load_database(DATABASE_CSV, DATABASE_COLUMNS)
    cluster_data = load_cluster_data(SELECTION_CSV, CLUSTER_STRENGTH_CSV, SUBJECT_LIST)

    df_pma = fit_models(cluster_data, db_subset)
    df_pma = add_fdr_correction(df_pma)

    save_results(df_pma, OUTPUT_DIR)
    print(df_pma)

    sns.set_style("dark")
    if RUN_PMA_PLOTS:
        plot_pma_effects(df_pma, OUTPUT_DIR)


if __name__ == "__main__":
    main()