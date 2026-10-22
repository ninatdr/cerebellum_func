"""
Cluster Strength vs. PMA at Scan
==================================

Computes total connectivity strength within a functional cluster (defined by a
binary mask in cerebellar space) for each subject, then tests whether that
strength is associated with postmenstrual age (PMA) at scan, controlling for
motion outliers and sex, and produces the accompanying scatterplot.

Pipeline
--------
0. (Optional) Compute per-subject cluster strength: for each subject's
   cerebrum-to-cerebellum functional connectivity matrix, sum the (positive)
   connectivity values falling within a binary cluster mask. Skip this step if
   CLUSTER_PATH already exists.
1. Load the clinical/demographic database and per-subject cluster strength
   values, restricting to a given subject list.
2. Merge clinical covariates onto the cluster data.
3. Fit tot_strength ~ PMA_at_scan + motion_outliers + sex (OLS).
4. Compute the partial Spearman correlation between cluster strength and PMA,
   controlling for motion outliers and sex.
5. Save model/partial-correlation results to CSV.
6. Plot cluster strength (residualised for covariates) vs. PMA at scan.

Usage
-----
Edit the paths in the CONFIG section below, then run:

    python cluster_strength_pma_analysis.py

Requirements
------------
numpy, pandas, nibabel, matplotlib, seaborn, statsmodels, pingouin
"""

import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --------------------------------------------------------------------------- #
# CONFIG - edit these paths/parameters before running
# --------------------------------------------------------------------------- #

BASE_DIR = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/02_FC_conn/postnatal_weeks"
DB_PATH = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/db_outcome_imd.csv"
CLUSTER_PATH = os.path.join(BASE_DIR, "cluster_res_subset1_thr0.csv")
SUBJECT_LIST_PATH = os.path.join(BASE_DIR, "subjects_37_to_42_grad.txt")
OUT_PATH = "results/results_pma_cluster1_thr0.csv"
PLOT_PATH = "results/cluster1_strength_vs_pma.png"

# Whether to (re)compute cluster strength from subject FC matrices before
# running the PMA analysis. If False, CLUSTER_PATH is loaded as-is.
COMPUTE_CLUSTER_STRENGTH = False

# --- Inputs for the cluster strength computation (only used if
#     COMPUTE_CLUSTER_STRENGTH is True) ---------------------------------------
CLUSTER_MASK_PATH = "data/clustering/crb_crbl_ICA_allconn0003_thr3_bin.nii.gz"  # binary cluster mask, cerebellar space
CEREBELLUM_MASK_PATH = "masks/crb_dhcp_2mm.nii.gz"                              # binary cerebellar mask
SUBJECT_FC_MATRIX_DIR = "data/subject_FC_matrices"                              # per-subject .npy FC matrices
ALL_SUBJECTS_LIST_PATH = "data/all_subjects.txt"                                # subjects to compute strength for
N_CEREBELLUM = 2881                                                             # cerebellar voxel count (columns)

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


# --------------------------------------------------------------------------- #
# Cluster strength computation
# --------------------------------------------------------------------------- #

def compute_cluster_strength(
    cluster_mask_path, cerebellum_mask_path, subject_fc_matrix_dir,
    subject_list_path, n_cerebellum, out_path,
):
    """For each subject, sum the positive cerebrum-to-cerebellum connectivity
    values falling within a binary cluster mask, and save subject_id/tot_strength
    to out_path."""
    cluster_mask = nib.load(cluster_mask_path).get_fdata()
    cerebellum_mask = nib.load(cerebellum_mask_path).get_fdata()

    n_cerebrum = cerebellum_mask[cerebellum_mask == 1].size
    cluster_filter = cluster_mask[cerebellum_mask == 1].astype("bool")

    with open(subject_list_path) as f:
        subject_ids = f.read().splitlines()

    results = {}
    for subject_id in subject_ids:
        print(f"Processing {subject_id}...")
        match = glob.glob(os.path.join(subject_fc_matrix_dir, f"{subject_id}*.npy"))[0]
        fc_matrix = np.load(match)
        fc_matrix[fc_matrix < 0] = 0

        cerebrum_to_cerebellum = fc_matrix[:n_cerebrum, -n_cerebellum:]
        cluster_values = cerebrum_to_cerebellum[cluster_filter]
        results[subject_id] = np.sum(cluster_values)

    results_df = (
        pd.DataFrame.from_dict(results, orient="index", columns=["tot_strength"])
        .rename_axis("subject_id")
        .reset_index()
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    return results_df


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #

def load_and_merge_data(db_path, cluster_path, subject_list_path, database_columns):
    """Load clinical DB and cluster strength data, restrict to the given subject
    list, and merge clinical covariates onto the cluster data."""
    db = pd.read_csv(db_path)
    cluster = pd.read_csv(cluster_path)

    with open(subject_list_path, "r") as f:
        subject_ids = [s.strip() for s in f.readlines()]
    cluster_filt = cluster.loc[cluster["subject_id"].isin(subject_ids)]

    db_subset = db.loc[:, database_columns].copy()
    db_subset["sex_bin"] = db_subset["baby_gender"].map({2: 1, 1: 0})
    db_subset['pnw'] = db_subset['nscan_ga_at_scan_weeks'] - db_subset['baby_ga_at_birth_weeks']

    cluster_db = pd.merge(cluster_filt, db_subset, how="left", left_on="subject_id", right_on="id")
    return cluster_db


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def fit_pma_model(cluster_db):
    """Fit tot_strength ~ PMA_at_scan + motion_outliers + sex."""
    formula = "tot_strength ~ nscan_ga_at_scan_weeks + fd_outliers + sex_bin"
    lm = smf.ols(formula, data=cluster_db).fit()
    print(lm.summary())
    return lm


def compute_partial_correlation(cluster_db):
    """Partial Spearman correlation between cluster strength and PMA, controlling
    for motion outliers and sex."""
    vars_needed = ["tot_strength", "nscan_ga_at_scan_weeks", "fd_outliers", "sex_bin"]
    df_pcorr = cluster_db[vars_needed].apply(pd.to_numeric, errors="coerce").dropna()

    pcorr = pg.partial_corr(
        data=df_pcorr,
        x="tot_strength",
        y="nscan_ga_at_scan_weeks",
        covar=["fd_outliers", "sex_bin"],
        method="spearman",
    )
    r = pcorr["r"].values[0]
    p_partial = pcorr["p-val"].values[0]
    return r, p_partial


def save_results(lm, r, p_partial, out_path):
    df_results = pd.DataFrame(
        [{
            "Partial_r": r,
            "Partial_p": p_partial,
            "Beta_PMA": lm.params["nscan_ga_at_scan_weeks"],
            "P_PMA": lm.pvalues["nscan_ga_at_scan_weeks"],
            "R2": lm.rsquared,
        }]
    )
    df_results.to_csv(out_path, index=False)
    return df_results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def compute_residuals(cluster_db, y_var="tot_strength"):
    """Residualise y_var on covariates (motion outliers, sex) only, for plotting."""
    x_cov = sm.add_constant(cluster_db[["fd_outliers", "sex_bin"]])
    y = cluster_db[y_var]
    cov_model = sm.OLS(y, x_cov, missing="drop").fit()

    cluster_db_resid = cluster_db.copy()
    cluster_db_resid["strength_resid"] = cov_model.resid
    return cluster_db_resid


def plot_strength_vs_pma(cluster_db_resid, r, p_partial, out_path):
    sns.set_style("dark")
    fig, ax = plt.subplots(dpi=600)
    sns.regplot(
        data=cluster_db_resid,
        x="nscan_ga_at_scan_weeks",
        y="strength_resid",
        scatter_kws=dict(color="black", alpha=0.8),
        line_kws=dict(color="blue", lw=2),
        ax=ax,
    )
    ax.set_xlabel("PMA at scan (weeks)", fontsize=20)
    ax.set_ylabel("Cluster strength", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.text(
        0.5, 1.02,
        rf"$r={r:.3f}\;|\;P_{{\mathrm{{PART}}}}={p_partial:.3f}\;$",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=14, color="black",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if COMPUTE_CLUSTER_STRENGTH:
        compute_cluster_strength(
            CLUSTER_MASK_PATH, CEREBELLUM_MASK_PATH, SUBJECT_FC_MATRIX_DIR,
            ALL_SUBJECTS_LIST_PATH, N_CEREBELLUM, CLUSTER_PATH,
        )

    cluster_db = load_and_merge_data(DB_PATH, CLUSTER_PATH, SUBJECT_LIST_PATH, DATABASE_COLUMNS)

    lm = fit_pma_model(cluster_db)
    r, p_partial = compute_partial_correlation(cluster_db)
    save_results(lm, r, p_partial, OUT_PATH)

    cluster_db_resid = compute_residuals(cluster_db)
    plot_strength_vs_pma(cluster_db_resid, r, p_partial, PLOT_PATH)


if __name__ == "__main__":
    main()