#!/usr/bin/env python3Created on Fri Mar  6 09:52:27 2026
"""
@author: ninatreder
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

# ==============================
# PATHS
# ==============================

BASE_DIR = "/Users/ninatreder/Desktop/Desktop_new/dHCP/16_new_clustering"
DB_PATH = "/Users/ninatreder/Desktop/Desktop_new/dHCP/DB_with_cerebellum_coverage.csv"
CLUSTER_PATH = f"{BASE_DIR}/cluster_res_subset1_thr0.csv"
SUBJECT_LIST_PATH = f"{BASE_DIR}/subjects_37_to_42_grad.txt"

OUT_PATH = os.path.join(f"/Users/ninatreder/Desktop/Desktop_new/dHCP/cerebellum-func/02_FC_conn/output/cluster_strength/results_pma_cluster1_thr0.csv")

# ==============================
# LOAD DATA
# ==============================

db = pd.read_csv(DB_PATH)
cluster = pd.read_csv(CLUSTER_PATH)

with open(SUBJECT_LIST_PATH, "r") as f:
    subject_ids = [s.strip() for s in f.readlines()]

cluster_filt = cluster.loc[cluster["subject_id"].isin(subject_ids)]

# ==============================
# PREPARE DATABASE
# ==============================

db_cols = [
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

db_subset = db.loc[:, db_cols].copy()
db_subset["sex_bin"] = db_subset["baby_gender"].map({2: 1, 1: 0})

cluster_db = pd.merge(
    cluster_filt,
    db_subset,
    how="left",
    left_on="subject_id",
    right_on="id",
)

# ==============================
# LINEAR MODEL
# ==============================

formula = "tot_strength ~ nscan_ga_at_scan_weeks + fd_outliers + sex_bin"
lm = smf.ols(formula, data=cluster_db).fit()

print(lm.summary())

# ==============================
# PARTIAL CORRELATION
# ==============================

vars_needed = [
    "tot_strength",
    "nscan_ga_at_scan_weeks",
    "fd_outliers",
    "sex_bin",
]

df_pcorr = (
    cluster_db[vars_needed]
    .apply(pd.to_numeric, errors="coerce")
    .dropna()
)

pcorr = pg.partial_corr(
    data=df_pcorr,
    x="tot_strength",
    y="nscan_ga_at_scan_weeks",
    covar=["fd_outliers", "sex_bin"],
    method="spearman",
)

r = pcorr["r"].values[0]
p_partial = pcorr["p-val"].values[0]

# ==============================
# SAVE RESULTS
# ==============================

df_results = pd.DataFrame(
    [{
        "Partial_r": r,
        "Partial_p": p_partial,
        "Beta_PMA": lm.params["nscan_ga_at_scan_weeks"],
        "P_PMA": lm.pvalues["nscan_ga_at_scan_weeks"],
        "R2": lm.rsquared,
    }]
)

df_results.to_csv(OUT_PATH, index=False)

# ==============================
# RESIDUALS FOR PLOTTING
# ==============================

y_var = 'tot_strength'
X_cov = cluster_db[["fd_outliers", "sex_bin"]]
X_cov = sm.add_constant(X_cov)
y = cluster_db[y_var]

cov_model = sm.OLS(y, X_cov, missing="drop").fit()

cluster_db_resid = cluster_db.copy()
cluster_db_resid["strength_resid"] = cov_model.resid

# ==============================
# PLOT
# ==============================

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
ax.tick_params(axis='both', labelsize=16)


ax.text(
    0.5,
    1.02,
    rf"$r={r:.3f}\;|\;P_{{\mathrm{{PART}}}}={p_partial:.3f}\;$",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=14,
    color="black",
)

plt.tight_layout()
plt.show()
# -*- coding: utf-8 -*-

