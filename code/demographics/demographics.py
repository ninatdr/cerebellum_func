#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:53:29 2026
@author: ninatreder
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, norm

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/Users/ninatreder/Desktop/Desktop_new/dHCP/cerebellum_func_local/db_outcome_imd.csv")

subset1 = "/Users/ninatreder/Desktop/Desktop_new/dHCP/cerebellum-func/subject_list/template_subjects_42-5wks.txt"
subset2 = "/Users/ninatreder/Desktop/Desktop_new/dHCP/cerebellum-func/subject_list/subjects_37_to_42_grad"

with open(subset1, 'r') as f:
    subset1_ids = [s.strip() for s in f.readlines()]
with open(subset2, 'r') as f:
    subset2_ids = [s.strip() for s in f.readlines()]

# ---------------------------------------------------------------------------
# Build subsets
# ---------------------------------------------------------------------------
df1 = df[df['id'].isin(subset1_ids)]
df2 = df[df['id'].isin(subset2_ids)]
df1['subset'] = 1
df2['subset'] = 2

df_subsets = pd.concat([df1, df2])

x_var = 'subset'
y_vars = ['baby_ga_at_birth_weeks', 'nscan_ga_at_scan_weeks', 'fd_outliers', 'baby_gender']

groups = df_subsets[x_var].unique()
assert len(groups) == 2, f"Expected 2 groups, found {len(groups)}: {groups}"
g1, g2 = groups

# ---------------------------------------------------------------------------
# Group comparison tests (Mann-Whitney for continuous, chi-square for categorical)
# ---------------------------------------------------------------------------
results = []
for y in y_vars:
    if y != 'baby_gender':
        vals1 = df_subsets.loc[df_subsets[x_var] == g1, y].dropna()
        vals2 = df_subsets.loc[df_subsets[x_var] == g2, y].dropna()
        mw_stat, mw_p = mannwhitneyu(vals1, vals2)
        results.append({
            "X_var": x_var,
            "Y_var": y,
            "Test_stat": mw_stat,
            "P": mw_p
        })
    else:
        contingency = pd.crosstab(df_subsets[x_var], df_subsets[y])
        chi2, p, dof, expected = chi2_contingency(contingency)
        results.append({
            "X_var": x_var,
            "Y_var": y,
            "Test_stat": chi2,
            "P": p
        })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Mann-Whitney z-score helper
# ---------------------------------------------------------------------------
def mannwhitney_z(U, n1, n2):
    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu_U) / sigma_U
    return z

n1 = 20   # subset 1 size
n2 = 135  # subset 2 size
z = mannwhitney_z(863.5, n1, n2)
print(z)

# ---------------------------------------------------------------------------
# Exploratory / ad-hoc checks (as in original script)
# ---------------------------------------------------------------------------
chi2_uncorr, p_uncorr, dof, expected = chi2_contingency(contingency, correction=False)

mw = mannwhitneyu(df_subsets[x_var], df_subsets[y])

contingency = pd.crosstab(df_subsets['subset'], df_subsets['baby_gender'])
print(contingency)
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"chi2={chi2:.4f}, p={p:.4f}, dof={dof}")
print("Expected counts:")
print(expected)

# Build contingency table: rows = group, columns = sex
contingency = pd.crosstab(df['group'], df['sex'])
print(contingency)
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4f}")

mw = mannwhitneyu(df_subsets[x_var], df_subsets['baby_ga_at_birth_weeks'])
mannwhitneyu

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

fig, axes = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.histplot(data=df2, x='nscan_ga_at_scan_weeks', kde=True, ax=axes)
ax.set_xlabel("PMA at scan (weeks)", fontsize=24)
ax.set_ylabel("Count", fontsize=24)
ax.tick_params(labelsize=16)

# Force integer ticks on x-axis (PMA in weeks)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()