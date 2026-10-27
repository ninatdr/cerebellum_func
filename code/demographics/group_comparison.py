#!/usr/bin/env python3
"""
Compare demographic / acquisition variables between two subject subsets.

Continuous variables : Mann-Whitney U test (two-sided), with tie-corrected z-score.
Categorical variables: Pearson chi-square test (uncorrected, no continuity
                       correction) and Fisher's exact test (2x2 tables only).

Usage
-----
python group_comparison.py \
    --data db_outcome_imd.csv \
    --subset1 subjects_subset1.txt \
    --subset2 subjects_subset2.txt \
    --out group_comparison_results.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, rankdata

CONTINUOUS_VARS = ["baby_ga_at_birth_weeks", "nscan_ga_at_scan_weeks", "fd_outliers"]
CATEGORICAL_VARS = ["baby_gender"]


def read_ids(path):
    """Read one subject ID per line."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def mannwhitney_z(a, b):
    """z-score for the Mann-Whitney U statistic of sample `a` (normal
    approximation, tie-corrected, no continuity correction)."""
    n1, n2 = len(a), len(b)
    n = n1 + n2
    ranks = rankdata(np.concatenate([a, b]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(ranks, return_counts=True)
    tie_term = (counts**3 - counts).sum() / (n * (n - 1))
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term))
    return (u1 - n1 * n2 / 2) / sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV with an 'id' column and the variables to compare")
    parser.add_argument("--subset1", required=True, help="Text file of subject IDs for subset 1")
    parser.add_argument("--subset2", required=True, help="Text file of subject IDs for subset 2")
    parser.add_argument("--out", default="group_comparison_results.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    ids1, ids2 = read_ids(args.subset1), read_ids(args.subset2)

    overlap = ids1 & ids2
    if overlap:
        raise ValueError(f"{len(overlap)} subject(s) appear in both subsets; groups must be independent.")

    df = df[df["id"].isin(ids1 | ids2)].copy()
    df["subset"] = np.where(df["id"].isin(ids1), 1, 2)

    results = []

    for var in CONTINUOUS_VARS:
        a = df.loc[df["subset"] == 1, var].dropna().to_numpy()
        b = df.loc[df["subset"] == 2, var].dropna().to_numpy()
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        results.append({"variable": var, "test": "Mann-Whitney U", "n1": len(a), "n2": len(b),
                        "statistic": u, "z": mannwhitney_z(a, b), "p": p})

    for var in CATEGORICAL_VARS:
        table = pd.crosstab(df["subset"], df[var])
        print(f"\nContingency table: {var}\n{table}")
        n1_cat, n2_cat = int(table.loc[1].sum()), int(table.loc[2].sum())
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        results.append({"variable": var, "test": f"Chi-square, uncorrected (dof={dof})",
                        "n1": n1_cat, "n2": n2_cat, "statistic": chi2, "z": np.nan, "p": p})
        if table.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(table)
            results.append({"variable": var, "test": "Fisher's exact (statistic = odds ratio)",
                            "n1": int(table.loc[1].sum()), "n2": int(table.loc[2].sum()),
                            "statistic": odds_ratio, "z": np.nan, "p": p_fisher})
        print(f"Minimum expected cell count: {expected.min():.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print("\n", results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()