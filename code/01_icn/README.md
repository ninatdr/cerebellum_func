# Core Network Strength (CNS) analysis

Code for quantifying the strength of resting-state networks in the neonatal cerebellum
and testing how it relates to age at scan, using the developing Human Connectome Project
(dHCP) dataset.

Networks are derived with group ICA (FSL MELODIC) and dual regression. For each
independent component (IC), CNS is the mean subject-level dual-regression value within
the component's "core" (voxels with z > 3 in the group map).

## Pipeline

```
group MELODIC maps ─┐
                    ├─► core_network_strength.py ─► core_network_strengths.csv ─┐
dual-regression     │                                                          ├─► icn_analysis.py ─► statistics + figures
stage-2 outputs ────┘                                    clinical database ────┘
```

| Script | Purpose |
|---|---|
| `core_network_strength.py` | Computes CNS per subject and component; writes `core_network_strengths.csv` |
| `ICN_analysis.py` | Models CNS against age at scan, FDR-corrects across components, and produces the figures |

## Requirements

Python ≥ 3.9 and:

```bash
pip install numpy pandas matplotlib seaborn statsmodels "pingouin==0.5.5" nibabel SUITPy
```

`pingouin` is pinned because newer releases (tested: 0.7.0) renamed the `p-val` column
returned by `partial_corr`, which the analysis script expects. The scripts were tested
with pingouin 0.5.5. FSL is needed upstream to run MELODIC and dual regression.

## Data

The dHCP data are available from the [dHCP data release](https://www.developingconnectome.org/data-release/)
under its data-use agreement and are not included in this repository. The clinical
database used here (`db_outcome_imd.csv`) is derived from dHCP metadata and likewise
not shared.

## 1. Compute core network strength

```bash
python core_network_strength.py <melodic_file> <dual_regression_folder> <subject_list>
```

| Argument | Description |
|---|---|
| `melodic_file` | 4D NIfTI of group ICA spatial maps (x, y, z, components), in the same space as the dual-regression outputs |
| `dual_regression_folder` | FSL `dual_regression` output folder containing `dr_stage2_subjectXXXXX.nii.gz` |
| `subject_list` | Text file, one subject ID per line, in the same order as the subjects passed to `dual_regression` |

Notes:

- Each component's core mask is `melodic > 3` (`THRESHOLD` at the top of the script).
- Subjects are matched to stage-2 files by position (sorted file names vs. subject list
  order). The script stops with an error if the counts differ.
- Output: `core_network_strengths.csv` in the dual-regression folder, with columns
  `subject_id, Component_1, …, Component_N` (components numbered from 1, in the order
  of the MELODIC map).

## 2. Statistics and figures

Edit the **CONFIG** section at the top of `core_network_strength_analysis.py`, then run:

```bash
python core_network_strength_analysis.py
```

### Configuration

| Setting | Description |
|---|---|
| `RUN_PMA_ANALYSIS` | PMA-at-scan statistics and scatterplots |
| `RUN_NETWORK_MAPS` | Network flatmaps and winner-take-all atlas |
| `RUN_COMPONENT_MAPS` | Single-component map and by-week maps |
| `CNS_CSV`, `DATABASE_CSV` | Input tables |
| `COMPONENTS_TO_DROP` | Component numbers (1-indexed) excluded from the statistics (noise/artefact ICs) |
| `NETWORK_COMPONENTS_TO_EXCLUDE` | The same exclusions for the network flatmaps, as 0-indexed columns of the IC map |
| `NETWORK_LABELS` | Anatomical labels for the plotted networks |
| `OUTPUT_DIR` | Where results are written |

The map steps need the surface-mapped IC map and the dual-regression group maps
(paths in the CONFIG section); the statistics steps only need the two CSV files.

### Inputs

`CNS_CSV` is the output of step 1. `DATABASE_CSV` must contain the columns listed in
`DATABASE_COLUMNS`; those used in the models are:

| Column | Use |
|---|---|
| `id` | Subject ID (matched to `subject_id` in the CNS table) |
| `nscan_ga_at_scan_weeks` | Postmenstrual age (PMA) at scan |
| `baby_ga_at_birth_weeks` | Gestational age (GA) at birth |
| `baby_gender` | Sex (coded 1 → 0, 2 → 1 as `sex_bin`) |
| `fd_outliers` | Motion outliers (covariate) |

Postnatal weeks (`pnw`) = PMA at scan − GA at birth.

### Statistical models

Fitted separately for every retained component:

1. **PMA model:** `CNS ~ PMA at scan + sex + motion outliers`
2. **PNW/GA model:** `CNS ~ postnatal weeks + GA at birth + sex + motion outliers`

For both, the effect of interest is tested with a partial Spearman correlation
(`pingouin.partial_corr`) controlling for the other predictors. P-values are
FDR-corrected (Benjamini–Hochberg, α = 0.05) across components, separately for each
tested effect (PMA; PNW; GA). OLS estimates (β, SE, t, 95% CI, R²) are reported
alongside for effect sizes.

### Outputs (in `OUTPUT_DIR`)

| File | Content |
|---|---|
| `ols_results_pma.txt` / `.csv` | PMA model: OLS summaries and partial correlations with raw and FDR-corrected p-values |
| `ols_results_pnw.txt` / `.csv` | PNW/GA model: partial correlations for PNW and GA, FDR-corrected p-values, R² |
| `fig2c_cns_vs_pma_significant.png` | CNS (residualised for sex and motion outliers) vs. PMA for FDR-significant components |
| `fig2c_cns_vs_pnw_significant.png` | CNS (residualised for GA at birth, sex, motion outliers) vs. postnatal weeks, FDR-significant components |
| `fig1a_networks.png`, `fig1a_atlas.png` | Cerebellar flatmaps of each network; winner-take-all atlas |
| `fig2b_component_map.png`, `fig2b_weekly_maps.png` | Single-component map; the same map by postmenstrual week |

Scatterplots are only produced if at least one component survives FDR correction.

## Citation

[Add paper reference here]

## License

[Add license here]