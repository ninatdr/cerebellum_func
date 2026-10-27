# Cerebro-cerebellar cluster analysis

Code for (1) mapping how strongly ICA-derived cerebral clusters are connected to the
cerebellum in a template functional connectivity (FC) matrix, and (2) testing how
per-subject cluster connectivity strength relates to age, using the developing Human
Connectome Project (dHCP) dataset.

| Script | Purpose |
|---|---|
| `template_cluster_dc.py` | Builds binary cluster masks from an ICA, computes template degree-centrality (DC) maps over the cerebellum, transforms them to MNI space and plots them on the cerebellar flatmap |
| `cluster_strength_pma_analysis.py` | Models per-subject cluster strength against age (PMA at scan; postnatal weeks and GA at birth), FDR-corrects across clusters and plots significant effects |

## Requirements

Python ≥ 3.9 and:

```bash
pip install numpy pandas matplotlib seaborn statsmodels "pingouin==0.5.5" nibabel SUITPy
```

`pingouin` is pinned because newer releases (tested: 0.7.0) renamed the `p-val` column
returned by `partial_corr`, which `cluster_strength_pma_analysis.py` expects. FSL
(`applywarp`) is required for the MNI transform step of `template_cluster_dc.py`.

## Data

The dHCP data are available from the [dHCP data release](https://www.developingconnectome.org/data-release/)
under its data-use agreement and are not included here. The clinical database
(`DB_with_cerebellum_coverage.csv`) is derived from dHCP metadata and is likewise not shared.

---

## 1. Template cluster DC maps (`template_cluster_dc.py`)

Edit the **CONFIG** section at the top of the script, then run:

```bash
python template_cluster_dc.py
```

### Steps

Each step reads the previous step's output from disk, so steps can be switched off or
re-run individually with the `RUN_*` settings.

| Switch | Step |
|---|---|
| `RUN_VOXEL_COUNTS` | Prints the number of voxels in each cerebral / cerebellar mask |
| `RUN_MAKE_MASKS` | Thresholds each signal component of the ICA map (z ≥ `THRESHOLD`, default 3) into a binary cluster mask |
| `RUN_CLUSTER_DC` | For each cluster, sums the template FC (cluster voxels × cerebellar voxels) over the cluster voxels, giving one DC value per cerebellar voxel; saves raw and z-scored maps (z-scored across cerebellar voxels) |
| `RUN_MNI_TRANSFORM` | Warps all DC maps from dHCP 40-week template space to MNI152NLin6Asym with FSL `applywarp` (existing outputs are skipped) |
| `RUN_PLOTS` | Plots each map on the cerebellar flatmap |

### Inputs

| Setting | Description |
|---|---|
| `MASK_FOLDER` | Contains `crbl/` and `extracrbl/` mask folders; `CRB_MASK` (cerebral) and `CRBL_MASK` (cerebellar) are the 2 mm masks that define the rows and columns of the FC block |
| `CLUSTER_ICA_FILE` | 4D NIfTI of ICA spatial maps (x, y, z, components) |
| `SIGNAL_COMPS_FILE` | CSV with columns `n` (0-indexed component number) and `components` (label; `noise` components are skipped). Must have one row per ICA component |
| `TEMPLATE_MATRIX_PATH` | `.npy` template FC matrix. The cerebral × cerebellar block is taken as `matrix[:n_cerebral_voxels, -n_cerebellar_voxels:]` |
| `DC_THRESHOLD` | FC values below this are set to 0 before summing (default 0) |
| `MNI_REFERENCE`, `MNI_WARP` | MNI152 2 mm reference image and the dHCP 40-week → MNI152NLin6Asym warp field |
| `FSLDIR` | FSL installation (read from the `FSLDIR` environment variable if set) |

### Outputs

Written under `CLUSTER_FOLDER` (default `d20/`):

| Folder | Content |
|---|---|
| `masks/` | Binary cluster masks, `ic<NN>_<label>_thr3_binmask.nii.gz` |
| `template_cluster_DC/` | Raw and `_zscored` DC maps in template space |
| `template_cluster_DC_MNI/` | The same maps in MNI space (`_MNILin6aAsym`) |
| `template_cluster_DC_plots/` | Flatmap PNGs: `*_raw.png` (`hot`, colour bar from the 2nd–98th percentile of non-zero values, rounded outward to the nearest 5) and `*_zscored.png` (`jet`, fixed −2 to 2) |

---

## 2. Cluster strength vs. age (`cluster_strength_pma_analysis.py`)

Edit the **CONFIG** section, then run:

```bash
python cluster_strength_pma_analysis.py
```

### Inputs

| Setting | Description |
|---|---|
| `DATABASE_CSV` | Clinical database. Columns used: `id`, `nscan_ga_at_scan_weeks` (PMA at scan), `baby_ga_at_birth_weeks` (GA at birth), `baby_gender` (coded 1 → 0, 2 → 1 as `sex_bin`), `fd_outliers` (motion outliers) |
| `SUBJECT_LIST` | Text file, one subject ID per line; only these subjects are analysed |
| `SELECTION_CSV` | Component selection table with columns `component` (0-indexed), `label` and `selected` (boolean). Only selected components are analysed |
| `CLUSTER_STRENGTH_CSV` | Per-subject cluster strengths with columns `subject_id` and `cluster_01_mean … cluster_20_mean` (component *k* ↔ `cluster_{k+1:02d}_mean`) |

Postnatal weeks (`pnw`) = PMA at scan − GA at birth.

### Statistical models

Fitted separately for every retained cluster:

1. **PMA model:** `strength ~ PMA at scan + motion outliers + sex`
2. **PNW/GA model:** `strength ~ postnatal weeks + GA at birth + motion outliers + sex`

The effect of interest is tested with a partial Spearman correlation
(`pingouin.partial_corr`) controlling for the other predictors (PNW and GA each control
for the other). P-values are FDR-corrected (Benjamini–Hochberg, α = 0.05) across
clusters, separately for each effect (PMA, PNW, GA). OLS β, p-values and R² are reported
alongside as effect sizes.

### Outputs (in `OUTPUT_DIR`)

| File | Content |
|---|---|
| `results_pma_retained_clusters_FDR.csv` | Partial r / p, FDR-corrected p and significance flag, OLS β / p / R² for the PMA model |
| `results_pnw_ga_retained_clusters_FDR.csv` | The same for PNW and GA effects |
| `plot_<cluster>_pma.png` | Strength (residualised for motion outliers and sex) vs. PMA at scan, FDR-significant clusters only |
| `plot_<cluster>_pnw.png` | Strength (residualised for GA at birth, motion outliers, sex) vs. postnatal age |
| `plot_<cluster>_ga.png` | Strength (residualised for PNW, motion outliers, sex) vs. GA at birth |

Plots are only produced for clusters that survive FDR correction for that effect.
Set `RUN_PMA_PLOTS` / `RUN_PNW_GA_PLOTS` to `False` to skip plotting, and `SHOW_PLOTS` to
`False` to save figures without displaying them.

## Citation

[Add paper reference here]

## License

[Add license here]