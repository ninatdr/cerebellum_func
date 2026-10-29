# Cerebellar Functional Gradient Analysis

Code for computing functional connectivity gradients of the neonatal cerebellum from resting-state fMRI data, generating group- and subject-level gradient maps, and assessing developmental associations with age-related measures in the developing Human Connectome Project (dHCP) dataset.

| Script | Purpose |
|---|---|
| `run_gradients.py` | Computes group-level cerebro-cerebellar and intra-cerebellar gradients, aligns subject-level gradients to the group template, and calculates subject-template similarity measures |
| `gradient_utils.py` | Utility functions for gradient decomposition, alignment, volumetric/surface projection, and similarity analyses |
| `gradients_analyses_plotting.py` | Statistical analyses and visualisation of gradient organisation, network distributions, and developmental effects |

## Requirements

Python ≥ 3.9 and:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels \
pingouin nibabel brainspace SUITPy
```

Additional requirements:

- FSL (`applywarp`) for template-to-MNI transformations
- SUIT flatmap resources

## Data

The dHCP data are available from the [dHCP data release](https://www.developingconnectome.org/data-release/) under its data-use agreement and are not included here. Functional connectivity matrices, demographic data, and derived subject lists must be generated separately.

---

## Running the gradient pipeline

Edit the paths and configuration settings in `run_gradients.py`, then run:

```bash
python run_gradients.py
```

### Analysis steps

| Step | Description |
|---|---|
| Group gradients | Computes diffusion map embeddings from the template connectivity matrix |
| Subject gradients | Computes gradients for each subject and aligns them to the group solution using Procrustes alignment |
| Similarity analysis | Quantifies correspondence between subject and template gradients |
| Surface projection | Projects gradients to cerebellar flatmaps for visualisation |

### Inputs

| Setting | Description |
|---|---|
| `MATRIX_FILE` | Template functional connectivity matrix |
| `MASK_FOLDER` | Cerebral and cerebellar masks defining matrix dimensions |
| `SUBJECT_LIST` | Text file containing subject IDs |
| `DATABASE_CSV` | Clinical/demographic database |
| `MNI_WARP` | dHCP-to-MNI transformation field |
| `MNI_REFERENCE` | MNI reference image |

### Outputs

| Folder/File | Content |
|---|---|
| `group_gradients/` | Group-level gradient maps and variance explained |
| `subject_gradients/` | Subject-level aligned gradients |
| `flatmaps/` | Cerebellar surface projections |
| `correlations/` | Subject-template similarity metrics |

---

## Gradient analyses and plotting

Run:

```bash
python gradients_analyses_plotting.py
```

### Analyses

- Variance explained by each gradient
- Gradient scatterplots and pairwise relationships
- Comparisons between cerebro-cerebellar and intra-cerebellar gradients
- Distribution of gradients across Buckner functional networks
- Developmental effects of PMA at scan

### Statistical models

Associations between gradient similarity and developmental measures are assessed using:

- Spearman partial correlations
- Linear regression models
- Benjamini-Hochberg FDR correction

### Outputs

| File | Content |
|---|---|
| `*_variance_explained.*` | Variance explained plots |
| `*_scatter.*` | Gradient relationship plots |
| `*_network_distributions.*` | Functional network analyses |
| `*_developmental_results.csv` | Age-related statistical results |

## Citation

