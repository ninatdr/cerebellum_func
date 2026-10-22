#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:36:36 2026

@author: ninatreder
"""

import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy import ndimage
import pandas as pd
import os
# ---------------------------------------------------------------------------
# Config -- match these to your original ICA run
# ---------------------------------------------------------------------------
FC_MATRIX_PATH = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/02_FC_conn/template_FC_matrix.npy"
CRB_MASK_PATH = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/masks/extracrbl/crb_dhcp_2mm.nii.gz"   # cerebrum mask (matches ICA output space)

OUT_PATH = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/02_FC_conn/d20/cluster_select_perm/n1000"

os.makedirs(OUT_PATH, exist_ok=True)

N_CRB = 18360      # cerebrum
N_BG_THAL = 3380    # basal ganglia / thalamus -- match your actual counts
N_PONS = 217
N_CRBL = 2881      # cerebellum

N_COMPONENTS = 20
THRESHOLD = 3

N_PERMUTATIONS = 1000  # start here; increase if you have compute budget (100+ is more standard)
NULL_PERCENTILE = 95  # extent threshold = this percentile of the null max-cluster-size distribution

RNG_SEED = 0

# ---------------------------------------------------------------------------
# Rebuild the same cerebrum-to-cerebellum matrix used for the real ICA fit
# ---------------------------------------------------------------------------
corr_matr = np.load(FC_MATRIX_PATH)
corr_matr[corr_matr < 0] = 0

region_total = N_CRB + N_BG_THAL + N_PONS + N_CRBL
assert region_total == corr_matr.shape[1], (
    f"Region counts sum to {region_total}, but matrix has {corr_matr.shape[1]} columns."
)

crb_matrix = corr_matr[0:N_CRB, :]
scaler = StandardScaler()
crb_matrix_z = scaler.fit_transform(crb_matrix)[:, -N_CRBL:].copy()
print(f"Cerebrum x cerebellum matrix shape: {crb_matrix_z.shape}")

crb_mask = nib.load(CRB_MASK_PATH)
crb_mask_data = crb_mask.get_fdata() > 0

n_crb_mask_voxels = int(crb_mask_data.sum())
assert n_crb_mask_voxels == N_CRB, (
    f"Cerebrum mask has {n_crb_mask_voxels} voxels, expected N_CRB={N_CRB}."
)

struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity, matches the compactness script

# ---------------------------------------------------------------------------
# Build the null distribution: permute rows (break real cerebral-voxel
# spatial structure while preserving each row's own connectivity profile),
# rerun ICA, project back to voxel space, threshold, find largest cluster
# per component, per permutation.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
null_max_cluster_sizes = []

for perm_i in range(N_PERMUTATIONS):
    # shuffle which cerebral voxel gets which connectivity profile ->
    # destroys real spatial structure, keeps the value distribution intact
    perm_idx = rng.permutation(crb_matrix_z.shape[0])
    permuted_matrix = crb_matrix_z[perm_idx, :]

    ica_null = FastICA(n_components=N_COMPONENTS, random_state=perm_i)
    null_components = ica_null.fit_transform(permuted_matrix)

    perm_max = 0
    for c in range(N_COMPONENTS):
        vol = np.zeros(crb_mask_data.shape)
        vol[crb_mask_data] = np.abs(null_components[:, c])

        thresholded = (vol >= THRESHOLD) & crb_mask_data
        labeled, n_clusters = ndimage.label(thresholded, structure=struct)

        if n_clusters > 0:
            sizes = ndimage.sum(thresholded, labeled, index=range(1, n_clusters + 1))
            perm_max = max(perm_max, int(sizes.max()))

    null_max_cluster_sizes.append(perm_max)
    print(f"Permutation {perm_i + 1}/{N_PERMUTATIONS}: max cluster size = {perm_max}")

null_max_cluster_sizes = np.array(null_max_cluster_sizes)
extent_threshold = np.percentile(null_max_cluster_sizes, NULL_PERCENTILE)

print(f"\nNull max-cluster-size distribution (n={N_PERMUTATIONS}):")
print(f"  mean: {null_max_cluster_sizes.mean():.1f}")
print(f"  {NULL_PERCENTILE}th percentile (extent threshold): {extent_threshold:.1f} voxels")

pd.DataFrame({"permutation": range(N_PERMUTATIONS), "max_cluster_size": null_max_cluster_sizes}) \
  .to_csv(os.path.join(OUT_PATH, "ica_null_max_cluster_sizes.csv"), index=False)
print("Saved: ica_null_max_cluster_sizes.csv")

with open(os.path.join(OUT_PATH, "ica_extent_threshold.txt"), "w") as f:
    f.write(f"Extent threshold ({NULL_PERCENTILE}th percentile of null, n={N_PERMUTATIONS} permutations): "
            f"{extent_threshold:.1f} voxels\n")
print("Saved: ica_extent_threshold.txt")

# ---------------------------------------------------------------------------
# Apply the threshold to your REAL (already-fit) ICA output to decide which
# components to keep
# ---------------------------------------------------------------------------
REAL_ICA_PATH = "/Users/ninatreder/Desktop/dHCP/cerebellum_func_local/02_FC_conn/crb_crbl_ICA20_allconn.nii.gz"  # your actual saved 4D ICA output

real_ica_img = nib.load(REAL_ICA_PATH)
real_ica_data = real_ica_img.get_fdata()

assert real_ica_data.shape[:3] == crb_mask_data.shape, (
    f"Real ICA volume shape {real_ica_data.shape[:3]} doesn't match mask shape {crb_mask_data.shape}"
)
assert real_ica_data.shape[3] == N_COMPONENTS, (
    f"Real ICA output has {real_ica_data.shape[3]} components, expected N_COMPONENTS={N_COMPONENTS}"
)

selection_rows = []
for i in range(N_COMPONENTS):
    comp_vol = real_ica_data[..., i]
    thresholded = (comp_vol >= THRESHOLD) & crb_mask_data
    labeled, n_clusters = ndimage.label(thresholded, structure=struct)

    if n_clusters == 0:
        largest = 0
    else:
        sizes = ndimage.sum(thresholded, labeled, index=range(1, n_clusters + 1))
        largest = int(sizes.max())

    selected = largest >= extent_threshold
    selection_rows.append({
        "component": i,
        "n_clusters": n_clusters,
        "largest_cluster_voxels": largest,
        "extent_threshold": extent_threshold,
        "selected": selected,
    })

selection_df = pd.DataFrame(selection_rows)
print("\nComponent selection based on permutation-derived extent threshold:")
print(selection_df.to_string(index=False))

selection_df.to_csv(os.path.join(OUT_PATH, "ica_component_selection.csv"), index=False)
print("\nSaved: ica_component_selection.csv")
print(f"\n{selection_df['selected'].sum()} / {N_COMPONENTS} components retained "
      f"(largest cluster >= {extent_threshold:.1f} voxels)")
