#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template cluster degree centrality (DC) maps
=============================================

Builds binary masks from the signal components of a cerebro-cerebellar ICA, computes
how strongly each cluster is connected to every cerebellar voxel in a template
functional connectivity (FC) matrix, and plots the resulting maps on the cerebellar
flatmap.

Pipeline (each step reads the previous step's files from disk, so any step can be
switched off or re-run on its own with the RUN_* switches below)
--------
1. Voxel counts:   number of voxels in each cerebral / cerebellar mask.
2. Cluster masks:  threshold each signal component of the ICA map (z >= THRESHOLD)
                   into a binary mask.
3. Cluster DC:     for each cluster mask, sum the template FC (cerebral cluster
                   voxels x cerebellar voxels) over the cluster voxels -> one DC value
                   per cerebellar voxel. Saved raw and z-scored (across cerebellar
                   voxels).
4. MNI transform:  warp every DC map from dHCP 40-week template space to MNI152NLin6Asym
                   with FSL applywarp.
5. Flatmap plots:  raw maps (percentile-scaled colour bar, `hot`) and z-scored maps
                   (fixed -2 to 2 colour bar, `jet`).

Usage
-----
Edit the CONFIG section below (paths and the RUN_* switches), then run:

    python template_cluster_dc.py

Requirements
------------
numpy, pandas, nibabel, matplotlib, SUITPy, and FSL (applywarp) for step 4.
"""

import glob
import os
import subprocess

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from SUITPy import flatmap

# --------------------------------------------------------------------------- #
# CONFIG - edit these paths/parameters before running
# --------------------------------------------------------------------------- #

# Optional steps (set to False to skip)
RUN_VOXEL_COUNTS = True
RUN_MAKE_MASKS = True
RUN_CLUSTER_DC = True
RUN_MNI_TRANSFORM = True
RUN_PLOTS = True

# FSL
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
os.environ["FSLDIR"] = FSLDIR
os.environ.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
APPLYWARP = os.path.join(FSLDIR, "share/fsl/bin/applywarp")

# Masks (voxel counts and DC calculation)
MASK_FOLDER = "masks"
MASK_SUBFOLDERS = ["crbl", "extracrbl"]
CRB_MASK = os.path.join(MASK_FOLDER, "extracrbl", "crb_dhcp_2mm.nii.gz")   # cerebral mask
CRBL_MASK = os.path.join(MASK_FOLDER, "crbl", "crbl_dhcp_2mm.nii.gz")      # cerebellar mask

# Cluster masks from the ICA
CLUSTER_FOLDER = "d20"
CLUSTER_ICA_FILE = os.path.join(CLUSTER_FOLDER, "crb_crbl_ICA20_allconn.nii.gz")  # 4D ICA spatial maps
SIGNAL_COMPS_FILE = os.path.join(CLUSTER_FOLDER, "signal_comps")                  # CSV with columns: n, components
CLUSTER_MASK_DIR = os.path.join(CLUSTER_FOLDER, "masks")
THRESHOLD = 3

# Template DC
TEMPLATE_MATRIX_PATH = "data/template_FC_matrix.npy"  # template FC matrix (cerebral voxels x cerebellar voxels block)
DC_THRESHOLD = 0                                      # FC values below this are set to 0
OUTPUT_SUFFIX = "template"
DC_DIR = os.path.join(CLUSTER_FOLDER, "template_cluster_DC")

# MNI transform
MNI_REFERENCE = "templates/MNI_transform/MNI152_T1_2mm_brain.nii.gz"
MNI_WARP = "templates/warp/extdhcp40wk_to_MNI152NLin6ASym_warp.nii.gz"
MNI_DIR = os.path.join(CLUSTER_FOLDER, "template_cluster_DC_MNI")

# Plots
PLOT_DIR = os.path.join(CLUSTER_FOLDER, "template_cluster_DC_plots")
PCTILE_LOW, PCTILE_HIGH = 2, 98  # percentile range of the raw-score colour bar


# --------------------------------------------------------------------------- #
# Step 1: voxel counts per mask
# --------------------------------------------------------------------------- #

def count_mask_voxels(mask_folder, subfolders):
    """Print and return the number of voxels equal to 1 in every mask file."""
    voxel_counts = {}

    for sub in subfolders:
        sub_path = os.path.join(mask_folder, sub)
        for fname in os.listdir(sub_path):
            fpath = os.path.join(sub_path, fname)
            matr = nib.load(fpath).get_fdata()
            count = np.sum(matr == 1)
            voxel_counts[f"{sub}/{fname}"] = count
            print(f"{sub}/{fname}: {count}")

    return voxel_counts


# --------------------------------------------------------------------------- #
# Step 2: binary masks from ICA clusters
# --------------------------------------------------------------------------- #

def make_cluster_masks(cluster_ica_file, signal_comps_file, out_dir, threshold):
    """Threshold each signal component of the ICA map into a binary mask and save it."""
    os.makedirs(out_dir, exist_ok=True)

    signal_comps = pd.read_csv(signal_comps_file, header=0, skipinitialspace=True)

    is_signal = signal_comps["components"] != "noise"
    signal_idx = signal_comps.loc[is_signal, "n"].values
    signal_labels = signal_comps.loc[is_signal, "components"].values

    clusters_img = nib.load(cluster_ica_file)
    clusters_data = clusters_img.get_fdata()

    affine = clusters_img.affine

    assert clusters_data.shape[3] == len(signal_comps), (
        f"clusters data has {clusters_data.shape[3]} components, but"
        f" signal_comps has {len(signal_comps)} rows: labels won't line up")

    for idx, label in zip(signal_idx, signal_labels):
        comp_vol = clusters_data[..., idx]
        binary_mask = (comp_vol >= threshold).astype(np.uint8)

        n_voxels = int(binary_mask.sum())
        print(f"component {idx} ({label}): {n_voxels} voxels >= {threshold}")

        mask_img = nib.Nifti1Image(binary_mask, affine=affine)
        out_path = os.path.join(out_dir, f"ic{idx:02d}_{label}_thr3_binmask.nii.gz")
        nib.save(mask_img, out_path)

    print(f"Saved {len(signal_idx)} masks to {out_dir}")


# --------------------------------------------------------------------------- #
# Step 3: DC per template cluster
# --------------------------------------------------------------------------- #

def compute_cluster_dc(template_matrix_path, crb_mask_path, crbl_mask_path,
                       cluster_mask_dir, output_folder, thr, output_suf):
    """Compute raw and z-scored DC maps (over cerebellar voxels) for each cluster mask."""
    crb_mask = nib.load(crb_mask_path)
    crbl_mask = nib.load(crbl_mask_path)

    crb_mask_data = crb_mask.get_fdata()
    crbl_mask_data = crbl_mask.get_fdata()

    crb_size = crb_mask_data[crb_mask_data > 0].size
    crbl_size = crbl_mask_data[crbl_mask_data > 0].size

    os.makedirs(output_folder, exist_ok=True)

    matrix = np.load(template_matrix_path)
    mc = matrix.copy()
    mc[mc < thr] = 0

    crb_crbl_matrix = mc[:crb_size, -crbl_size:]

    cluster_mask_files = sorted(glob.glob(os.path.join(cluster_mask_dir, "*.nii.gz")))
    if not cluster_mask_files:
        print(f"No cluster masks found in {cluster_mask_dir}")
        return

    for cluster_mask_path in cluster_mask_files:
        cluster_name = os.path.basename(cluster_mask_path).replace(".nii.gz", "")

        cluster_mask = nib.load(cluster_mask_path)
        mask_data = cluster_mask.get_fdata()

        cluster_vals = crb_crbl_matrix[mask_data[crb_mask_data > 0].astype("bool")]
        cluster_dc = np.sum(cluster_vals, axis=0)

        img = np.zeros_like(crbl_mask_data)
        img[crbl_mask_data > 0] = cluster_dc

        new_img = nib.Nifti1Image(img, affine=crbl_mask.affine)

        out_path = os.path.join(output_folder, f"{cluster_name}_{output_suf}.nii.gz")
        nib.save(new_img, out_path)
        print(f"Saved: {out_path}")

        # z-scored DC (across cerebellar voxels within this cluster's map)
        dc_mean = cluster_dc.mean()
        dc_std = cluster_dc.std()
        if dc_std == 0:
            print(f"  WARNING: zero variance for {cluster_name}, skipping z-score")
        else:
            cluster_dc_z = (cluster_dc - dc_mean) / dc_std

            img_z = np.zeros_like(crbl_mask_data)
            img_z[crbl_mask_data > 0] = cluster_dc_z

            new_img_z = nib.Nifti1Image(img_z, affine=crbl_mask.affine)

            out_path_z = os.path.join(output_folder, f"{cluster_name}_{output_suf}_zscored.nii.gz")
            nib.save(new_img_z, out_path_z)
            print(f"Saved: {out_path_z}")


# --------------------------------------------------------------------------- #
# Step 4: transform DC maps to MNI space
# --------------------------------------------------------------------------- #

def dhcp_to_mni(infile, out_dir, applywarp, reference, warp):
    """Warp one map from dHCP 40-week space to MNI152NLin6Asym with FSL applywarp."""
    outfile = os.path.join(
        out_dir,
        os.path.basename(infile).replace(".nii.gz", "_MNILin6aAsym.nii.gz")
    )
    result = subprocess.run([
        applywarp,
        "-i", str(infile),
        "-r", reference,
        "-o", str(outfile),
        "-w", warp,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"applywarp failed for {infile} (exit code {result.returncode})\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if not os.path.exists(outfile):
        raise RuntimeError(
            f"applywarp reported success but {outfile} was not created.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return outfile


def transform_all_to_mni(dc_dir, mni_dir):
    """Transform all raw and z-scored DC maps to MNI space (skipping existing outputs)."""
    os.makedirs(mni_dir, exist_ok=True)

    output_files = sorted(glob.glob(os.path.join(dc_dir, "*.nii.gz")))
    raw_files = [f for f in output_files if not f.endswith("_zscored.nii.gz")]
    zscored_files = [f for f in output_files if f.endswith("_zscored.nii.gz")]

    for src_path in raw_files + zscored_files:
        mni_path = os.path.join(
            mni_dir,
            os.path.basename(src_path).replace(".nii.gz", "_MNILin6aAsym.nii.gz")
        )
        if os.path.exists(mni_path):
            print(f"Already transformed, skipping: {mni_path}")
            continue
        dhcp_to_mni(src_path, mni_dir, APPLYWARP, MNI_REFERENCE, MNI_WARP)
        print(f"Transformed: {mni_path}")


# --------------------------------------------------------------------------- #
# Step 5: flatmap plots
# --------------------------------------------------------------------------- #

def plot_dc_maps(mni_dir, plot_dir, output_suf, pctile_low, pctile_high):
    """Plot each cluster's DC map on the flatmap (raw: percentile scale; z-scored: -2 to 2)."""
    os.makedirs(plot_dir, exist_ok=True)

    mni_files = sorted(glob.glob(os.path.join(mni_dir, "*.nii.gz")))
    mni_raw_files = [f for f in mni_files if "_zscored" not in f]
    mni_zscored_files = [f for f in mni_files if "_zscored" in f]

    # --- raw DC maps: percentile-scaled colourbar, hot cmap ---
    for raw_path in mni_raw_files:
        cluster_name = (
            os.path.basename(raw_path)
            .replace(f"_{output_suf}_MNILin6aAsym.nii.gz", "")
        )

        img = nib.load(raw_path)
        surface = flatmap.vol_to_surf(img, space="FSL", ignore_zeros=True)
        nonzero_mask = (surface != 0) & np.isfinite(surface)

        if not nonzero_mask.any():
            print(f"  WARNING: {cluster_name} has no nonzero surface values, skipping plot")
            continue

        vmin = np.percentile(surface[nonzero_mask], pctile_low)
        vmax = np.percentile(surface[nonzero_mask], pctile_high)

        # round outward to nearest 5
        vmin = 5 * np.floor(vmin / 5)
        vmax = 5 * np.ceil(vmax / 5)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            print(f"  WARNING: {cluster_name} has degenerate vmin/vmax ({vmin}, {vmax}), skipping plot")
            continue

        ax = flatmap.plot(data=surface, cmap="hot",
                          new_figure=True,
                          colorbar=True,
                          cscale=[vmin, vmax])

        cbar_ax = ax.get_figure().axes[-1]
        cbar_ax.set_ylabel("raw DC")
        cbar_ax.tick_params(labelsize=18)
        cbar_ax.set_yticks([vmin, vmax])
        cbar_ax.yaxis.label.set_size(24)
        cbar_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, z_: f"{float(x):.2f}"))

        fig = ax.get_figure()
        out_png = os.path.join(plot_dir, f"{cluster_name}_raw.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")

    # --- z-scored DC maps: fixed -2 to 2 colourbar, jet cmap ---
    for z_path in mni_zscored_files:
        cluster_name = (
            os.path.basename(z_path)
            .replace(f"_{output_suf}_zscored_MNILin6aAsym.nii.gz", "")
        )

        img = nib.load(z_path)
        surface_z = flatmap.vol_to_surf(img, space="FSL", ignore_zeros=True)

        ax_z = flatmap.plot(data=surface_z, cmap="jet",
                            new_figure=True,
                            colorbar=True,
                            cscale=[-2, 2])

        cbar_ax_z = ax_z.get_figure().axes[-1]
        cbar_ax_z.set_ylabel("z-score")
        cbar_ax_z.tick_params(labelsize=18)
        cbar_ax_z.set_yticks([-2, 2])
        cbar_ax_z.yaxis.label.set_size(24)
        cbar_ax_z.yaxis.set_major_formatter(FuncFormatter(lambda x, z_: f"{float(x)}"))

        fig_z = ax_z.get_figure()
        out_png_z = os.path.join(plot_dir, f"{cluster_name}_zscored.png")
        fig_z.savefig(out_png_z, dpi=300, bbox_inches="tight")
        plt.close(fig_z)
        print(f"Saved: {out_png_z}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if RUN_VOXEL_COUNTS:
        count_mask_voxels(MASK_FOLDER, MASK_SUBFOLDERS)

    if RUN_MAKE_MASKS:
        make_cluster_masks(CLUSTER_ICA_FILE, SIGNAL_COMPS_FILE, CLUSTER_MASK_DIR, THRESHOLD)

    if RUN_CLUSTER_DC:
        compute_cluster_dc(
            TEMPLATE_MATRIX_PATH, CRB_MASK, CRBL_MASK,
            CLUSTER_MASK_DIR, DC_DIR, DC_THRESHOLD, OUTPUT_SUFFIX,
        )

    if RUN_MNI_TRANSFORM:
        transform_all_to_mni(DC_DIR, MNI_DIR)

    if RUN_PLOTS:
        plot_dc_maps(MNI_DIR, PLOT_DIR, OUTPUT_SUFFIX, PCTILE_LOW, PCTILE_HIGH)


if __name__ == "__main__":
    main()