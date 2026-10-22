#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Network Strength (CNS) computation
=========================================

For each independent component (IC) in a group MELODIC map, thresholds the
component to define a binary "core" mask, then for each subject computes the
mean dual-regression stage-2 value within that mask. Saves one CNS value per
subject per component to core_network_strengths.csv in the dual-regression
output folder.

Usage
-----
    python core_network_strength.py <melodic_file> <dual_regression_folder> <subject_list>
"""

import os
import sys

import nibabel as nib
import pandas as pd

THRESHOLD = 3  # z-score threshold used to define each component's core mask


def main():
    if len(sys.argv) < 4:
        print("Usage: python core_network_strength.py <melodic_file> <dual_regression_folder> <subject_list>")
        sys.exit(1)

    melodic_path = sys.argv[1]
    dr_folder_path = sys.argv[2]
    subject_ids_file = sys.argv[3]

    # --- Load data -----------------------------------------------------------
    melodic = nib.load(melodic_path)
    melodic_data = melodic.get_fdata()

    # Binary mask per IC: voxels exceeding THRESHOLD define that component's core
    component_masks = (melodic_data > THRESHOLD).astype(int)

    with open(subject_ids_file, "r") as f:
        subject_ids = [line.strip() for line in f.readlines()]

    dr_files = sorted([
        file for file in os.listdir(dr_folder_path)
        if file.startswith("dr_stage2_subject") and not file.endswith("Z.nii.gz")
    ])

    # --- Core network strength calculation ------------------------------------
    core_network_strengths = {}

    for subject_id, file in zip(subject_ids, dr_files):
        st2_file = os.path.join(dr_folder_path, file)

        subject_img = nib.load(st2_file)
        subject_data = subject_img.get_fdata()

        if subject_data.shape[-1] != component_masks.shape[-1]:
            print(f"WARNING: mismatch in components for {subject_id}")
            continue

        print(f"Processing subject: {subject_id}")

        core_network_strengths[subject_id] = []
        for j in range(component_masks.shape[-1]):
            mask = component_masks[..., j]
            component_values = subject_data[..., j] * mask
            core_strength = component_values[mask > 0].mean()
            core_network_strengths[subject_id].append(core_strength)

    # --- Save results ----------------------------------------------------------
    print("\nCreating DataFrame...")
    core_network_strengths_df = pd.DataFrame.from_dict(core_network_strengths, orient="index")
    core_network_strengths_df.columns = [f"Component_{i + 1}" for i in range(core_network_strengths_df.shape[1])]
    core_network_strengths_df.index.name = "subject_id"

    output_file = os.path.join(dr_folder_path, "core_network_strengths.csv")
    print(f"Saving to CSV at:\n{output_file}")
    core_network_strengths_df.to_csv(output_file)


if __name__ == "__main__":
    main()