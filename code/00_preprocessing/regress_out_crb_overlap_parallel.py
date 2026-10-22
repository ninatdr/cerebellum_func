import nibabel as nib
from nilearn import image, masking
import numpy as np
import os
import sys
import glob

# Get command-line arguments
subject_id = sys.argv[1]
folder1 = sys.argv[2]  # First search folder
folder2 = sys.argv[3]  # Second search folder
output_folder = sys.argv[4]  # Output directory

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load masks
crbl_mask = nib.load("/home/k2252300/Desktop/01_dHCP/cerebellum_func/masks/crbl/crbl_dhcp_2mm.nii.gz")
overlap_mask = nib.load("/home/k2252300/Desktop/01_dHCP/data/01_preprocessed_termborn/masks_templates/whole_brain/crb_crbl_overlap_6mm_sphere_bin_offset.nii.gz")

# Search for the subject file in both folders
match = glob.glob(os.path.join(folder1, f"{subject_id}*.nii.gz"))
if not match:
    match = glob.glob(os.path.join(folder2, f"{subject_id}*.nii.gz"))

# If a match is found, proceed
if match:
    print(f"{subject_id} was found.")
    fname = match[0]
    img = nib.load(fname)

    # Extract time series from overlap mask
    overlap_data = masking.apply_mask(img, overlap_mask)
    overlap_mean_ts = np.mean(overlap_data, axis=1, keepdims=True)

    # Regress out overlap mean timeseries
    cleaned_crbl = image.clean_img(img, confounds=overlap_mean_ts, mask_img=crbl_mask)

    # Save output
    output_file = os.path.join(output_folder, f"{subject_id}_2mmRegress_CRBLcleaned.nii.gz")
    cleaned_crbl.to_filename(output_file)
else:
    print(f"{subject_id} not found in either folder.")
