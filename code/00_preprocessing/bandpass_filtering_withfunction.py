import nibabel as nib
from nilearn import image, signal
import numpy as np
import os
import sys
import glob

filename = sys.argv[1]
input_folder = sys.argv[2]
output_folder = sys.argv[3]

os.makedirs(output_folder, exist_ok=True)

search_pattern = os.path.join(input_folder, f"{filename}*.nii.gz")
matching_files = glob.glob(search_pattern)
full_path = matching_files[0]

print(f"Loading file: {full_path}")

if not matching_files:
    raise FileNotFoundError(f"No matching file found for {filename} in {input_folder}")


def bandpass_filter_4D(image_path, TR, low_pass, high_pass, output_folder, save=False, order=5):
    basename=os.path.basename(image_path).replace(".nii.gz", "")
    image = nib.load(image_path)
    data = image.get_fdata()
    print(f"{basename} loaded.")
    x,y,z,t = data.shape
    data_reshape = data.reshape(-1,t).T
    
    fs=1/TR
    filtered_data=signal.butterworth(data_reshape, fs, low_pass=low_pass, high_pass=high_pass, order=order)
    filtered_data_reshape=filtered_data.T.reshape(x,y,z,t)
    print(f"Low-pass filter {low_pass} and high-pass filter {high_pass} applied.")
    save=save
    
    if save == True:
        filtered_data_reshape_img = nib.Nifti1Image(filtered_data_reshape, affine=image.affine)
        output_file=f"{output_folder}/{basename}_bpft{low_pass}.nii.gz"
        filtered_data_reshape_img.to_filename(output_file)
        print(f"{basename} has been saved as {os.path.basename(output_file)}.")
    
    else:
        print(f"{basename} has not been saved as file.")

TR=0.392
bandpass_filter_4D(full_path,TR,0.2,0.01,output_folder, save=True)
	
