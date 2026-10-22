import numpy as np
import nibabel as nib
from pathlib import Path

import sys
sys.path.append("/home/k2252300/Desktop/01_dHCP/cerebellum_func/03_gradients/utilities")


from gradient_utils import (
    compute_group_gradients,
    compute_subject_gradients,
    subject_template_correlations,
)

# ------------------
# paths + inputs
# ------------------

BASE = Path("/home/k2252300/Desktop/01_dHCP/cerebellum_func")

mask = nib.load(BASE / "masks/crbl/crbl_dhcp_2mm.nii.gz")
template_fc = np.load(BASE / "02_FC_conn/output/template_FC_matrix/template_FC_matrix.npy")

subject_ids = (BASE / "subject_list/all_subjects.txt").read_text().splitlines()
subject_matrix_dir = "/home/k2252300/Desktop/01_dHCP/cerebellum_func/correct_FC_matrices_crblCORR/subject_FC_matrices"

mni_ref = Path("/home/k2252300/Desktop/01_dHCP/cerebellum_func/templates_transforms/MNI152_T1_2mm_brain.nii.gz")
warp_file = Path("/home/k2252300/Desktop/01_dHCP/cerebellum_func/templates_transforms/extdhcp40wk_to_MNI152NLin6ASym_warp.nii.gz")

# ------------------
# GROUP GRADIENTS
# ------------------

# cerebro–cerebellar
cbcrb_grad, cbcrb_surf = compute_group_gradients(
    template_fc,
    mask,
    BASE / "03_gradients/cerebro_cerebellar",
    mode="crb_crbl",
    n_crb=18360,
    threshold=0.0,
    mni_ref=mni_ref,
    warp_file=warp_file,
)

# intra–cerebellar
crbcrb_grad, crbcrb_surf = compute_group_gradients(
    template_fc,
    mask,
    BASE / "03_gradients/intracerebellar",
    mode="intracrbl",
    n_crb=18360,
    threshold=0.0,
    mni_ref=mni_ref,
    warp_file=warp_file,
)

# ------------------
# SUBJECT GRADIENTS
# ------------------

compute_subject_gradients(
    subject_ids,
    subject_matrix_dir,
    mask,
    cbcrb_grad,
    BASE / "03_gradients/cerebro_cerebellar/subjects",
    mode="crb_crbl",
    n_crb=18360,
    threshold=0.0,
    mni_ref=mni_ref,
    warp_file=warp_file,
)

compute_subject_gradients(
    subject_ids,
    subject_matrix_dir,
    mask,
    crbcrb_grad,
    BASE / "03_gradients/intracerebellar/subjects",
    mode="intracrbl",
    n_crb=18360,
    threshold=0.0,
    mni_ref=mni_ref,
    warp_file=warp_file,
)

# ------------------
# CORRELATIONS
# ------------------

subject_template_correlations(
    subject_ids,
    cbcrb_grad,
    cbcrb_surf,
    BASE / "03_gradients/cerebro_cerebellar/subjects",
    BASE / "03_gradients/cerebro_cerebellar/correlations",
)

subject_template_correlations(
    subject_ids,
    crbcrb_grad,
    crbcrb_surf,
    BASE / "03_gradients/intracerebellar/subjects",
    BASE / "03_gradients/intracerebellar/correlations",
)
