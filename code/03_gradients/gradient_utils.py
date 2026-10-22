import glob
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import spearmanr
from brainspace.gradient import GradientMaps
from SUITPy import flatmap


# =====================
# IO utilities
# =====================

def dhcp_to_mni(infile, mni_ref, warp_file):
    infile = Path(infile)
    outfile = infile.with_name(
        infile.stem.replace(".nii", "") + "_MNILin6aAsym.nii.gz"
    )
    subprocess.run(
        [
            "applywarp",
            "-i", str(infile),
            "-r", str(mni_ref),
            "-o", str(outfile),
            "-w", str(warp_file),
        ],
        check=True,
    )
    return outfile


def volume_to_surface(nifti_file, space="FSL"):
    return flatmap.vol_to_surf(nib.load(nifti_file), space=space)


# =====================
# Shared helpers
# =====================

def threshold_and_crop(matrix, mask, n_crb, threshold, mode):
    """
    mode:
        'crb_crbl'  -> cerebro–cerebellar  [-n_vox:, :n_crb]
        'intracrbl' -> intracerebellar     [-n_vox:, -n_vox:]
    """
    mask_data = mask.get_fdata()
    n_vox = (mask_data == 1).sum()

    mat = matrix.copy()
    mat[mat < threshold] = 0

    if mode == "crb_crbl":
        data = mat[-n_vox:, :n_crb]
    elif mode == "intracrbl":
        data = mat[-n_vox:, -n_vox:]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return data, mask_data


def gradients_to_volume(gradients, mask_data, affine):
    vol = np.zeros(mask_data.shape + (gradients.shape[1],))
    for g in range(gradients.shape[1]):
        vol[mask_data == 1, g] = gradients[:, g]
    return nib.Nifti1Image(vol, affine)


# =====================
# Group gradients
# =====================

def compute_group_gradients(
    fc_matrix,
    mask,
    outdir,
    *,
    mode,
    n_crb,
    threshold,
    mni_ref,
    warp_file,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, mask_data = threshold_and_crop(
        fc_matrix, mask, n_crb, threshold, mode
    )

    G = GradientMaps(kernel="cosine", approach="dm")
    G.fit(data)

    # --- save gradients & lambdas ---
    np.save(outdir / "group_gradients.npy", G.gradients_)
    np.savetxt(outdir / "group_lambdas.txt", G.lambdas_, fmt="%.6f")

    # --- volume ---
    vol = gradients_to_volume(G.gradients_, mask_data, mask.affine)
    vol_file = outdir / "group_gradients.nii.gz"
    vol.to_filename(vol_file)

    # --- MNI + surface ---
    mni = dhcp_to_mni(vol_file, mni_ref, warp_file)
    surf = volume_to_surface(mni)
    np.save(outdir / "group_gradients_vert.npy", surf)

    return G.gradients_, surf


# =====================
# Subject gradients
# =====================

def compute_subject_gradients(
    subject_ids,
    matrix_dir,
    mask,
    template_gradients,
    outdir,
    *,
    mode,
    n_crb,
    threshold,
    mni_ref,
    warp_file,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sid in subject_ids:
        matches = glob.glob(str(Path(matrix_dir) / f"{sid}*.npy"))
        if not matches:
            print(f"[WARN] no matrix for {sid}")
            continue

        mat = np.load(matches[0])
        data, mask_data = threshold_and_crop(
            mat, mask, n_crb, threshold, mode
        )

        G = GradientMaps(
            kernel="cosine",
            approach="dm",
            alignment="procrustes",
            
        )
        G.fit(data, sparsity=0, reference=template_gradients)

        np.save(outdir / f"{sid}_aligned_gradients.npy", G.aligned_)

        vol = gradients_to_volume(G.aligned_, mask_data, mask.affine)
        vol_file = outdir / f"{sid}_aligned_gradients.nii.gz"
        vol.to_filename(vol_file)

        mni = dhcp_to_mni(vol_file, mni_ref, warp_file)
        surf = volume_to_surface(mni)
        np.save(outdir / f"{sid}_aligned_gradients_vert.npy", surf)


# =====================
# Correlations
# =====================

def subject_template_correlations(
    subject_ids,
    template_grad,
    template_surf,
    subject_dir,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_v, rows_s = [], []

    for sid in subject_ids:
        vol = np.load(Path(subject_dir) / f"{sid}_aligned_gradients.npy")
        surf = np.load(Path(subject_dir) / f"{sid}_aligned_gradients_vert.npy")

        rv, rs = {"subject": sid}, {"subject": sid}

        for g in range(template_grad.shape[1]):
            rv[f"g{g+1}"] = abs(spearmanr(vol[:, g], template_grad[:, g])[0])
            rs[f"g{g+1}"] = abs(spearmanr(surf[:, g], template_surf[:, g])[0])

        rows_v.append(rv)
        rows_s.append(rs)

    pd.DataFrame(rows_v).to_csv(outdir / "corr_volume.csv", index=False)
    pd.DataFrame(rows_s).to_csv(outdir / "corr_surface.csv", index=False)
