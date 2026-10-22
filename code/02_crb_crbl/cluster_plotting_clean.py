#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cerebellar flatmap plotting
────────────────────────────
Three independent plotting jobs, each boolean-gated below:

  1. Weekly degree-centrality (DC) flatmaps, term-born infants.
  2. A single dual-regression ICN t-stat flatmap (effect of PMA at scan).
  3. Weekly dual-regression group-map flatmaps for a given ICN component.

Requires FSL (for dHCP -> MNI warping) and SUITPy.
"""

import glob
import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.ticker import FuncFormatter
from SUITPy import flatmap

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

# ── Which jobs to run ───────────────────────────────────────────────────────
RUN_WEEKLY_DC_FLATMAPS   = True
RUN_SINGLE_ICN_FLATMAP   = True
RUN_WEEKLY_GROUP_MAPS    = True

# ── FSL setup ────────────────────────────────────────────────────────────────
FSL_DIR = "/path/to/fsl/share/fsl"
os.environ['FSLDIR'] = FSL_DIR
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['PATH'] += f':{FSL_DIR}/bin'

APPLYWARP_BIN     = os.path.join(FSL_DIR, "bin", "applywarp")
MNI_TEMPLATE_PATH = "/path/to/templates/MNI_transform/MNI152_T1_2mm_brain.nii.gz"
DHCP_TO_MNI_WARP  = "/path/to/extdhcp40wk_to_MNI152NLin6ASym_warp.nii.gz"

# ── Paths ────────────────────────────────────────────────────────────────────
DC_FOLDER      = "/path/to/cerebellum_func_local/02_FC_conn/degree_centrality_per_week"
DC_SUFFIX      = "centrality_map_crbl_term_thr0"
DC_WEEKS       = [39, 40, 41, 42, 43]
DC_CSCALE      = [0, 2300]

DR_FOLDER      = "/path/to/12_core_network_strength/new_dr_feb2026/dual_regression/dual_regression_all_IC20_0-2hz_perm.dr"
ICN_PMA_TSTAT  = os.path.join(DR_FOLDER, "dr_stage3_ic0001_tstat1_thrSig.nii.gz")
ICN_PMA_CSCALE = [2, 5]

GROUP_MAP_FOLDER = "/path/to/12_core_network_strength/new_dr_feb2026/dual_regression/comp3_group_maps"
GROUP_MAP_WEEKS   = [39, 40, 41, 42, 43]
GROUP_MAP_CSCALE  = [3, 12]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def dHCP_to_MNI(infile):
    """Warp a dHCP-space image into MNI152 space (nearest-neighbour interp).

    NB: an earlier version of this function used a different output suffix
    and omitted --interp=nn; this is the version that was actually in effect
    (Python uses the last definition when a function is redefined), kept here
    for behavioural continuity.
    """
    outfile = infile.replace(".nii.gz", "_MNILin6aAsym.nii.gz")
    subprocess.run([
        APPLYWARP_BIN,
        "-i", str(infile),
        "-r", MNI_TEMPLATE_PATH,
        "-o", str(outfile),
        "-w", DHCP_TO_MNI_WARP,
    ])
    return outfile


def style_colorbar(ax, ylabel, cscale, labelsize=18, label_fontsize=24):
    cbar_ax = ax.get_figure().axes[-1]
    cbar_ax.set_ylabel(ylabel)
    cbar_ax.tick_params(labelsize=labelsize)
    cbar_ax.set_yticks(cscale)
    cbar_ax.yaxis.label.set_size(label_fontsize)
    cbar_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{float(x)}'))
    return cbar_ax


# ═══════════════════════════════════════════════════════════════════════════
# JOB 1 — Weekly degree-centrality flatmaps
# ═══════════════════════════════════════════════════════════════════════════

if RUN_WEEKLY_DC_FLATMAPS:
    for wk in DC_WEEKS:
        raw_match = glob.glob(os.path.join(DC_FOLDER, f"{wk}_{DC_SUFFIX}.nii.gz"))[0]
        dhcp_map = nib.load(raw_match).get_fdata()
        global_dc = np.sum(dhcp_map)
        avg_dc = np.mean(dhcp_map[dhcp_map > 0])

        mni_match = glob.glob(os.path.join(DC_FOLDER, f"{wk}_{DC_SUFFIX}_MNI*.nii.gz"))[0]
        surface = flatmap.vol_to_surf(nib.load(mni_match), space='FSL')

        ax = flatmap.plot(data=surface, cmap='hot', new_figure=True,
                           colorbar=True, cscale=DC_CSCALE)
        ax.set_title(f"Week {wk}", fontsize=30)
        ax.text(0.5, -0.05,
                f"Average DC: {avg_dc:.2f}\nGlobal DC: {global_dc:.2f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=24)

        style_colorbar(ax, 'DC (raw)', DC_CSCALE)

        plt.savefig(os.path.join(DC_FOLDER, f"{wk}_dc_surface.png"),
                    dpi=600, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════════
# JOB 2 — Single ICN t-stat flatmap (effect of PMA at scan)
# ═══════════════════════════════════════════════════════════════════════════

if RUN_SINGLE_ICN_FLATMAP:
    mni_file = dHCP_to_MNI(ICN_PMA_TSTAT)
    surface = flatmap.vol_to_surf(nib.load(mni_file), space='FSL', ignore_zeros=True)

    ax = flatmap.plot(data=surface, cmap='hot', new_figure=True,
                       colorbar=True, cscale=ICN_PMA_CSCALE)
    style_colorbar(ax, 't-stat\n(Effect of PMA at scan)', ICN_PMA_CSCALE)


# ═══════════════════════════════════════════════════════════════════════════
# JOB 3 — Weekly dual-regression group-map flatmaps
# ═══════════════════════════════════════════════════════════════════════════

if RUN_WEEKLY_GROUP_MAPS:
    for wk in GROUP_MAP_WEEKS:
        raw_match = glob.glob(os.path.join(GROUP_MAP_FOLDER, f"{wk}wks_*group_map.nii.gz"))[0]
        mni_file = dHCP_to_MNI(raw_match)
        surface = flatmap.vol_to_surf(nib.load(mni_file), space='FSL', ignore_zeros=True)

        ax = flatmap.plot(data=surface, cmap='hot', new_figure=True,
                           colorbar=True, cscale=GROUP_MAP_CSCALE)
        ax.set_title(f"Week {wk}", fontsize=24)

        style_colorbar(ax, 't-stat', GROUP_MAP_CSCALE)

        plt.savefig(os.path.join(DR_FOLDER, f"{wk}_comp2_maps.png"), dpi=600)