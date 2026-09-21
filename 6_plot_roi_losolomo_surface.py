# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: August 22, 2026

"""
Per-ROI brain maps from the ROI-mean LOSO x LOMO encoding results.

Reads the master CSV (built by build_results_csv_roi_losolomo_roimean.py),
averages each ROI's values across subjects, paints them onto the
Schaefer-200 parcellation, and saves a NIfTI plus fsaverage surface views --
per layer.

Unlike plot_roi_surface.py (the plain-LOSO version), there's no --movie
argument here and no per-movie vs all-movies distinction: LOSO x LOMO
already pooled across every movie inside the cross-validation loop itself,
so the master CSV has exactly one row per (roi, subject) already -- there's
nothing left to average across movies for.

Three quantities can be mapped (--value):
    norm     layer_R / ceiling_R   <- DEFAULT. What the semantic model ADDS over
                                      simply averaging the other subjects.
    R        raw model R           <- tracks how predictable a region is in
                                      general, NOT what the model contributes.
    ceiling  inter-subject ceiling <- how much shared signal exists at all.

Usage:
    python plot_roi_surface_losolomo_roimean.py
    python plot_roi_surface_losolomo_roimean.py --value R
    python plot_roi_surface_losolomo_roimean.py --layer layer16
"""

import os
import ast
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
from nilearn import datasets, image, surface, plotting

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CSV_PATH    = "../outputs/encoding_results_roi_mean_losolomo_trimmed/master_per_subject_roi_losolomo_roimean.csv"
SUBJECT_DIR = "../data/subjects"
FMRIPREP    = "/project/ycleong/datasets/EmoFilm_prep"
OUT_DIR     = "../outputs/encoding_results_roi_mean_losolomo_trimmed/figures"

ATLAS_PATH = ("/project/ycleong/users/kumiko/methods_general/Atlas/Schaefer200/"
              "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")

LAYERS = ["layer12", "layer16", "layer20"]
N_ROI  = 200
MESH   = "fsaverage5"

INTERP    = "linear"
N_SAMPLES = 50
RADIUS    = 3.0

VMAX_PCT   = 99
THRESH_PCT = 50

def load_subject_ids(txt_path):
    with open(txt_path) as f:
        return ast.literal_eval(f.read())

def find_reference_mask():
    """Any subject's MNI brain mask gives the grid to resample the atlas onto."""
    ids = load_subject_ids(Path(SUBJECT_DIR) / "all_IDs.txt")
    for sid in ids:
        for sess in ["ses-1", "ses-2", "ses-3", "ses-4"]:
            pattern = os.path.join(
                FMRIPREP, f"sub-{sid}", sess, "func",
                f"sub-{sid}_{sess}_task-*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz"
            )
            hits = sorted(glob.glob(pattern))
            if hits:
                return hits[0]
    return None

def load_atlas(ref_img, atlas_path=None):
    """Schaefer-200 resampled onto the BOLD grid. Nearest-neighbour: labels
    are categorical, so interpolating them would invent nonexistent parcels."""
    path = atlas_path or ATLAS_PATH
    if os.path.exists(path):
        atlas_img = image.load_img(path)
        print(f"  atlas: {path}")
    else:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROI)
        atlas_img = image.load_img(atlas["maps"])
        print(f"  atlas: fetched Schaefer-{N_ROI}")

    atlas_rs = image.resample_to_img(atlas_img, ref_img, interpolation="nearest")
    labels = np.asarray(atlas_rs.get_fdata()).astype(np.int32)
    print(f"  atlas resampled to {labels.shape}")
    return labels

def roi_values_to_img(roi_values, atlas_data, affine):
    """Paint a length-200 vector onto the parcellation. NaN entries (missing
    ROIs) are left at zero, so they render as background."""
    out = np.zeros(atlas_data.shape, dtype=np.float32)
    for k in range(1, N_ROI + 1):
        v = roi_values[k - 1]
        if not np.isnan(v):
            out[atlas_data == k] = v
    return nib.Nifti1Image(out, affine)

def _plot_hemis(lh_tex, rh_tex, fsavg, thresh, vmax, vmin, prefix, title_stem):
    """Four panels: lateral + medial, both hemispheres."""
    for hemi, tex, infl, sulc in [
        ("lh", lh_tex, fsavg.infl_left, fsavg.sulc_left),
        ("rh", rh_tex, fsavg.infl_right, fsavg.sulc_right),
    ]:
        for view in ["lateral", "medial"]:
            out_png = os.path.join(OUT_DIR, f"{prefix}_surf_{view}_{hemi}.png")
            plotting.plot_surf_stat_map(
                infl, tex,
                hemi="left" if hemi == "lh" else "right",
                view=view,
                cmap="hot",
                colorbar=True,
                threshold=thresh,
                vmin=vmin,
                vmax=vmax,
                bg_map=sulc,
                bg_on_data=False,
                title=f"{title_stem} ({hemi}, {view})",
                output_file=out_png,
            )
            print(f"  saved: {out_png} (threshold={thresh:.4f}, vmin={vmin}, vmax={vmax:.4f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="layer12", choices=LAYERS)
    ap.add_argument("--value", default="norm", choices=["norm", "R", "ceiling"])
    ap.add_argument("--atlas-path", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    col = {
        "norm":    f"{args.layer}_norm",
        "R":       f"{args.layer}_R",
        "ceiling": "ceiling_R",
    }[args.value]

    # norm = R / ceiling is a fraction (typically in [-0.5, 1]); threshold at 0,
    # not 1 (which would blank almost every parcel).
    zero_point = 0.0

    print(f"=== LOSO x LOMO (ROI-mean) | {args.layer} | {args.value} ({col}) ===")

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            f"{CSV_PATH} does not exist. Run "
            f"build_results_csv_roi_losolomo_roimean.py first."
        )
    sub = pd.read_csv(CSV_PATH)
    if sub.empty:
        raise RuntimeError(f"{CSV_PATH} is empty.")

    means = sub.groupby("roi")[col].mean()
    roi_values = np.full(N_ROI, np.nan)
    roi_values[means.index.values - 1] = means.values

    n_present = int(np.sum(~np.isnan(roi_values)))
    print(f"  {n_present}/{N_ROI} ROIs have values")
    print(f"  {args.value}: mean={np.nanmean(roi_values):.4f} "
          f"median={np.nanmedian(roi_values):.4f} "
          f"min={np.nanmin(roi_values):.4f} max={np.nanmax(roi_values):.4f}")

    order = np.argsort(-np.nan_to_num(roi_values, nan=-np.inf))
    print(f"\n  Top 15 ROIs by {args.value}:")
    for rank, i in enumerate(order[:15], 1):
        roi_id = i + 1
        r_val = sub[sub["roi"] == roi_id][f"{args.layer}_R"].mean()
        c_val = sub[sub["roi"] == roi_id]["ceiling_R"].mean()
        print(f"    {rank:2d}. ROI {roi_id:3d}: {args.value}={roi_values[i]:.4f} "
              f"(R={r_val:.4f}, ceiling={c_val:.4f})")

    print(f"\n  Bottom 5 ROIs by {args.value}:")
    valid = order[:n_present]
    for rank, i in enumerate(valid[-5:], 1):
        roi_id = i + 1
        print(f"    ROI {roi_id:3d}: {args.value}={roi_values[i]:.4f}")

    ref_path = find_reference_mask()
    if ref_path is None:
        raise FileNotFoundError("No reference brain mask found.")
    ref_img = nib.load(ref_path)
    atlas_data = load_atlas(ref_img, args.atlas_path)

    img = roi_values_to_img(roi_values, atlas_data, ref_img.affine)

    nii_out = os.path.join(
        OUT_DIR, f"losolomo_roimean_{args.layer}_{args.value}_ROI_MNI.nii.gz"
    )
    nib.save(img, nii_out)
    print(f"\n  saved NIfTI: {nii_out}")

    vol = np.asarray(img.get_fdata())
    vol_pos = np.where(vol > zero_point, vol, np.nan)
    nii_out_pos = os.path.join(
        OUT_DIR, f"losolomo_roimean_{args.layer}_{args.value}_ROI_MNI_positive.nii.gz"
    )
    nib.save(nib.Nifti1Image(vol_pos.astype(np.float32), ref_img.affine), nii_out_pos)
    print(f"  saved NIfTI: {nii_out_pos}")

    fsavg = datasets.fetch_surf_fsaverage(mesh=MESH)
    lh = surface.vol_to_surf(img, fsavg.pial_left,
                             interpolation=INTERP, n_samples=N_SAMPLES, radius=RADIUS)
    rh = surface.vol_to_surf(img, fsavg.pial_right,
                             interpolation=INTERP, n_samples=N_SAMPLES, radius=RADIUS)
    lh = np.nan_to_num(lh, nan=0.0, posinf=0.0, neginf=0.0)
    rh = np.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0)

    combined = np.concatenate([lh, rh])
    pos_vals = combined[combined > zero_point]
    if pos_vals.size:
        vmax_pos = float(np.percentile(pos_vals, VMAX_PCT))
    else:
        vmax_pos = zero_point + 0.01
    vmax_pos = max(vmax_pos, zero_point + 1e-6)
    thresh_pos = float(np.percentile(pos_vals, THRESH_PCT)) if pos_vals.size else zero_point + 1e-3
    print(f"  positive-only display: threshold={thresh_pos:.4f} "
          f"vmin={zero_point} vmax={vmax_pos:.4f} (p{VMAX_PCT})")

    _plot_hemis(
        lh, rh, fsavg, thresh=thresh_pos, vmax=vmax_pos, vmin=zero_point,
        prefix=f"losolomo_roimean_{args.layer}_{args.value}_positive",
        title_stem=f"LOSO x LOMO (ROI-mean) {args.layer} {args.value} (positive only)",
    )

    print(f"\nDone. Figures in {OUT_DIR}")

if __name__ == "__main__":
    main()
