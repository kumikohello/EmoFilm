# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 6, 2026

"""
Per-ROI brain maps from the ROI-mean LOSO x LOMO encoding results.

CONDITION-AWARE VERSION: adds --condition and --prompt-version so this reads
the per-condition master CSV built by
build_results_csv_roi_losolomo_roimean_anxious.py, under
    ../outputs/encoding_results_anxious_conditions/{prompt_version}_condition_{condition}/[run_tag/]
and writes figures to a matching per-condition figures subfolder, so
different conditions never overwrite each other's output.

Reads the master CSV, averages each ROI's values across subjects, paints
them onto the Schaefer-200 parcellation, and saves a NIfTI plus fsaverage
surface views -- per layer.

There's no --movie argument here and no per-movie vs all-movies distinction:
LOSO x LOMO already pooled across every movie inside the cross-validation
loop itself, so the master CSV has exactly one row per (roi, subject)
already -- there's nothing left to average across movies for.

Three quantities can be mapped (--value):
    norm     layer_R / ceiling_R   <- DEFAULT. What the semantic model ADDS over
                                      simply averaging the other subjects.
    R        raw model R           <- tracks how predictable a region is in
                                      general, NOT what the model contributes.
    ceiling  inter-subject ceiling <- how much shared signal exists at all.

Usage:
    python plot_roi_surface_losolomo_roimean_anxious.py --condition trait_strong
    python plot_roi_surface_losolomo_roimean_anxious.py --condition base --value R
    python plot_roi_surface_losolomo_roimean_anxious.py --condition vignette_3 --layer layer16
"""

import os
import ast
import glob
import json
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
BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"
SUBJECT_DIR = "../data/subjects"
FMRIPREP    = "/project/ycleong/datasets/EmoFilm_prep"
CONDITIONS_LOC = "../data/conditions.json"

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

def results_dir_for(prompt_version, condition, run_tag=None):
    """Matches OUT_DIR in 8_anxious_llama_encoding_condition_aware.py and
    results_dir_for() in build_results_csv_roi_losolomo_roimean_anxious.py."""
    d = os.path.join(BASE_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

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

def _plot_hemis(lh_tex, rh_tex, fsavg, thresh, vmax, vmin, out_dir, prefix, title_stem):
    """Four panels: lateral + medial, both hemispheres.

    Guards against two ways plot_surf_stat_map's colorbar crashes deep
    inside matplotlib (ValueError: x and y arguments to pcolormesh cannot
    have non-finite values) instead of raising a clear message:

    1. vmin == vmax (or non-finite): degenerate color range with no spread.
    2. Nothing survives thresholding: even with a valid vmin/vmax range, if
       every vertex value is <= thresh, nilearn masks the entire surface
       before handing it to matplotlib, leaving the colorbar with nothing
       finite to draw -- this is the actual cause seen with "norm > 1.0"
       positive-only maps, since a model's R essentially never exceeds the
       noise ceiling, so pos_vals is empty/near-empty on every layer and
       condition, not just when data happens to be incomplete.
    """
    for hemi, tex, infl, sulc in [
        ("lh", lh_tex, fsavg.infl_left, fsavg.sulc_left),
        ("rh", rh_tex, fsavg.infl_right, fsavg.sulc_right),
    ]:
        finite = tex[np.isfinite(tex)]
        if finite.size == 0:
            print(f"  [skip] {prefix} ({hemi}): texture is entirely non-finite, nothing to plot")
            continue
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print(f"  [skip] {prefix} ({hemi}): degenerate color range "
                  f"(vmin={vmin}, vmax={vmax}), skipping to avoid a matplotlib crash")
            continue
        n_above_thresh = int(np.sum(tex > thresh))
        if n_above_thresh == 0:
            print(f"  [skip] {prefix} ({hemi}): no vertices exceed threshold={thresh:.4f} "
                  f"(surface would be fully masked), skipping to avoid a matplotlib crash")
            continue

        for view in ["lateral", "medial"]:
            out_png = os.path.join(out_dir, f"{prefix}_surf_{view}_{hemi}.png")
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
            print(f"  saved: {out_png} (threshold={thresh:.4f}, vmin={vmin}, vmax={vmax:.4f}, "
                  f"{n_above_thresh}/{tex.size} vertices above threshold)")

def main():
    with open(CONDITIONS_LOC, "r") as f:
        cons = json.load(f)
    conditions = [k for k in cons]

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=conditions)
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--run-tag", default=None,
                     help="Only needed if the encoding/CSV-building run used --run-tag.")
    ap.add_argument("--layer", default="layer12", choices=LAYERS)
    ap.add_argument("--value", default="norm", choices=["norm", "R", "ceiling"])
    ap.add_argument("--atlas-path", default=None)
    args = ap.parse_args()

    results_dir = results_dir_for(args.prompt_version, args.condition, args.run_tag)
    csv_path = os.path.join(
        results_dir,
        f"master_per_subject_roi_losolomo_roimean_{args.prompt_version}_condition_{args.condition}.csv",
    )
    out_dir = os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    col = {
        "norm":    f"{args.layer}_norm",
        "R":       f"{args.layer}_R",
        "ceiling": "ceiling_R",
    }[args.value]

    # norm = R / ceiling is a fraction (typically in [-0.5, 1]); threshold at 0,
    # not 1 (which would blank almost every parcel).
    zero_point = 0.0

    print(f"=== LOSO x LOMO (ROI-mean) | condition={args.condition} | "
          f"{args.layer} | {args.value} ({col}) ===")

    if not os.path.exists(csv_path):
        raise RuntimeError(
            f"{csv_path} does not exist. Run "
            f"build_results_csv_roi_losolomo_roimean_anxious.py --condition {args.condition} "
            f"--prompt-version {args.prompt_version} first."
        )
    sub = pd.read_csv(csv_path)
    if sub.empty:
        raise RuntimeError(f"{csv_path} is empty.")

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

    prefix_base = f"losolomo_roimean_{args.prompt_version}_condition_{args.condition}_{args.layer}_{args.value}"

    nii_out = os.path.join(out_dir, f"{prefix_base}_ROI_MNI.nii.gz")
    nib.save(img, nii_out)
    print(f"\n  saved NIfTI: {nii_out}")

    vol = np.asarray(img.get_fdata())
    vol_pos = np.where(vol > zero_point, vol, np.nan)
    nii_out_pos = os.path.join(out_dir, f"{prefix_base}_ROI_MNI_positive.nii.gz")
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
        out_dir=out_dir,
        prefix=f"{prefix_base}_positive",
        title_stem=f"LOSO x LOMO (ROI-mean) {args.condition} {args.layer} {args.value} (positive only)",
    )

    print(f"\nDone. Figures in {out_dir}")

if __name__ == "__main__":
    main()
