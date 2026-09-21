# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 17, 2026

"""
Per-network (Schaefer 7-network) brain maps from the ROI-mean LOSO x LOMO encoding results.

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
    python3 6_plot_roimean_anxiety_network.py --condition trait_strong
    python3 6_plot_roimean_anxiety_network.py --condition base --value R
    python3 6_plot_roimean_anxiety_network.py --condition vignette_3 --layer layer16
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
NETWORK_RESULTS_DIR = "../outputs/encoding_results_anxious_network"
SUBJECT_DIR = "../data/subjects"
FMRIPREP    = "/project/ycleong/datasets/EmoFilm_prep"
CONDITIONS_LOC = "../data/conditions.json"

ATLAS_PATH = ("/project/ycleong/users/kumiko/methods_general/Atlas/Schaefer200/"
              "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")

NETWORK_LUT_PATH = ("../../resources/atlases/Schaefer200/Schaefer2018_200Parcels_7Networks_order.txt")

LAYERS = ["layer12", "layer16", "layer20"]
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
N_ROI  = 200
MESH   = "fsaverage5"

INTERP    = "linear"
N_SAMPLES = 50
RADIUS    = 3.0

VMAX_PCT   = 99
THRESH_PCT = 50
BACKGROUND_EPS = 1e-6

def load_subject_ids(txt_path):
    with open(txt_path) as f:
        return ast.literal_eval(f.read())

def results_dir_for(prompt_version, condition, run_tag=None):
    """Matches OUT_DIR in 8_anxious_llama_encoding_condition_aware.py and
    results_dir_for() in build_results_csv_roi_losolomo_roimean_anxious.py."""
    d = os.path.join(NETWORK_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def load_roi_to_network(lut_path=NETWORK_LUT_PATH):
    """
    Parses Schaefer order.txt lines
    roi_id -> network name
    """
    roi_to_net = {}
    with open(lut_path) as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            roi_id = int(parts[0])
            label = parts[1]
            roi_to_net[roi_id] = label.split("_")[2]
    return roi_to_net

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

def network_values_to_img(roi_values, atlas_data, affine):
    """Paint a length-200 vector (every ROI already set to its network's value) 
    onto the parcellation. NaN entries (missing networks) are left at zero, so 
    they render as background."""
    out = np.zeros(atlas_data.shape, dtype=np.float32)
    for k in range(1, N_ROI + 1):
        v = roi_values[k - 1]
        if not np.isnan(v):
            out[atlas_data == k] = v
    return nib.Nifti1Image(out, affine)

def _plot_hemis(lh_tex, rh_tex, fsavg, thresh, vmax, vmin, cmap, out_dir, prefix, title_stem):
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
        n_above_thresh = int(np.sum(np.abs(tex) > thresh))
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
                cmap=cmap,
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
    ap.add_argument("--network-lut", default=None)
    args = ap.parse_args()

    results_dir = results_dir_for(args.prompt_version, args.condition, args.run_tag)
    csv_path = os.path.join(
        results_dir,
        f"masters_per_subject_network7_losolomo_roimean_{args.prompt_version}_condition_{args.condition}.csv",
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

    net_means = sub.groupby("network")[col].mean()
    print(f"\n Network values ({args.value}, {args.layer}):")
    for net in NETWORKS:
        if net in net_means.index:
            r_val = sub[sub["network"] == net][f"{args.layer}_R"].mean()
            c_val = sub[sub["network"] == net]["ceiling_R"].mean()
            print(f"    {net:12s}: {args.value}={net_means[net]:.4f} (R={r_val:.4f}, ceiling={c_val:.4f})")
        else:
            print(f"    {net:12s}: MISSING")

    # Flatten to a length-200 ROI vector: every ROI in a network gets that
    # network's single aggregate value, so the parcellation renders as 7 flat-colored regions.
    lut_path = args.network_lut or NETWORK_LUT_PATH
    roi_to_net = load_roi_to_network(lut_path)
    roi_values = np.full(N_ROI, np.nan)
    for roi_id, net in  roi_to_net.items():
        if net in net_means.index:
            roi_values[roi_id - 1] = net_means[net]

    n_present = int(np.sum(~np.isnan(roi_values)))
    print(f"  {n_present}/{N_ROI} ROIs assigned a network value "
          f"({len(net_means)}/{len(NETWORKS)} networks present)")
    print(f"  {args.value}: mean={np.nanmean(roi_values):.4f} "
          f"median={np.nanmedian(roi_values):.4f} "
          f"min={np.nanmin(roi_values):.4f} max={np.nanmax(roi_values):.4f}")

    ranked = net_means.reindex(NETWORKS).dropna().sort_values(ascending=False)
    print(f"\n Networks ranked by {args.value}:")
    for rank, (net, val) in enumerate(ranked.items(), 1):
        print(f"    {rank}. {net:12s}: {args.value}={val:.4f}")

    ref_path = find_reference_mask()
    if ref_path is None:
        raise FileNotFoundError("No reference brain mask found.")
    ref_img = nib.load(ref_path)
    atlas_data = load_atlas(ref_img, args.atlas_path)

    img = network_values_to_img(roi_values, atlas_data, ref_img.affine)

    prefix_base = f"losolomo_roimean_{args.prompt_version}_condition_{args.condition}_{args.layer}_{args.value}_network7"

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
    real_vals = combined[np.abs(combined) > BACKGROUND_EPS]
    if real_vals.size:
        vmin = float(np.min(real_vals))
        vmax = float(np.max(real_vals))
    else:
        vmin, vmax = -1e-3, 1e-3
    # Guard against a totally flat map (e.g. only one network present).
    if vmax <= vmin:
        pad = max(abs(vmax), 1e-3) * 0.05
        vmin, vmax = vmin - pad, vmax + pad
    cmap = "coolwarm" if (vmin < 0 < vmax) else "hot"
    print(f"  full-range display: threshold={BACKGROUND_EPS:.6f} (background only) "
          f"vmin={vmin:.4f} vmax={vmax:.4f} cmap={cmap}")

    _plot_hemis(
        lh, rh, fsavg, thresh=BACKGROUND_EPS, vmax=vmax, vmin=vmin, cmap=cmap,
        out_dir=out_dir,
        prefix=f"{prefix_base}_full",
        title_stem=f"LOSO x LOMO (ROI-mean) {args.condition} {args.layer} {args.value} (7 networks)",
    )

    print(f"\nDone. Figures in {out_dir}")

if __name__ == "__main__":
    main()
