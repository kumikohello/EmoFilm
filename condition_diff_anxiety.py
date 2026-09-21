# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 6, 2026

"""
Compute the per-subject, per-ROI, per-layer difference in LOSO x LOMO
ROI-mean encoding R between two anxious-model conditions, correlate that
difference with trait anxiety (DASS_anx), AND plot both as Schaefer-200
surface maps -- all in one script.

This asks: does how much a subject's encoding-model fit CHANGES between two
conditions relate to that subject's trait anxiety? That's different from
correlating raw R against anxiety -- it's specifically about the
anxiety-condition MANIPULATION's effect (condition_a minus condition_b),
not the model's absolute performance in either condition alone.

Works for ANY pair of conditions in conditions.json, not just control vs
base -- e.g. trait_strong vs base, vignette_3 vs control, etc. Default is
control - base since that's the most natural "manipulation vs baseline"
comparison, but every condition (including base itself) can be either side.

Reads the pkls written by 8_anxious_llama_encoding_condition_aware.py:
    ../outputs/encoding_results_anxious_conditions/{prompt_version}_condition_{condition}/roi*_losolomo_meanR.pkl

Outputs:
  1. per-subject diff CSV (roi, layer, subject_id, R_a, R_b, diff)
  2. diff-anxiety correlation CSV
     (source, condition_a, condition_b, layer, roi, r, p, n_subjects)
  3. per-layer surface maps (NIfTI + fsaverage PNGs) of:
       - the mean diff across subjects, per ROI
       - the diff-anxiety Spearman r, per ROI

Usage:
    python condition_diff_anxiety.py
    python condition_diff_anxiety.py --condition-a trait_strong --condition-b base
    python condition_diff_anxiety.py --condition-a vignette_3 --condition-b control --skip-plots
"""

import os
import ast
import glob
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from nilearn import datasets, image, surface, plotting

from correlate_anxiety import IN_TSV, N_ROI, MIN_SUBJECTS_PER_ROI, load_anxiety_scores

# nicer diverging colormap if nilearn's "cold_hot" is registered with
# matplotlib; otherwise fall back to coolwarm (same fallback pattern as
# plot_schaefer_betas_anxious.py / correlate_anxiety_roi_losolomo_anxious_grouped.py)
try:
    CMAP = plt.get_cmap("cold_hot")
except (ValueError, KeyError):
    CMAP = plt.get_cmap("coolwarm")

BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"
CONDITIONS_LOC = "../data/conditions.json"
LAYERS = ["layer12", "layer16", "layer20"]
OUT_DIR = "../outputs/condition_diff_anxiety"

# ---- plotting config (same as plot_roi_surface_losolomo_roimean_anxious.py) ----
SUBJECT_DIR = "../data/subjects"
FMRIPREP    = "/project/ycleong/datasets/EmoFilm_prep"
ATLAS_PATH  = ("/project/ycleong/users/kumiko/methods_general/Atlas/Schaefer200/"
               "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
MESH        = "fsaverage5"
INTERP      = "linear"
N_SAMPLES   = 50
RADIUS      = 3.0
VMAX_PCT    = 99
THRESH_PCT  = 50

def results_dir_for(prompt_version, condition, run_tag=None):
    """Matches OUT_DIR in 8_anxious_llama_encoding_condition_aware.py."""
    d = os.path.join(BASE_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def load_roi_R(results_dir):
    """Read all LOSO x LOMO ROI-mean pkls for one condition.

    Returns {layer: {roi: {sid: R}}}.
    """
    pkls = sorted(glob.glob(os.path.join(results_dir, "roi*_losolomo_meanR.pkl")))
    if not pkls:
        raise RuntimeError(f"No pkls found in {results_dir}")
    print(f"  found {len(pkls)} ROI pkls in {results_dir}")

    by_layer = {layer: {} for layer in LAYERS}
    for path in pkls:
        with open(path, "rb") as f:
            res = pickle.load(f)
        roi = res["roi"]
        for layer in LAYERS:
            if layer in res.get("layers", {}):
                by_layer[layer][roi] = dict(res["layers"][layer])  # {sid: R}
    return by_layer

def compute_diff(by_layer_a, by_layer_b):
    """Returns a long dataframe: roi, layer, subject_id, R_a, R_b, diff.

    Only ROIs/subjects present in BOTH conditions are kept (an intersection,
    not a union) -- a subject or ROI missing from one condition can't
    contribute a difference.
    """
    rows = []
    for layer in LAYERS:
        rois_a = by_layer_a.get(layer, {})
        rois_b = by_layer_b.get(layer, {})
        common_rois = sorted(set(rois_a) & set(rois_b))
        missing_rois = sorted(set(rois_a) ^ set(rois_b))
        if missing_rois:
            print(f"  [{layer}] {len(missing_rois)} ROIs present in only one condition, skipped: "
                  f"{missing_rois[:10]}{' ...' if len(missing_rois) > 10 else ''}")

        for roi in common_rois:
            sid_R_a = rois_a[roi]
            sid_R_b = rois_b[roi]
            common_sids = sorted(set(sid_R_a) & set(sid_R_b))
            missing_sids = sorted(set(sid_R_a) ^ set(sid_R_b))
            if missing_sids:
                print(f"    [roi {roi:03d}/{layer}] {len(missing_sids)} subjects present in only "
                      f"one condition, dropped: {missing_sids}")

            for sid in common_sids:
                r_a = sid_R_a[sid]
                r_b = sid_R_b[sid]
                rows.append({
                    "roi": roi,
                    "layer": layer,
                    "subject_id": sid,
                    "R_a": r_a,
                    "R_b": r_b,
                    "diff": r_a - r_b,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No overlapping ROI/subject data between the two conditions. "
                            "Did both encoding jobs finish?")
    return df

def roiwise_correlation(df, anxiety):
    """df: long dataframe with columns roi, layer, subject_id, diff.

    Returns list of row dicts: {layer, roi, r, p, n_subjects}.
    """
    rows = []
    for layer in LAYERS:
        sub_layer = df[df["layer"] == layer]
        if sub_layer.empty:
            print(f"  [skip] no rows for {layer}")
            continue

        n_valid = 0
        for roi, sub_roi in sub_layer.groupby("roi"):
            sids = [sid for sid in sub_roi["subject_id"] if sid in anxiety.index]
            if len(sids) < MIN_SUBJECTS_PER_ROI:
                continue

            sub_roi_indexed = sub_roi.set_index("subject_id")
            x = sub_roi_indexed.loc[sids, "diff"].to_numpy(dtype=np.float64)
            y = anxiety.loc[sids].to_numpy(dtype=np.float64)

            if np.std(x) == 0 or np.std(y) == 0:
                continue

            r, p = stats.spearmanr(x, y)
            rows.append({"layer": layer, "roi": int(roi), "r": r, "p": p, "n_subjects": len(sids)})
            n_valid += 1

        print(f"  {layer}: {n_valid} ROIs correlated")

    return rows

# -----------------------------------------------------------------------------
# Plotting (same atlas/guard logic as plot_roi_surface_losolomo_roimean_anxious.py)
# -----------------------------------------------------------------------------
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

def plot_roi_map_grouped(roi_values, atlas_data, ref_img, fsavg, out_dir, prefix, title_stem):
    """Paint a length-N_ROI vector onto the surface and save NIfTI + ONE
    combined 2x2 figure (left/right x lateral/medial), matching the grouped
    layout in correlate_anxiety_roi_losolomo_anxious_grouped.py, instead of
    four separate per-view PNGs.

    Always symmetric (vmin = -vmax) since diffs and correlations can be
    positive or negative -- unlike the raw R/norm/ceiling maps elsewhere in
    this project, which only ever plot the positive, above-ceiling-relative
    range. No thresholding is applied here (matching the grouped reference
    script), which also sidesteps the "everything gets masked out" crash we
    hit with threshold-based positive-only maps -- there's no threshold to
    mask everything out with.
    """
    img = roi_values_to_img(roi_values, atlas_data, ref_img.affine)

    nii_out = os.path.join(out_dir, f"{prefix}_ROI_MNI.nii.gz")
    nib.save(img, nii_out)

    texture_l = surface.vol_to_surf(img, fsavg.pial_left,
                                     interpolation=INTERP, n_samples=N_SAMPLES, radius=RADIUS)
    texture_r = surface.vol_to_surf(img, fsavg.pial_right,
                                     interpolation=INTERP, n_samples=N_SAMPLES, radius=RADIUS)

    finite_vals = np.concatenate([
        texture_l[np.isfinite(texture_l)],
        texture_r[np.isfinite(texture_r)],
    ])
    finite_vals = finite_vals[finite_vals != 0]
    vmax = float(np.percentile(np.abs(finite_vals), VMAX_PCT)) if finite_vals.size else 1e-3
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-3

    print(f"  {prefix}: vmin={-vmax:.4f} vmax={vmax:.4f} (no threshold, grouped 2x2 figure)")

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(14, 10))
    panels = [
        (fsavg.infl_left,  texture_l, "left",  "lateral", axes[0, 0], "Left lateral"),
        (fsavg.infl_right, texture_r, "right", "lateral", axes[0, 1], "Right lateral"),
        (fsavg.infl_left,  texture_l, "left",  "medial",  axes[1, 0], "Left medial"),
        (fsavg.infl_right, texture_r, "right", "medial",  axes[1, 1], "Right medial"),
    ]
    for mesh, tex, hemi, view, ax, ttl in panels:
        plotting.plot_surf_stat_map(
            mesh, tex, hemi=hemi, view=view, axes=ax,
            colorbar=False, cmap=CMAP, vmin=-vmax, vmax=vmax,
            symmetric_cbar=True, title=ttl,
        )

    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    fig.colorbar(sm, cax=fig.add_axes([0.92, 0.35, 0.015, 0.30]))
    fig.suptitle(title_stem, fontsize=16, fontweight="bold")
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"{prefix}_ROI_grouped.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved NIfTI: {nii_out}")
    print(f"  saved PNG:   {png_path}")

def main():
    with open(CONDITIONS_LOC, "r") as f:
        cons = json.load(f)
    conditions = [k for k in cons]

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-a", default="base", choices=conditions,
                     help="Diff = R(condition-a) - R(condition-b). Default 'control'. "
                          "Any condition, including 'base', can go on either side.")
    ap.add_argument("--condition-b", default="control", choices=conditions,
                     help="Diff = R(condition-a) - R(condition-b). Default 'base'.")
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--run-tag", default=None,
                     help="Only needed if the encoding run used --run-tag.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--atlas-path", default=None)
    ap.add_argument("--skip-plots", action="store_true",
                     help="Only write the CSVs, skip the surface-map plotting step.")
    args = ap.parse_args()

    if args.condition_a == args.condition_b:
        raise ValueError("--condition-a and --condition-b must differ (diff would be all zero).")

    tag = f"{args.condition_a}_minus_{args.condition_b}"
    os.makedirs(args.out_dir, exist_ok=True)
    diff_out_path = os.path.join(args.out_dir, f"roi_diff_per_subject_{tag}.csv")
    corr_out_path = os.path.join(args.out_dir, f"roi_diff_anxiety_corr_{tag}.csv")
    fig_dir = os.path.join(args.out_dir, "figures", tag)

    dir_a = results_dir_for(args.prompt_version, args.condition_a, args.run_tag)
    dir_b = results_dir_for(args.prompt_version, args.condition_b, args.run_tag)

    print(f"=== diff = {args.condition_a} - {args.condition_b} (prompt_version={args.prompt_version}) ===")
    print(f"[1/4] Loading condition_a ({args.condition_a}): {dir_a}")
    by_layer_a = load_roi_R(dir_a)
    print(f"[1/4] Loading condition_b ({args.condition_b}): {dir_b}")
    by_layer_b = load_roi_R(dir_b)

    print("[2/4] Computing per-subject diff...")
    diff_df = compute_diff(by_layer_a, by_layer_b)
    diff_df.to_csv(diff_out_path, index=False)
    print(f"  wrote {len(diff_df)} rows -> {diff_out_path}")
    print("  mean diff by layer:")
    print(diff_df.groupby("layer")["diff"].agg(["mean", "std", "count"]))

    print("\n[3/4] Correlating diff with DASS_anx...")
    anxiety = load_anxiety_scores(IN_TSV)
    print(f"  loaded DASS_anx for {len(anxiety)} participants")

    corr_rows = roiwise_correlation(diff_df, anxiety)
    if not corr_rows:
        raise RuntimeError("No valid ROI/layer correlations produced. Check subject overlap "
                            "between the diff data and the anxiety scores file.")

    corr_df = pd.DataFrame(corr_rows)
    corr_df["source"] = "condition_diff_anxiety"
    corr_df["condition_a"] = args.condition_a
    corr_df["condition_b"] = args.condition_b
    corr_df = corr_df[["source", "condition_a", "condition_b", "layer", "roi", "r", "p", "n_subjects"]]
    corr_df.to_csv(corr_out_path, index=False)
    print(f"  wrote {len(corr_df)} rows -> {corr_out_path}")
    print(corr_df.groupby("layer").size())

    if args.skip_plots:
        print("\n--skip-plots set, done.")
        return

    print("\n[4/4] Plotting surface maps...")
    os.makedirs(fig_dir, exist_ok=True)

    ref_path = find_reference_mask()
    if ref_path is None:
        raise FileNotFoundError("No reference brain mask found.")
    ref_img = nib.load(ref_path)
    atlas_data = load_atlas(ref_img, args.atlas_path)
    fsavg = datasets.fetch_surf_fsaverage(mesh=MESH)

    for layer in LAYERS:
        # --- mean diff map, one value per ROI (averaged across subjects) ---
        diff_layer = diff_df[diff_df["layer"] == layer]
        if not diff_layer.empty:
            mean_diff = diff_layer.groupby("roi")["diff"].mean()
            diff_values = np.full(N_ROI, np.nan)
            diff_values[mean_diff.index.values - 1] = mean_diff.values

            plot_roi_map_grouped(
                diff_values, atlas_data, ref_img, fsavg, fig_dir,
                prefix=f"diff_{tag}_{layer}",
                title_stem=f"{args.condition_a} - {args.condition_b} | {layer} mean diff",
            )
        else:
            print(f"  [skip] no diff data for {layer}")

        # --- diff-anxiety correlation map, one r per ROI ---
        corr_layer = corr_df[corr_df["layer"] == layer]
        if not corr_layer.empty:
            r_values = np.full(N_ROI, np.nan)
            r_values[corr_layer["roi"].values - 1] = corr_layer["r"].values

            plot_roi_map_grouped(
                r_values, atlas_data, ref_img, fsavg, fig_dir,
                prefix=f"diff_anxiety_corr_{tag}_{layer}",
                title_stem=f"{args.condition_a} - {args.condition_b} | {layer} diff-anxiety r",
            )
        else:
            print(f"  [skip] no correlation data for {layer}")

    print(f"\nDone. Figures in {fig_dir}")

if __name__ == "__main__":
    main()
