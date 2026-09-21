# Authors: Kumiko Ueda (kumiko@uchicago.edu), Andrea Liu
# Last Edited: September 6, 2026

"""
Correlate LEAVE-ONE-MOVIE-OUT x LEAVE-ONE-SUBJECT-OUT, ROI-MEAN encoding
generalization (from 8_anxious_llama_encoding_condition_aware.py) with
DASS_anx anxiety score, per ROI, per layer.

CONDITION-AWARE VERSION: adds --condition and --prompt-version (and optional
--run-tag) so this reads the per-condition encoding results under
    ../outputs/encoding_results_anxious_conditions/{prompt_version}_condition_{condition}/[run_tag/]
(matching 8_anxious_llama_encoding_condition_aware.py's OUT_DIR layout)
instead of the old flat, single-condition
../outputs/encoding_results_roi_mean_losolomo_trimmed/.

Each subject's mean_R already reflects genuine cross-movie generalization:
the model was fit with that subject AND an entire movie held out
simultaneously, so this is NOT the same as correlate_anxiety.py's
avg_then_corr/corr_then_avg (post-hoc averaging of independently-fit,
single-movie LOSO results).

Because the encoding script only ever produces ROI-level results, this
script only does ROI surface plots -- no voxel-level/whole-brain option.

CSV output uses source="losolomo_roimean_anxious" and includes condition +
prompt_version columns, and is written per-condition so different
conditions' stats CSVs never collide.

Usage:
    python correlate_anxiety_roi_losolomo_anxious.py --condition trait_strong
    python correlate_anxiety_roi_losolomo_anxious.py --condition base --prompt-version original
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from nilearn import datasets, surface

from correlate_anxiety import (
    REFERENCE_MASK, IN_TSV, ATLAS_PATH, N_ROI, MESH,
    load_anxiety_scores, load_atlas, roi_values_to_img, _plot_hemis,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"   # BASE_OUT_DIR in 8_anxious_llama_encoding.py
BASE_OUT_DIR      = "../outputs/anxiety_correlation_maps_losolomo_roi_anxious"
BASE_CSV_DIR      = "../outputs/roi_anxiety_stats_losolomo_roi_anxious"
CONDITIONS_LOC = "../data/conditions.json"

LAYERS = ["layer12", "layer16", "layer20"]
MIN_SUBJECTS_PER_ROI = 5

def results_dir_for(prompt_version, condition, run_tag=None):
    """Matches OUT_DIR in 8_anxious_llama_encoding_condition_aware.py."""
    d = os.path.join(BASE_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def find_roi_pkl_roimean(roi, results_dir):
    """
    Locate the pkl for this ROI, produced by
    8_anxious_llama_encoding_condition_aware.py, for one condition.

    Returns the path, or None if not found.
    """
    p = os.path.join(results_dir, f"roi{roi:03d}_losolomo_meanR.pkl")
    return p if os.path.exists(p) else None

def roiwise_correlation_from_losolomo_roimean(anxiety, results_dir):
    """
    Reads every ROI pkl found via find_roi_pkl_roimean() for this condition's
    results_dir, and correlates each ROI's per-subject mean_R (already
    pooled across movies by the LOSO x LOMO fit itself) against DASS_anx.

    Returns:
        r_maps, p_maps, n_maps -- each {layer: (N_ROI,) array}
        missing_rois -- list of ROI ids with no pkl found
    """
    r_maps = {layer: np.full(N_ROI, np.nan, dtype=np.float64) for layer in LAYERS}
    p_maps = {layer: np.full(N_ROI, np.nan, dtype=np.float64) for layer in LAYERS}
    n_maps = {layer: np.zeros(N_ROI, dtype=np.int64) for layer in LAYERS}
    missing_rois = []
    reported_format = False

    for k in range(1, N_ROI + 1):
        path = find_roi_pkl_roimean(k, results_dir)
        if path is None:
            missing_rois.append(k)
            continue

        if not reported_format:
            print(f"  found ROI pkls at, e.g.: {path}")
            reported_format = True

        with open(path, "rb") as f:
            res = pickle.load(f)

        for layer in LAYERS:
            sid_to_R = res.get("layers", {}).get(layer, {})
            sids = [sid for sid in sid_to_R if sid in anxiety.index]
            missing_anx = [sid for sid in sid_to_R if sid not in anxiety.index]
            if missing_anx:
                print(f"    [roi {k:03d}/{layer}] {len(missing_anx)} subjects have no "
                      f"DASS_anx match, dropped: {missing_anx}")

            if len(sids) < MIN_SUBJECTS_PER_ROI:
                continue

            x = anxiety.loc[sids].to_numpy(dtype=np.float64)
            y = np.array([sid_to_R[sid] for sid in sids], dtype=np.float64)

            if np.std(y) == 0:
                continue

            r, p = stats.spearmanr(x, y)
            r_maps[layer][k - 1] = r
            p_maps[layer][k - 1] = p
            n_maps[layer][k - 1] = len(sids)

    if missing_rois:
        print(f"  {len(missing_rois)} / {N_ROI} ROI pkls not found "
              f"{missing_rois[:10]}{' ...' if len(missing_rois) > 10 else ''}")

    return r_maps, p_maps, n_maps, missing_rois

def plot_roi_surface_losolomo_roimean(roi_corr, atlas_labels_3d, ref_img, fsaverage,
                                       layer, condition, prompt_version, out_dir):
    """Same painting/projection logic as correlate_anxiety.plot_roi_surface,
    labeled for this condition-aware ROI-mean LOSO x LOMO source."""
    os.makedirs(out_dir, exist_ok=True)
    img = roi_values_to_img(roi_corr, atlas_labels_3d, ref_img.affine)

    prefix = f"losolomo_roimean_{prompt_version}_condition_{condition}_{layer}"
    nii_path = os.path.join(out_dir, f"{prefix}_anxiety_corr_ROI.nii.gz")
    nib.save(img, nii_path)

    texture_l = surface.vol_to_surf(img, fsaverage.pial_left)
    texture_r = surface.vol_to_surf(img, fsaverage.pial_right)

    _plot_hemis(texture_l, texture_r, fsaverage, prefix, f"{layer}_ROI",
                suffix="", cmap="cold_hot",
                title_suffix=f" (LOSO x LOMO, ROI-mean, {condition})", out_dir=out_dir)

    print(f"      saved -> {nii_path}")

def main():
    with open(CONDITIONS_LOC, "r") as f:
        cons = json.load(f)
    conditions = [k for k in cons]

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=conditions)
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--run-tag", default=None,
                     help="Only needed if the encoding run used --run-tag.")
    args = ap.parse_args()

    results_dir = results_dir_for(args.prompt_version, args.condition, args.run_tag)
    out_dir = os.path.join(BASE_OUT_DIR, f"{args.prompt_version}_condition_{args.condition}")
    csv_out = os.path.join(
        BASE_CSV_DIR,
        f"roi_anxiety_stats_losolomo_roi_{args.prompt_version}_condition_{args.condition}.csv",
    )

    print(f"=== condition={args.condition}, prompt_version={args.prompt_version} ===")
    print(f"reading encoding results from: {results_dir}")

    if not os.path.exists(REFERENCE_MASK):
        raise RuntimeError(f"REFERENCE_MASK does not exist: {REFERENCE_MASK}")
    ref_img = nib.load(REFERENCE_MASK)
    print(f"Reference grid: shape={ref_img.shape}, from {REFERENCE_MASK}")

    anxiety = load_anxiety_scores(IN_TSV)
    print(f"Loaded DASS_anx for {len(anxiety)} participants")

    print("\nLoading Schaefer-200 atlas...")
    atlas_flat = load_atlas(ref_img, ATLAS_PATH)
    atlas_labels_3d = atlas_flat.reshape(ref_img.shape)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=MESH)

    print(f"\n=== LOSO x LOMO generalization (ROI-mean only), condition={args.condition} ===")
    r_maps, p_maps, n_maps, missing_rois = roiwise_correlation_from_losolomo_roimean(
        anxiety, results_dir
    )

    if all(np.all(np.isnan(r_maps[layer])) for layer in LAYERS):
        raise RuntimeError(
            f"No valid correlations for any ROI/layer under {results_dir}. "
            f"Did the encoding job for condition={args.condition} finish?"
        )

    rows = []
    for layer in LAYERS:
        n_valid = int(np.sum(~np.isnan(r_maps[layer])))
        print(f"  {layer}: {n_valid} / {N_ROI} ROIs with a valid correlation")

        plot_roi_surface_losolomo_roimean(
            r_maps[layer], atlas_labels_3d, ref_img, fsaverage,
            layer, args.condition, args.prompt_version, out_dir=out_dir,
        )

        for k in range(N_ROI):
            if np.isnan(r_maps[layer][k]):
                continue
            rows.append({
                "source": "losolomo_roimean_anxious",
                "condition": args.condition,
                "prompt_version": args.prompt_version,
                "movie": "all_movies",
                "layer": layer,
                "roi": k + 1,
                "r": r_maps[layer][k],
                "p": p_maps[layer][k],
                "n_subjects": n_maps[layer][k],
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"\nwrote {len(df)} rows -> {csv_out}")
    print("Columns match export_roi_stats_for_r.py's CSV -- source='losolomo_roimean_anxious' "
          "won't collide with correlate_anxiety_losolomo.py's 'losolomo_generalization' or "
          "correlate_anxiety_roi_losolomo.py's 'losolomo_roimean', so all can be concatenated "
          "and reused in roi_anxiety_stats.Rmd as-is. The added condition/prompt_version "
          "columns let you facet or filter by anxiety condition downstream.")

    print("\nDone.")

if __name__ == "__main__":
    main()
