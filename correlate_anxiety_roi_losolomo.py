# Authors: Kumiko Ueda (kumiko@uchicago.edu), Andrea Liu
# Last Edited: August 10, 2026

"""
Correlate LEAVE-ONE-MOVIE-OUT x LEAVE-ONE-SUBJECT-OUT, ROI-MEAN encoding
generalization (from train_encoding_roi_losolomo.py) with DASS_anx anxiety
score, per ROI, per layer.

This is the ROI-mean-only counterpart to correlate_anxiety_losolomo.py.

Expected input: ../outputs/encoding_results_roi/loso_lomo/roi{k:03d}_losolomo_meanR.pkl.

Each subject's mean_R already reflects genuine cross-movie generalization:
the model was fit with that subject AND an entire movie held out
simultaneously, so this is NOT the same as correlate_anxiety.py's
avg_then_corr/corr_then_avg (post-hoc averaging of independently-fit,
single-movie LOSO results).

Because doc17's training script only ever produces ROI-level results, this
script only does ROI surface plots -- no voxel-level/whole-brain option.

CSV output uses source="losolomo_roimean" (not "losolomo_generalization",
which is what correlate_anxiety_losolomo.py's CSV uses) so the two can be
concatenated without collision if you ever want to compare a toggle-version
run against a doc17 run in the same Rmd.

Usage:
    python correlate_anxiety_roi_losolomo.py
"""

import os
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
LOSO_LOMO_DIR = "../outputs/encoding_results_roi_mean_losolomo_trimmed"   # BASE_OUT_DIR in train_encoding_roi_losolomo.py
OUT_DIR       = "../outputs/anxiety_correlation_maps_losolomo_roi_trimmed"
CSV_OUT       = "../outputs/roi_anxiety_stats_losolomo_roi_trimmed.csv"

LAYERS = ["layer12", "layer16", "layer20"]
MIN_SUBJECTS_PER_ROI = 5

def find_roi_pkl_roimean(roi):
    """
    Locate the pkl for this ROI, produced by train_encoding_roi_losolomo.py.

    Returns the first existing path, or None if neither layout has it.
    """
    dirs = []
    dirs.append(LOSO_LOMO_DIR)

    name = f"roi{roi:03d}_losolomo_meanR.pkl"

    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def roiwise_correlation_from_losolomo_roimean(anxiety):
    """
    Reads every ROI pkl found via find_roi_pkl_roimean(), and correlates each
    ROI's per-subject mean_R (already pooled across movies by the LOSO x
    LOMO fit itself) against DASS_anx.

    Returns:
        r_maps, p_maps, n_maps -- each {layer: (N_ROI,) array}
        missing_rois -- list of ROI ids with no pkl found under either layout
    """
    r_maps = {layer: np.full(N_ROI, np.nan, dtype=np.float64) for layer in LAYERS}
    p_maps = {layer: np.full(N_ROI, np.nan, dtype=np.float64) for layer in LAYERS}
    n_maps = {layer: np.zeros(N_ROI, dtype=np.int64) for layer in LAYERS}
    missing_rois = []
    reported_format = False

    for k in range(1, N_ROI + 1):
        path = find_roi_pkl_roimean(k)
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
        print(f"  {len(missing_rois)} / {N_ROI} ROI pkls not found"
              f"{missing_rois[:10]}{' ...' if len(missing_rois) > 10 else ''}")

    return r_maps, p_maps, n_maps, missing_rois

def plot_roi_surface_losolomo_roimean(roi_corr, atlas_labels_3d, ref_img, fsaverage, layer, out_dir):
    """Same painting/projection logic as correlate_anxiety.plot_roi_surface,
    labeled for this ROI-mean LOSO x LOMO source."""
    os.makedirs(out_dir, exist_ok=True)
    img = roi_values_to_img(roi_corr, atlas_labels_3d, ref_img.affine)

    nii_path = os.path.join(out_dir, f"losolomo_roimean_{layer}_anxiety_corr_ROI.nii.gz")
    nib.save(img, nii_path)

    texture_l = surface.vol_to_surf(img, fsaverage.pial_left)
    texture_r = surface.vol_to_surf(img, fsaverage.pial_right)

    _plot_hemis(texture_l, texture_r, fsaverage, "losolomo_roimean", f"{layer}_ROI",
                suffix="", cmap="cold_hot", title_suffix=" (LOSO x LOMO, ROI-mean)", out_dir=out_dir)

    print(f"      saved -> {nii_path}")

def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

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

    print(f"\n=== LOSO x LOMO generalization (ROI-mean only) ===")
    r_maps, p_maps, n_maps, missing_rois = roiwise_correlation_from_losolomo_roimean(
        anxiety
    )

    rows = []
    for layer in LAYERS:
        n_valid = int(np.sum(~np.isnan(r_maps[layer])))
        print(f"  {layer}: {n_valid} / {N_ROI} ROIs with a valid correlation")

        plot_roi_surface_losolomo_roimean(r_maps[layer], atlas_labels_3d, ref_img, fsaverage,
                                           layer, out_dir=OUT_DIR)

        for k in range(N_ROI):
            if np.isnan(r_maps[layer][k]):
                continue
            rows.append({
                "source": "losolomo_roimean",
                "movie": "all_movies",
                "layer": layer,
                "roi": k + 1,
                "r": r_maps[layer][k],
                "p": p_maps[layer][k],
                "n_subjects": n_maps[layer][k],
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nwrote {len(df)} rows -> {CSV_OUT}")
    print("Columns match export_roi_stats_for_r.py's CSV -- source='losolomo_roimean' "
          "won't collide with correlate_anxiety_losolomo.py's 'losolomo_generalization', "
          "so you can concatenate both CSVs and reuse roi_anxiety_stats.Rmd as-is.")

    print("\nDone.")

if __name__ == "__main__":
    main()
