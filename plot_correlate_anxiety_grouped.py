# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 6, 2026

"""
Correlate LEAVE-ONE-MOVIE-OUT x LEAVE-ONE-SUBJECT-OUT, ROI-MEAN encoding
generalization with DASS_anx anxiety score, per ROI, per layer.

GROUPED-SURFACE VERSION of correlate_anxiety_roi_losolomo_anxious.py.

This script is IDENTICAL to correlate_anxiety_roi_losolomo_anxious.py in every
respect -- it imports and reuses that script's own results-loading and
per-ROI Spearman-correlation logic (results_dir_for,
roiwise_correlation_from_losolomo_roimean), and writes the exact same CSV
output (same columns, same source="losolomo_roimean_anxious" tag) -- EXCEPT
for how the surface maps are rendered.

correlate_anxiety_roi_losolomo_anxious.py calls correlate_anxiety._plot_hemis,
which saves each hemisphere/view as its own separate image. This script
instead paints all four views (left lateral, right lateral, left medial,
right medial) into a single combined 2x2 matplotlib figure -- the same
"four hemispheres grouped together" layout used by
plot_schaefer_betas_anxious.py's plot_surface_outlined (minus the
FDR-significance outlining, since this script has no significance CSV to
outline -- it plots the raw correlation map only).

Usage:
    python correlate_anxiety_roi_losolomo_anxious_grouped.py --condition trait_strong
    python correlate_anxiety_roi_losolomo_anxious_grouped.py --condition base --prompt-version original
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib as mpl
mpl.use("Agg")                       # headless
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from nilearn import datasets, plotting, surface

from correlate_anxiety import (
    REFERENCE_MASK, IN_TSV, ATLAS_PATH, N_ROI, MESH,
    load_anxiety_scores, load_atlas, roi_values_to_img,
)
from correlate_anxiety_anxiousmodel import (
    BASE_OUT_DIR, BASE_CSV_DIR, CONDITIONS_LOC, LAYERS,
    results_dir_for, roiwise_correlation_from_losolomo_roimean,
)

# nicer diverging colormap if nilearn's "cold_hot" is registered with
# matplotlib; otherwise fall back to coolwarm (same fallback pattern as
# plot_schaefer_betas_anxious.py)
try:
    CMAP = plt.get_cmap("cold_hot")
except (ValueError, KeyError):
    CMAP = plt.get_cmap("coolwarm")

def plot_roi_surface_losolomo_roimean_grouped(roi_corr, atlas_labels_3d, ref_img, fsaverage,
                                               layer, condition, prompt_version, out_dir):
    """Same volume->surface painting/projection as
    correlate_anxiety_roi_losolomo_anxious.plot_roi_surface_losolomo_roimean
    (roi_values_to_img + surface.vol_to_surf onto fsaverage pial), but instead
    of calling correlate_anxiety._plot_hemis (one saved image per hemisphere
    x view), this draws all four views into ONE combined 2x2 figure -- the
    grouped-hemisphere layout from plot_schaefer_betas_anxious.py."""
    os.makedirs(out_dir, exist_ok=True)
    img = roi_values_to_img(roi_corr, atlas_labels_3d, ref_img.affine)

    prefix = f"losolomo_roimean_{prompt_version}_condition_{condition}_{layer}"
    nii_path = os.path.join(out_dir, f"{prefix}_anxiety_corr_ROI.nii.gz")
    nib.save(img, nii_path)

    texture_l = surface.vol_to_surf(img, fsaverage.pial_left)
    texture_r = surface.vol_to_surf(img, fsaverage.pial_right)

    finite_vals = np.concatenate([
        texture_l[np.isfinite(texture_l)],
        texture_r[np.isfinite(texture_r)],
    ])
    vmax = np.nanmax(np.abs(finite_vals)) if finite_vals.size else 1e-3
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-3

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(14, 10))
    panels = [
        (fsaverage.infl_left,  texture_l, "left",  "lateral", axes[0, 0], "Left lateral"),
        (fsaverage.infl_right, texture_r, "right", "lateral", axes[0, 1], "Right lateral"),
        (fsaverage.infl_left,  texture_l, "left",  "medial",  axes[1, 0], "Left medial"),
        (fsaverage.infl_right, texture_r, "right", "medial",  axes[1, 1], "Right medial"),
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
    fig.suptitle(
        f"{layer}_ROI anxiety corr (LOSO x LOMO, ROI-mean, {condition})",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"{prefix}_anxiety_corr_ROI_grouped.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"      saved -> {nii_path}")
    print(f"      saved -> {png_path}")

def main():
    import json
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

    print(f"=== condition={args.condition}, prompt_version={args.prompt_version} (GROUPED plots) ===")
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

        plot_roi_surface_losolomo_roimean_grouped(
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
    print("Columns match correlate_anxiety_roi_losolomo_anxious.py's CSV exactly -- "
          "source='losolomo_roimean_anxious', same schema -- only the surface PNGs "
          "differ (one combined 4-panel figure per layer instead of separate "
          "per-view images).")

    print("\nDone.")

if __name__ == "__main__":
    main()
