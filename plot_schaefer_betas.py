# Authors: Kumiko Ueda (kumiko@uchicago.edu), Alicia Liu
# Last Edited: September 2, 2026

"""
Surface maps of the LOSO x LOMO ROI-mean effect with black outlines around
FDR-significant parcels -- rendered with nilearn/matplotlib (Agg backend).

Input: the significance CSV from the R Markdown, with columns
    test, layer, roi, effect, q_ttest

Usage:
    python plot_schaefer_betas.py --tests d_ceiling --layers layer16
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")                       # headless
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from nilearn import datasets, plotting, surface
import nibabel.freesurfer.io as fsio

# optional nicer colormap; fall back to coolwarm if cmcrameri isn't installed
try:
    from cmcrameri import cm as ccm
    CMAP = ccm.vik
except Exception:
    CMAP = plt.get_cmap("coolwarm")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CSV_PATH  = "../outputs/roi_mean_significance_trimmed.csv"
LABEL_DIR = "plot_smoothbrain_sig"    # dir holding the two .annot files
FIG_DIR   = "../outputs/encoding_results_roi_mean_losolomo/figures_outlined_trimmed"

TESTS  = ["d_zero", "d_ceiling", "d_shuffled"]
LAYERS = ["layer12", "layer16", "layer20"]
N_ROI  = 200
ALPHA  = 0.05

# annot resolution: your annot read as 40962 vertices = fsaverage5
FSAVG_MESH = "fsaverage6"

def vertex_values_from_annot(values, parc, offset=0):
    """Map length-N_ROI values (1-based ROI ids) onto vertices via annot labels.
    parc holds local labels 1..100; offset shifts to the global roi id (0 for
    LH, 100 for RH)."""
    out = np.full(parc.shape, np.nan, dtype=float)
    for lab in np.unique(parc):
        if lab <= 0:
            continue
        out[parc == lab] = values[lab - 1 + offset]
    return out

def plot_surface_outlined(values, sig_lh, sig_rh, parc_lh, parc_rh,
                          fsaverage, cmap, vmax, title, out_path):
    surf_lh = vertex_values_from_annot(values, parc_lh, offset=0)
    surf_rh = vertex_values_from_annot(values, parc_rh, offset=100)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(14, 10))
    panels = [
        (fsaverage.infl_left,  surf_lh, parc_lh, sig_lh, "left",  "lateral", axes[0, 0], "Left lateral"),
        (fsaverage.infl_right, surf_rh, parc_rh, sig_rh, "right", "lateral", axes[0, 1], "Right lateral"),
        (fsaverage.infl_left,  surf_lh, parc_lh, sig_lh, "left",  "medial",  axes[1, 0], "Left medial"),
        (fsaverage.infl_right, surf_rh, parc_rh, sig_rh, "right", "medial",  axes[1, 1], "Right medial"),
    ]
    for mesh, surf, parc, sig, hemi, view, ax, ttl in panels:
        plotting.plot_surf_stat_map(
            mesh, surf, hemi=hemi, view=view, axes=ax,
            colorbar=False, cmap=cmap, vmin=-vmax, vmax=vmax,
            symmetric_cbar=True, title=ttl,
        )
        present = set(np.unique(parc).tolist())
        levels = [i for i in sig if i in present]
        if levels:
            plotting.plot_surf_contours(
                mesh, parc, levels=levels,
                colors=["k"] * len(levels), linewidths=2.0,
                axes=ax, figure=fig,
            )

    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, cax=fig.add_axes([0.92, 0.35, 0.015, 0.30]))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {os.path.basename(out_path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=TESTS)
    ap.add_argument("--layers", nargs="+", default=LAYERS)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--label-dir", default=LABEL_DIR)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- annot (defines the parcel-to-vertex mapping) ----
    lab_lh, _, _ = fsio.read_annot(
        f"{args.label_dir}/lh.Schaefer2018_200Parcels_7Networks_order.annot")
    lab_rh, _, _ = fsio.read_annot(
        f"{args.label_dir}/rh.Schaefer2018_200Parcels_7Networks_order.annot")
    parc_lh, parc_rh = lab_lh.astype(int), lab_rh.astype(int)

    # ---- surface mesh, resolution MUST match the annot ----
    fsaverage = datasets.fetch_surf_fsaverage(FSAVG_MESH)
    coords, _ = surface.load_surf_mesh(fsaverage.infl_left)
    print(f"annot vtx: {len(lab_lh)}  |  mesh vtx: {coords.shape[0]}  "
          f"({'MATCH' if len(lab_lh) == coords.shape[0] else 'MISMATCH!'})")
    if len(lab_lh) != coords.shape[0]:
        raise RuntimeError(
            f"annot ({len(lab_lh)}) and mesh ({coords.shape[0]}) vertex counts "
            f"differ -- set FSAVG_MESH to the resolution your annot was made on.")

    # ---- CSV ----
    print("Loading CSV ...")
    df = pd.read_csv(args.csv)
    need = {"test", "layer", "roi", "effect", "q_ttest"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}. Has: {list(df.columns)}")

    for test in args.tests:
        for layer in args.layers:
            sub = df[(df["test"] == test) & (df["layer"] == layer)]
            if sub.empty:
                print(f"  [skip] no rows for {test} / {layer}")
                continue

            wb = np.full(N_ROI, np.nan)
            wb[sub["roi"].values.astype(int) - 1] = sub["effect"].values

            sig_ids = sub.loc[sub["q_ttest"] < args.alpha, "roi"].astype(int).tolist()
            sig_lh = [i for i in sig_ids if i <= 100]
            sig_rh = [i - 100 for i in sig_ids if i > 100]

            vmax = np.nanmax(np.abs(wb))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1e-3

            print(f"\n{test} {layer}: {len(sig_ids)} sig ROIs "
                  f"(LH {len(sig_lh)}, RH {len(sig_rh)}), vmax={vmax:.4f}")

            out_path = os.path.join(args.fig_dir, f"{test}_{layer}_sig-outlined.png")
            plot_surface_outlined(
                wb, sig_lh, sig_rh, parc_lh, parc_rh, fsaverage, CMAP, vmax,
                f"{test} {layer} effect with FDR-significant ROIs outlined",
                out_path,
            )

    print(f"\nDone. Figures in {args.fig_dir}")

if __name__ == "__main__":
    main()