# Authors: Kumiko Ueda (kumiko@uchicago.edu), Alicia Liu
# Last Edited: September 21, 2026

"""
Surface maps of the LOSO x LOMO ROI-mean effect with black outlines around
FDR-significant parcels -- rendered with nilearn/matplotlib (Agg backend).

CONDITION-AWARE VERSION: adds --condition (required) to filter the combined
anxious-conditions significance CSV down to one condition, and routes output
to a per-condition figures subfolder so conditions never overwrite each
other's PNGs.

Input: the significance CSV from the R Markdown, with columns at minimum
    condition, test, layer, roi, <effect-col>, <q-col>

Usage:
    python plot_schaefer_betas_anxious.py --condition trait_strong
    python plot_schaefer_betas_anxious.py --condition base --tests d_zero --layers layer16
    python plot_schaefer_betas_anxious.py --condition base --q-col q --p-col p
"""

import os
import json
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
BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"
CSV_PATH_DEFAULT = "../outputs/roi_significance_anxious_conditions_network.csv"
CONDITIONS_LOC = "../data/conditions.json"
LABEL_DIR = "plot_smoothbrain_sig"    # dir holding the two .annot files

TESTS  = ["d_zero", "d_ceiling", "d_shuffled"]
LAYERS = ["layer12", "layer16", "layer20"]
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_ID   = {net: i + 1 for i, net in enumerate(NETWORKS)}   # 1..7, 0 = background
ALPHA  = 0.05

# annot resolution: your annot read as 40962 vertices = fsaverage5
FSAVG_MESH = "fsaverage6"

def network_map_from_annot(annot_path):
    """Read an annot and return an int array (one entry per vertex) holding the
    network id (1..7, see NET_ID) of that vertex's parcel, 0 for background /
    medial wall."""
    labels, _, names = fsio.read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    out = np.zeros(labels.shape, dtype=int)
    for lab in np.unique(labels):
        if lab < 0 or lab >= len(names):
            continue
        parts = names[lab].split("_")           # "7Networks_LH_Vis_1"
        if len(parts) < 3 or parts[0] != "7Networks":
            continue                             # Background / medial wall
        net = parts[2]
        if net in NET_ID:
            out[labels == lab] = NET_ID[net]
    return out

def vertex_values_from_networks(net_values, net_map):
    """net_values: dict {network name: value}. Returns per-vertex float array,
    NaN wherever the vertex is background or its network has no value."""
    out = np.full(net_map.shape, np.nan, dtype=float)
    for net, nid in NET_ID.items():
        v = net_values.get(net, np.nan)
        if np.isfinite(v):
            out[net_map == nid] = v
    return out

def plot_surface_outlined(net_values, sig_ids, net_map_lh, net_map_rh,
                          fsaverage, cmap, vmax, title, out_path):
    surf_lh = vertex_values_from_networks(net_values, net_map_lh)
    surf_rh = vertex_values_from_networks(net_values, net_map_rh)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(14, 10))
    panels = [
        (fsaverage.infl_left,  surf_lh, net_map_lh, "left",  "lateral", axes[0, 0], "Left lateral"),
        (fsaverage.infl_right, surf_rh, net_map_rh, "right", "lateral", axes[0, 1], "Right lateral"),
        (fsaverage.infl_left,  surf_lh, net_map_lh, "left",  "medial",  axes[1, 0], "Left medial"),
        (fsaverage.infl_right, surf_rh, net_map_rh, "right", "medial",  axes[1, 1], "Right medial"),
    ]
    for mesh, surf, nmap, hemi, view, ax, ttl in panels:
        plotting.plot_surf_stat_map(
            mesh, surf, hemi=hemi, view=view, axes=ax,
            colorbar=False, cmap=cmap, vmin=-vmax, vmax=vmax,
            symmetric_cbar=True, title=ttl,
        )
        present = set(np.unique(nmap).tolist())
        levels = [i for i in sig_ids if i in present]
        if levels:
            plotting.plot_surf_contours(
                mesh, nmap, levels=levels,
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
    with open(CONDITIONS_LOC, "r") as f:
        cons = json.load(f)
    conditions = [k for k in cons]

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=conditions)
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--tests", nargs="+", default=TESTS)
    ap.add_argument("--layers", nargs="+", default=LAYERS)
    ap.add_argument("--csv", default=CSV_PATH_DEFAULT,
                    help="Combined network significance CSV with a 'condition' "
                         "column (from the R Markdown export).")
    ap.add_argument("--effect-col", default="effect")
    ap.add_argument("--q-col", default="q",
                    help="Column holding FDR-corrected p-values (q vs q_ttest "
                         "depending on which Rmd produced your CSV).")
    ap.add_argument("--label-dir", default=LABEL_DIR)
    ap.add_argument("--fig-dir", default=None,
                    help="Default: BASE_RESULTS_DIR/{prompt_version}_condition_{condition}/figures_outlined_network")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    fig_dir = args.fig_dir or os.path.join(
        BASE_RESULTS_DIR, f"{args.prompt_version}_condition_{args.condition}",
        "figures_outlined_network"
    )
    os.makedirs(fig_dir, exist_ok=True)

    # ---- annot -> per-vertex network ids ----
    lh_annot = f"{args.label_dir}/lh.Schaefer2018_200Parcels_7Networks_order.annot"
    rh_annot = f"{args.label_dir}/rh.Schaefer2018_200Parcels_7Networks_order.annot"
    net_map_lh = network_map_from_annot(lh_annot)
    net_map_rh = network_map_from_annot(rh_annot)
    for hemi, m in [("lh", net_map_lh), ("rh", net_map_rh)]:
        counts = {net: int(np.sum(m == nid)) for net, nid in NET_ID.items()}
        print(f"{hemi} vertices per network: {counts}  (background: {int(np.sum(m == 0))})")

    # ---- surface mesh, resolution MUST match the annot ----
    fsaverage = datasets.fetch_surf_fsaverage(FSAVG_MESH)
    coords, _ = surface.load_surf_mesh(fsaverage.infl_left)
    print(f"annot vtx: {len(net_map_lh)}  |  mesh vtx: {coords.shape[0]}  "
          f"({'MATCH' if len(net_map_lh) == coords.shape[0] else 'MISMATCH!'})")
    if len(net_map_lh) != coords.shape[0]:
        raise RuntimeError(
            f"annot ({len(net_map_lh)}) and mesh ({coords.shape[0]}) vertex counts "
            f"differ -- set FSAVG_MESH to the resolution your annot was made on.")

    # ---- CSV ----
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    need = {"test", "layer", "network", args.effect_col, args.q_col}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV missing columns: {missing}. Has: {list(df.columns)}. "
            f"Note this script needs a 'network' column (the network-level "
            f"export), not the ROI-level one. Otherwise pass --effect-col/--q-col."
        )

    if "condition" in df.columns:
        df = df[df["condition"] == args.condition]
        print(f"Filtered to condition={args.condition}: {len(df)} rows")
        if df.empty:
            raise RuntimeError(
                f"No rows for condition={args.condition} in {args.csv}. "
                f"Available conditions: {sorted(pd.read_csv(args.csv)['condition'].unique())}"
            )
    else:
        print(f"  [note] CSV has no 'condition' column -- treating all rows as "
              f"condition={args.condition} (single-condition CSV assumed).")

    unknown = set(df["network"].unique()) - set(NETWORKS)
    if unknown:
        print(f"  [warn] unrecognized network names in CSV (ignored): {sorted(unknown)}")

    for test in args.tests:
        for layer in args.layers:
            sub = df[(df["test"] == test) & (df["layer"] == layer)]
            if sub.empty:
                print(f"  [skip] no rows for {test} / {layer}")
                continue

            net_values = dict(zip(sub["network"], sub[args.effect_col]))

            sig_names = sub.loc[sub[args.q_col] < args.alpha, "network"].tolist()
            sig_ids = [NET_ID[n] for n in sig_names if n in NET_ID]

            vals = np.array([v for v in net_values.values() if np.isfinite(v)])
            vmax = np.max(np.abs(vals)) if vals.size else np.nan
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1e-3

            print(f"\ncondition={args.condition} {test} {layer}: "
                  f"{len(sig_ids)}/{len(net_values)} sig networks "
                  f"({', '.join(sig_names) if sig_names else 'none'}), vmax={vmax:.4f}")

            out_path = os.path.join(
                fig_dir,
                f"{args.prompt_version}_condition_{args.condition}_{test}_{layer}_network_sig-outlined.png"
            )
            plot_surface_outlined(
                net_values, sig_ids, net_map_lh, net_map_rh, fsaverage, CMAP, vmax,
                f"{args.condition} | {test} {layer} network effect with FDR-significant networks outlined",
                out_path,
            )

    print(f"\nDone. Figures in {fig_dir}")

if __name__ == "__main__":
    main()
