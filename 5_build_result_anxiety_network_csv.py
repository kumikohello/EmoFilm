# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 9, 2026

"""
Build a per-subject master CSV from the ROI-mean LOSO x LOMO encoding pkls.

CONDITION-AWARE VERSION: adds --condition and --prompt-version (and optional
--run-tag) so this reads from
    ../outputs/encoding_results_anxious_conditions/{prompt_version}_condition_{condition}/[run_tag/]
(matching the encoding script's OUT_DIR layout) instead of the old flat,
single-condition ../outputs/encoding_results_roi_mean_losolomo_trimmed/.

One row per (network, subject). LOSO x LOMO
already pools across every movie inside the cross-validation loop itself,
so there's no separate per-movie result to keep as its own row; each ROI's
pkl already has one pooled R per subject.

Columns:
    roi, movies (comma-joined, from the pkl's own saved "movies" list),
    subject_id,
    layer12_R, layer16_R, layer20_R,
    ceiling_R, shuffled_R,
    layer12_norm, layer16_norm, layer20_norm

Ceiling/shuffled-null come from the saved diagnostics (present if the
encoding job was run with --diagnostics) -- run_diagnostics() already saves
them in the shape this script expects (diag[layer]["groupmean"/"shuffled"]),
so they're read directly rather than recomputed.

If a pkl has no diagnostics, the ceiling is recomputed from the raw
ROI-mean responses, pooled across movies the SAME way the encoding run
itself pooled them (see load_pooled_roimean_responses), using the same
condition-specific design directories. This doesn't require redoing any
ridge fitting -- the ceiling is a model-free leave-one-subject-out
group-mean correlation.

norm = layer_R / ceiling_R -- fraction of the achievable shared signal the
model captured, comparable across ROIs.

Usage:
    5_build_result_anxiety_csv.py [-h] --condition
        {base,control,trait_mild,trait_medium,trait_strong,vignette_1,vignette_2,vignette_3,vignette_4,vignette_5,vignette_6,vignette_7}
    python build_results_csv_roi_losolomo_roimean_anxious.py --condition base --prompt-version original
"""

import os
import sys
import json
import argparse
import pickle

import numpy as np
import pandas as pd

sys.path.append("../scripts")
from utils import load_group_roi_responses, load_design, column_corr

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHAEFER_LUT = "../../resources/atlases/Schaefer200/Schaefer2018_200Parcels_7Networks_order.txt"
BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"
NETWORK_RESULTS_DIR = "../outputs/encoding_results_anxious_network"
ROI_DIR       = "../data/roi_data_trimmed"
SUBJECT_DIR   = "../data/subjects"
CONDITIONS_LOC = "../data/conditions.json"
LAYERS        = ["layer12", "layer16", "layer20"]
GROUP         = "all"
N_ROI         = 200

# Ceilings below this are too small to divide by reliably; norm is set to NaN.
CEILING_FLOOR = 0.01

def design_dir_for(prompt_version, condition, movie):
    """Matches the output layout of 3_lanczos_and_FIR_anxious.py and
    8_anxious_llama_encoding_condition_aware.py."""
    return f"../outputs/{prompt_version}_condition_{condition}_{movie}_trimmed"

def results_dir_for(prompt_version, condition, run_tag=None):
    """Matches OUT_DIR in 8_anxious_llama_encoding_condition_aware.py."""
    d = os.path.join(BASE_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def network_results_dir_for(prompt_version, condition, run_tag=None):
    """Separate tree for the network-level CSV (and downstream figures)."""
    d = os.path.join(NETWORK_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def find_roi_pkl(roi, results_dir):
    p = os.path.join(results_dir, f"roi{roi:03d}_losolomo_meanR.pkl")
    return p if os.path.exists(p) else None

def load_roi_to_network(lut_path=SCHAEFER_LUT):
    """
    Parses lines like:
        1  7Networks_LH_VIS_1  120 18 134 0
    roi_id -> network name (e.g. 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn',
                                'Limbic', 'Cont', 'Default')
    """
    roi_to_net = {}
    with open(lut_path) as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            roi_id = int(parts[0])
            label = parts[1]
            network = label.split("_")[2]
            roi_to_net[roi_id] = network
    return roi_to_net

def fisherz_mean(r):
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    z = np.arctanh(r)
    return np.tanh(np.nanmean(z))

def load_pooled_roimean_responses(roi, movies, prompt_version, condition):
    """
    Rebuild the same pooled, ROI-mean, cross-movie response vectors the
    encoding run itself used: {sid: (total_trs_across_movies, 1)}. Movie
    order matches the pkl's own saved `movies` list, same pooling order
    the encoding script used when it fit the model. Design dirs are
    condition-specific, matching design_dir_for() in the encoding script.
    """
    per_movie = {}
    subj_sets = []
    for movie in movies:
        design_dir = design_dir_for(prompt_version, condition, movie)
        n_trs = load_design(design_dir, LAYERS[0]).shape[0]
        responses, _ = load_group_roi_responses(
            ROI_DIR, SUBJECT_DIR, GROUP, roi, movie, n_trs
        )
        responses = {sid: np.nanmean(r, axis=1, keepdims=True) for sid, r in responses.items()}
        per_movie[movie] = responses
        subj_sets.append(set(responses.keys()))

    subjects = sorted(set.intersection(*subj_sets)) if subj_sets else []
    pooled = {
        sid: np.concatenate([per_movie[m][sid] for m in movies], axis=0)
        for sid in subjects
    }
    return pooled

def compute_ceiling(responses):
    """Fallback: inter-subject leave-one-out ceiling: {sid: R}.
    Only used if a pkl was written without --diagnostics."""
    sids = list(responses.keys())
    ceiling = {}
    for out_sid in sids:
        others = [responses[s] for s in sids if s != out_sid]
        mean_resp = np.mean(others, axis=0)
        ceiling[out_sid] = float(np.nanmean(column_corr(mean_resp, responses[out_sid])))
    return ceiling

def get_ceiling_and_shuffled(res, roi, prompt_version, condition):
    """Pull ceiling + shuffled-X from the saved diagnostics if present."""
    diag = res.get("diagnostics") or {}

    if diag and LAYERS[0] in diag:
        ceiling = diag[LAYERS[0]]["groupmean"]
        sids = list(ceiling.keys())
        shuffled = {}
        for sid in sids:
            vals = [diag[L]["shuffled"].get(sid, np.nan) for L in LAYERS if L in diag]
            shuffled[sid] = float(np.nanmean(vals)) if vals else np.nan
        return ceiling, shuffled

    print(f"    (no diagnostics in roi {roi}; recomputing ceiling from pooled ROI-mean data)")
    pooled = load_pooled_roimean_responses(roi, res["movies"], prompt_version, condition)
    ceiling = compute_ceiling(pooled)
    shuffled = {sid: np.nan for sid in ceiling}
    return ceiling, shuffled

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
    out_csv = os.path.join(
        results_dir,
        f"master_per_subject_roi_losolomo_roimean_{args.prompt_version}_condition_{args.condition}.csv",
    )

    print(f"starting script: condition={args.condition}, prompt_version={args.prompt_version}")
    print(f"reading from: {results_dir}")

    rows = []
    n_found = 0

    for roi in range(1, N_ROI + 1):
        path = find_roi_pkl(roi, results_dir)
        if path is None:
            continue
        n_found += 1

        with open(path, "rb") as f:
            res = pickle.load(f)

        ceiling, shuffled = get_ceiling_and_shuffled(res, roi, args.prompt_version, args.condition)

        layer_dicts = {L: res["layers"][L] for L in LAYERS}
        sids = sorted(set(layer_dicts[LAYERS[0]]) & set(ceiling))
        movies_str = ",".join(res.get("movies", []))

        for sid in sids:
            row = {
                "roi": roi,
                "condition": args.condition,
                "prompt_version": args.prompt_version,
                "movies": movies_str,
                "subject_id": sid,
                "ceiling_R": ceiling[sid],
                "shuffled_R": shuffled.get(sid, np.nan),
            }
            for L in LAYERS:
                row[f"{L}_R"] = layer_dicts[L].get(sid, np.nan)
            rows.append(row)

        if roi % 50 == 0:
            print(f"  ... roi {roi}")

    print(f"\n{n_found} / {N_ROI} ROIs found")

    if len(rows) == 0:
        raise RuntimeError(f"No result pkls found in {results_dir}; nothing to write. "
                            f"Did the encoding job for condition={args.condition} finish?")

    df = pd.DataFrame(rows)

    roi_to_net = load_roi_to_network()
    df["network"] = df["roi"].map(roi_to_net)

    n_unmapped = df["network"].isna().sum()
    if n_unmapped:
        print(f"WARNING: {n_unmapped} rows have no network label "
            f"(roi ids not found in LUT) -- dropping them from the network CSV.")

    agg_cols = ["ceiling_R", "shuffled_R"] + [f"{L}_R" for L in LAYERS]
    net_df = (df.dropna(subset=["network"])
                .groupby(["network", "subject_id", "condition", "prompt_version"])[agg_cols]
                .agg(fisherz_mean)
                .reset_index())

    for L in LAYERS:
        norm = net_df[f"{L}_R"] / net_df["ceiling_R"]
        norm[net_df["ceiling_R"] <= CEILING_FLOOR] = np.nan
        net_df[f"{L}_norm"] = norm
    
    net_dir = network_results_dir_for(args.prompt_version, args.condition, args.run_tag)
    os.makedirs(net_dir, exist_ok=True)
    net_out_csv = os.path.join(
        net_dir,
        f"masters_per_subject_network7_losolomo_roimean_{args.prompt_version}_condition_{args.condition}.csv",
    )
    net_df.to_csv(net_out_csv, index=False)
    print(f"Wrote {len(net_df)} rows -> {net_out_csv}")

    print("\nNetowrk means (across subjects):")
    print(net_df.groupby("network")[[f"{L}_norm" for L in LAYERS]].mean()
                .sort_values("layer16_norm", ascending=False)
                .to_string())

if __name__ == "__main__":
    main()
