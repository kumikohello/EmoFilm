# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: August 22, 2026

"""
Build a per-subject master CSV from the ROI-mean LOSO x LOMO encoding pkls
(doc17_train_encoding_loso_lomo.py output).

One row per (roi, subject) -- NOT per (movie, roi, subject) like
build_results_csv_roi.py. LOSO x LOMO already pools across every movie
inside the cross-validation loop itself, so there's no separate per-movie
result to keep as its own row; each ROI's pkl already has one pooled R per
subject.

Columns:
    roi, movies (comma-joined, from the pkl's own saved "movies" list),
    subject_id,
    layer12_R, layer16_R, layer20_R,
    ceiling_R, shuffled_R,
    layer12_norm, layer16_norm, layer20_norm

Ceiling/shuffled-null come from the saved diagnostics (present if the
encoding job was run with --diagnostics) -- run_diagnostics() in
doc17_train_encoding_loso_lomo.py already saves them in exactly the shape
build_results_csv_roi.py expects (diag[layer]["groupmean"/"shuffled"]), so
they're read directly rather than recomputed.

If a pkl has no diagnostics, the ceiling is recomputed from the raw
ROI-mean responses, pooled across movies the SAME way the encoding run
itself pooled them (see load_pooled_roimean_responses). This doesn't
require redoing any ridge fitting -- the ceiling is a model-free
leave-one-subject-out group-mean correlation.

norm = layer_R / ceiling_R -- fraction of the achievable shared signal the
model captured, comparable across ROIs.

Usage:
    python build_results_csv_roi_losolomo_roimean.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.append("../scripts")
from utils import load_group_roi_responses, load_design, column_corr

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
LOSO_LOMO_DIR = "../outputs/encoding_results_roi_mean_losolomo_trimmed"
ROI_DIR       = "../data/roi_data_trimmed"
SUBJECT_DIR   = "../data/subjects"
LAYERS        = ["layer12", "layer16", "layer20"]
GROUP         = "all"
N_ROI         = 200
OUT_CSV       = "../outputs/encoding_results_roi_mean_losolomo_trimmed/master_per_subject_roi_losolomo_roimean.csv"

# Ceilings below this are too small to divide by reliably; norm is set to NaN.
CEILING_FLOOR = 0.01

def find_roi_pkl(roi):
    """Flat lookup only -- matches doc17_train_encoding_loso_lomo.py's
    no-run-tag output layout."""
    p = os.path.join(LOSO_LOMO_DIR, f"roi{roi:03d}_losolomo_meanR.pkl")
    return p if os.path.exists(p) else None

def load_pooled_roimean_responses(roi, movies):
    """
    Rebuild the same pooled, ROI-mean, cross-movie response vectors the
    encoding run itself used: {sid: (total_trs_across_movies, 1)}. Movie
    order matches the pkl's own saved `movies` list, same pooling order
    doc17_train_encoding_loso_lomo.py used when it fit the model.
    """
    per_movie = {}
    subj_sets = []
    for movie in movies:
        n_trs = load_design(f"../outputs/{movie}_trimmed", LAYERS[0]).shape[0]
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

def get_ceiling_and_shuffled(res, roi):
    """Pull ceiling + shuffled-X from the saved diagnostics if present,
    same convention as build_results_csv_roi.py."""
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
    pooled = load_pooled_roimean_responses(roi, res["movies"])
    ceiling = compute_ceiling(pooled)
    shuffled = {sid: np.nan for sid in ceiling}
    return ceiling, shuffled

def main():
    print("starting script")
    rows = []
    n_found = 0

    for roi in range(1, N_ROI + 1):
        path = find_roi_pkl(roi)
        if path is None:
            continue
        n_found += 1

        with open(path, "rb") as f:
            res = pickle.load(f)

        ceiling, shuffled = get_ceiling_and_shuffled(res, roi)

        layer_dicts = {L: res["layers"][L] for L in LAYERS}
        sids = sorted(set(layer_dicts[LAYERS[0]]) & set(ceiling))
        movies_str = ",".join(res.get("movies", []))

        for sid in sids:
            row = {
                "roi": roi,
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
        raise RuntimeError("No result pkls found; nothing to write.")

    df = pd.DataFrame(rows)

    # normalized R: fraction of the achievable (ceiling) signal captured.
    for L in LAYERS:
        norm = df[f"{L}_R"] / df["ceiling_R"]
        norm[df["ceiling_R"] <= CEILING_FLOOR] = np.nan
        df[f"{L}_norm"] = norm

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nSummary (mean across all ROIs and subjects):")
    parts = [f"ceiling={df['ceiling_R'].mean():.4f}",
             f"shuffled={df['shuffled_R'].mean():.4f}"]
    for L in LAYERS:
        parts.append(f"{L}: R={df[f'{L}_R'].mean():.4f} norm={df[f'{L}_norm'].mean():.3f}")
    print("  " + " | ".join(parts))

    print("\nTop 10 ROIs by layer16_norm (model / ceiling), averaged over subjects:")
    roi_norm = (df.groupby("roi")[["layer16_norm", "layer16_R", "ceiling_R"]]
                  .mean().reset_index())
    top = roi_norm.sort_values("layer16_norm", ascending=False).head(10)
    print(top.to_string(index=False))

    print("\nSample:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()