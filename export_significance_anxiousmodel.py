# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 6, 2026

"""
Flatten pickles into CSV of per-subject values, for the ANXIOUS-CONDITION
ROI-mean LOSO x LOMO encoding results (8_anxious_llama_encoding_condition_aware.py
output), looped over every condition.

Reads from:
    ../outputs/encoding_results_anxious_conditions/{prompt_version}_condition_{condition}/roi*_losolomo_meanR.pkl

Writes one combined CSV (all conditions stacked, with condition/prompt_version
columns) to:
    ../outputs/roi_mean_per_subject_R_anxious_conditions.csv
"""

import os
import glob
import pickle
import argparse
import pandas as pd

BASE_RESULTS_DIR = "../outputs/encoding_results_anxious_conditions"
# PROMPT_VERSION = "original"
LAYERS = ["layer12", "layer16", "layer20"]

CONDITIONS = [
    "base",
    "control",
    "trait_mild",
    "trait_medium",
    "trait_strong",
    "vignette_1",
    "vignette_2",
    "vignette_3",
    "vignette_4",
    "vignette_5",
    "vignette_6",
    "vignette_7",
]

def results_dir_for(prompt_version, condition, run_tag=None):
    """
    Matches OUT_DIR in 8_anxious_llama_encoding.py.
    """
    d = os.path.join(BASE_RESULTS_DIR, f"{prompt_version}_condition_{condition}")
    if run_tag:
        d = os.path.join(d, run_tag)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out", default="../outputs/roi_mean_per_subject_R_anxious_conditions.csv")
    args = ap.parse_args()

    rows = []
    n_no_diag_by_condition = {}

    for condition in CONDITIONS:
        results_dir = results_dir_for(args.prompt_version, condition, args.run_tag)
        paths = sorted(glob.glob(f"{results_dir}/roi*_losolomo_meanR.pkl"))
        print(f"{condition}: {len(paths)} ROI pkls found in {results_dir}")

        seen_rois = set()
        n_no_diag = 0

        for path in paths:
            with open(path, "rb") as f:
                res = pickle.load(f)
            roi = res["roi"]
            if roi in seen_rois:
                print(f"  [WARNING] duplicate pkl for roi {roi} in {results_dir} "
                      f"(from {path}) -- rows will be double-counted")
            seen_rois.add(roi)
            saved_condition = res.get("condition")
            saved_prompt_version = res.get("prompt_version")
            if saved_condition is not None and saved_condition != condition:
                print(f"  [WARNING] {path}: pkl says condition={saved_condition!r}, "
                      f"but was read from the {condition!r} folder -- skipping")
                continue
            if saved_prompt_version is not None and saved_prompt_version != args.prompt_version:
                print(f"  [WARNING] {path}: pkl says prompt_version={saved_prompt_version!r}, "
                      f"expected {args.prompt_version!r} -- skipping")
                continue
            
            diag = res.get("diagnostics") or {}
            if not diag:
                n_no_diag += 1
            
            for layer in LAYERS:
                R = res["layers"][layer]
                gm = diag.get(layer, {}).get("groupmean", {})
                sh = diag.get(layer, {}).get("shuffled", {})
                for sid, r in R.items():
                    rows.append({
                        "condition": condition,
                        "prompt_version": args.prompt_version,
                        "roi": roi,
                        "layer": layer,
                        "subject_id": sid,
                        "R": r,
                        "ceiling": gm.get(sid, float("nan")),
                        "shuffled": sh.get(sid, float("nan")),
                    })
            
        n_no_diag_by_condition[condition] = (n_no_diag, len(paths))
        if paths and n_no_diag:
            print(f"  [NOTE] {n_no_diag}/{len(paths)} ROI pkls for {condition} have no "
                  f"--diagnostics; their ceiling/shuffled columns are NaN")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo rows collected -- check BASE_RESULTS_DIR, --prompt-version, "
              "and --run-tag against where the encoding jobs actually wrote their pkls.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote empty CSV -> {args.out}")
        return
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(f"Conditions present: {sorted(df['condition'].unique()) if len(df) else '[]'}")
    print("\nDiagnostics coverage by condition (pkls without --diagnostics / total pkls):")
    for condition, (n_no_diag, n_total) in n_no_diag_by_condition.items():
        if n_total:
            print(f"  {condition}: {n_no_diag}/{n_total}")

if __name__ == "__main__":
    main()
