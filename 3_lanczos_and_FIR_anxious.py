# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: September 21, 2026

"""
Lanczos resampling + FIR delay stacking for the ANXIOUS-CONDITION PCA-reduced
scene embeddings (from 8_anxious_llama_embeddings.py), one movie x condition x
layer at a time.

Pipeline:
    1. Load (n_scenes, n_comp) PCA-reduced scene embeddings for one (prompt_version,
       condition, movie, layer).
    2. Read start_time_seconds / end_time_seconds from {movie}_annot.csv (same
       timing source as 3_lanczsco_and_FIR.py) and take the per-scene center in
       seconds. Do NOT use start_TR / end_TR from the summaries CSV -- those
       columns assume 1.75 s/unit and are not real BOLD TR indices (BOLD TR=1.3 s).
    3. Resample onto the TR grid np.arange(n_tr) * TR (seconds) with a 3-lobe
       Lanczos kernel. Output: (n_tr, n_comp).
    4. Z-score each feature column across time.
    5. FIR-stack with delays [1, 2, 3, 4]. Output: (n_tr, n_comp * 4).
    6. Save to {out_dir}/layer{L}_pca{n_comp}_lanczsco_FIR.npy.

N_TR is read from trim_qc_log.csv so responses are always truncated, never
zero-padded, downstream in load_group_roi_responses.
"""

import os
import argparse
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EMBED_ROOT = "../outputs/embeddings"
CSV_DIR = "../data/csv"
QC_LOG = "../data/bold_runY_trimmed/trim_qc_log.csv"
LANCZOS_A = 3
FIR_DELAYS = [1, 2, 3, 4]
LAYERS = [12, 16, 20]
TR = 1.3  # BOLD TR in seconds (must match 3_lanczsco_and_FIR.py)

QC_FILM_NAME = {
    "between_viewings": "BetweenViewings",
    "big_buck_bunny": "BigBuckBunny",
    "chatter": "Chatter",
    "the_secret_number": "TheSecretNumber",
    "sintel": "Sintel",
    "payload": "Payload",
    "first_bite": "FirstBite",
    "you_again": "YouAgain",
    "after_the_rain": "AfterTheRain",
    "lesson_learned": "LessonLearned",
    "spaceman": "Spaceman",
    "to_claire_from_sonny": "ToClaireFromSonny",
    "superhero": "Superhero",
    "tears_of_steel": "TearsOfSteel",
}

# -----------------------------------------------------------------------------
# Lanczos kernel (identical to word-level script)
# -----------------------------------------------------------------------------
def lanczos_kernel(x ,a=3):
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)

def lanczos_resample(embeddings, annot_times, target_times, spacing, a=3):
    dist = (target_times[:, None] - annot_times[None, :]) / spacing
    weights = lanczos_kernel(dist, a=a)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = weights / row_sums
    return weights @ embeddings

def delay_shift(X, k):
    out = np.zeros_like(X)
    if k > 0:
        out[k:] = X[:-k]
    elif k == 0:
        out[:] = X
    return out

def build_fir_design(X, delays):
    return np.concatenate([delay_shift(X, k) for k in delays], axis=1)

def get_canonical_n_tr(qc_log_path, movie_key):
    df = pd.read_csv(qc_log_path)
    film_name = QC_FILM_NAME.get(movie_key, movie_key)
    rows = df[df["film"] == film_name]
    if len(rows) == 0:
        raise ValueError(f"No QC log rows found for film '{film_name}' "
                         f"(movie key '{movie_key}'). Check QC_FILM_NAME mapping.")
    n_tr = int(rows["n_trs_used"].min())
    n_tr_max = int(rows["n_trs_used"].max())
    if n_tr != n_tr_max:
        print(f"  [NOTE] {movie_key}: n_trs_used ranges {n_tr}-{n_tr_max} across "
              f"{len(rows)} subjects; using min ({n_tr}) as the canonical design length.")
    return n_tr

# -----------------------------------------------------------------------------
# Main per-layer processing
# -----------------------------------------------------------------------------
def process_layer(L, prompt_version, condition, movie, annot_times, target_times,
                  n_tr, n_scenes, spacing, embed_dir, out_dir):
    in_path = os.path.join(
        embed_dir,
        f"{prompt_version}_condition_{condition}_{movie}_layer{L}_pca32.npy",
    )
    print(f"=== {movie} | condition={condition} | layer{L} ===")

    if not os.path.exists(in_path):
        print(f"[skip] missing embeddings: {in_path}\n")
        return
    
    emb = np.load(in_path)
    n_comp = emb.shape[1]
    print(f"  loaded: {emb.shape} (expected ({n_scenes}, n_comp))")
    if emb.shape[0] != n_scenes:
        print(f" [skip] layer {L}: embedding rows {emb.shape[0]} != scenes {n_scenes}\n")
        return
    
    out_path = os.path.join(out_dir, f"layer{L}_pca{n_comp}_lanczsco_FIR.npy")

    # Step 1: Lanczos resample (n_scenes, n_comp) -> (n_tr, n_comp), seconds throughout
    resampled = lanczos_resample(emb, annot_times, target_times, spacing=spacing, a=LANCZOS_A)
    print(f"  resampled: {resampled.shape} (expected ({n_tr}, {n_comp}))")
    assert resampled.shape == (n_tr, n_comp)

    # Step 2: z-score each feature column across time
    mean = resampled.mean(axis=0, keepdims=True)
    std = resampled.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    zscored = (resampled - mean) / std
    print(f"  z-scored: {zscored.shape}")

    # Step 3: FIR stack
    fir = build_fir_design(zscored, FIR_DELAYS)
    n_fir = n_comp * len(FIR_DELAYS)
    print(f"  FIR design: {fir.shape} (expected ({n_tr}, {n_fir}))")
    assert fir.shape == (n_tr, n_fir)

    np.save(out_path, fir)
    print(f"  saved to: {out_path}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True,
                    help="movie key watching {movie}_with_summaries_gpt4o.csv, "
                        "e.g. between_viewings")
    ap.add_argument("--condition", required=True,
                    help="condition key, e.g. base, control, trait_mild, vignette_3")
    ap.add_argument("--prompt-version", default="original")
    ap.add_argument("--out-dir", default=None,
                    help="Default: ../outputs/{prompt_version}_condition_{condition}_{movie}_trimmed")
    ap.add_argument("--qc-log", default=QC_LOG)
    args = ap.parse_args()

    movie = args.movie
    embed_dir = os.path.join(EMBED_ROOT, args.prompt_version)
    out_dir = args.out_dir or (
        f"../outputs/{args.prompt_version}_condition_{args.condition}_{movie}_trimmed"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Timing comes from the annot CSV (seconds), not the summaries CSV's
    # start_TR/end_TR. Scene order matches the summaries-derived embeddings
    # (same n_scenes / start_TR alignment verified across all movies).
    csv_path = os.path.join(CSV_DIR, f"{movie}_annot.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")
    for col in ("start_time_seconds", "end_time_seconds"):
        if col not in df.columns:
            raise ValueError(f"{movie}: CSV lacks required column '{col}'")

    annot_times = ((df["start_time_seconds"] + df["end_time_seconds"]) / 2.0).values
    n_scenes = len(annot_times)

    spacing = float(np.median(np.diff(np.sort(annot_times))))
    print(f"  n_scenes={n_scenes}, scene span=[{annot_times.min():.1f}, "
          f"{annot_times.max():.1f}]s, median scene spacing={spacing:.2f}s")

    n_tr = get_canonical_n_tr(args.qc_log, movie)
    target_times = np.arange(n_tr) * TR
    print(f"  N_TR={n_tr} (TR={TR}s) -> design spans "
          f"[0, {(n_tr - 1) * TR:.1f}]s of film")

    for L in LAYERS:
        process_layer(L, args.prompt_version, args.condition, movie, annot_times,
                    target_times, n_tr, n_scenes, spacing, embed_dir, out_dir)

if __name__ == "__main__":
    main()
