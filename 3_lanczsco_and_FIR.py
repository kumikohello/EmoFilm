"""
# Authors: Alicia Liu, Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 24, 2026
Lanczos resampling + FIR delay stacking for PCA-reduced annotation embeddings.

Pipeline:
  1. Load (n_scenes, 32) PCA-reduced embeddings, one row per scene annotation.
  2. Read each annotation's onset/offset in seconds (start_time_seconds / end_time_seconds
     columns of {MOVIE}_annot.csv) and place the annotation at the center of its window,
     in seconds since FILM ONSET.
  3. Resample to the fMRI TR grid (N_TRs, TR=1.3s, time 0 = film onset) using a 3-lobe Lanczos kernel.
     Output: (N_TR, 32).
  4. Z-score each feature column across time.
  5. Build the FIR design matrix by concatenating 4 delayed copies
     (shifts of 1, 2, 3, 4 TRs). Output: (N_TR, 128). # OLD VERSION (464, 128)
  6. Save.

N_TR is read from trim_qc_log.csv (produced by b1_extract_runY.py) as the MINIMUM
n_trs used across subjects for this movie. This guarantees every subject's trimmed
response array is at least N_TR long, so load_group_roi_responses only ever truncates
responses to match, never zero-pads.

Note on scipy: scipy.signal.windows.lanczos returns a window function for
filter design / spectral analysis, not a resampling kernel. The interpolation
kernel we need (sinc(x) * sinc(x/a)) is implemented directly below.
"""

import os
import argparse
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EMBED_DIR   = "../outputs/embeddings"
CSV_PATH    = "../data/csv"
QC_LOG = "../data/bold_runY_trimmed/trim_qc_log.csv"
# N_TR        = 464          # total fMRI TRs
LANCZOS_A   = 3            # number of lobes for lanczo filter 
FIR_DELAYS  = [1, 2, 3, 4] # delays in TRs for FIR 
LAYERS = [12, 16, 20]
TR = 1.3

QC_FILM_NAME = {
    "between_viewings":  "BetweenViewings",
    "big_buck_bunny":    "BigBuckBunny",
    "chatter":           "Chatter",
    "the_secret_number": "TheSecretNumber",
    "secret_number":     "TheSecretNumber",
    "sintel":            "Sintel",
    "payload":           "Payload",
    "first_bite":        "FirstBite",
    "you_again":         "YouAgain",
    "after_the_rain":    "AfterTheRain",
    "lesson_learned":    "LessonLearned",
    "spaceman":          "Spaceman",
    "to_claire_from_sonny": "ToClaireFromSonny",
    "superhero":         "Superhero",
    "tears_of_steel":    "TearsOfSteel",
}

# -----------------------------------------------------------------------------
# Lanczos kernel
# -----------------------------------------------------------------------------
def lanczos_kernel(x, a=3):
    """L(x) = sinc(x) * sinc(x/a) for |x| < a, else 0.

    np.sinc uses the normalized convention sinc(x) = sin(pi*x)/(pi*x), which
    matches what the Lanczos kernel needs.
    """
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)

def lanczos_resample(embeddings, annot_times, target_times, spacing, a=3):
    """Resample (N_annot, D) embeddings from annot_times onto target_times.

    spacing: characteristic distance between annotations (used to scale
    the kernel). With non-uniform annotation spacing, use the modal or
    typical spacing.
    """
    # (n_target, n_annot) distance matrix in units of spacing
    dist = (target_times[:, None] - annot_times[None, :]) / spacing

    # (n_target, n_annot) weight matrix
    weights = lanczos_kernel(dist, a=a)

    # normalize each row so weights sum to 1 (handles edges)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights = weights / row_sums

    # (n_target, D) = (n_target, n_annot) @ (n_annot, D)
    return weights @ embeddings

# -----------------------------------------------------------------------------
# FIR delay stacking
# -----------------------------------------------------------------------------
def delay_shift(X, k):
    """Shift X forward in time by k samples. Row t of output = row t-k of X.
    Rows 0..k-1 of the output are zero (no past data available).
    """
    out = np.zeros_like(X)
    if k > 0:
        out[k:] = X[:-k]
    elif k == 0:
        out[:] = X
    return out

def build_fir_design(X, delays):
    """Concatenate delayed copies of X along the feature axis."""
    return np.concatenate([delay_shift(X, k) for k in delays], axis=1)

# -----------------------------------------------------------------------------
# N_TR from the QC log
# -----------------------------------------------------------------------------
def get_canonical_n_tr(qc_log_path, movie_key):
    """
    Minimum n_trs_used across all subjects for this movie, from trim_qc_log.csv.
    Using the minimum guarantees every subject's trimmed response is >= N_TR, so
    load_group_roi_responses only ever truncates, never zero-pads.
    """
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
# Main
# -----------------------------------------------------------------------------
def process_layer(L, movie, annot_times, target_times, n_tr, n_annot, spacing, out_dir):
    in_path  = os.path.join(EMBED_DIR, f"{movie}_annot_word_embeddings_layer{L}.npz")
    out_path = os.path.join(out_dir, f"layer{L}_pca32_lanczsco_FIR.npy")

    print(f"=== {movie} layer {L} ===")

    if not os.path.exists(in_path):
        print(f"[skip] missing embeddings: {in_path}\n")
        return

    emb = np.load(in_path)["Z"]
    print(f"  loaded: {emb.shape}  (expected ({n_annot}, 32))")
    if emb.shape != (n_annot, 32):
        print(f"  [skip] layer {L}: embedding rows {emb.shape[0]} != annotations {n_annot}\n")
        return

    # Step 1: Lanczos resample (116, 32) -> (n_tr, 32)
    resampled = lanczos_resample(
        emb, annot_times, target_times, spacing=spacing, a=LANCZOS_A
    )
    print(f"  resampled:  {resampled.shape}  (expected ({n_tr}, 32))")
    assert resampled.shape == (n_tr, 32)

    # Step 2: z-score each feature column across time
    mean = resampled.mean(axis=0, keepdims=True)
    std  = resampled.std(axis=0, keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    zscored = (resampled - mean) / std
    print(f"  z-scored:   {zscored.shape}  (expected ({n_tr}, 32))")

    # Step 3: FIR stack (809, 32) -> (809, 128)
    fir = build_fir_design(zscored, FIR_DELAYS)
    n_fir = 32 * len(FIR_DELAYS)
    print(f"  FIR design: {fir.shape}  (expected ({n_tr}, {n_fir}))")
    assert fir.shape == (n_tr, n_fir)

    np.save(out_path, fir)
    print(f"  saved to:   {out_path}")
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True,
                    help="movie key matching {movie}_annot.csv, e.g. between_viewings")
    ap.add_argument("--out-dir", default=None,
                    help="Default: ../outputs/{movie}_trimmed")
    ap.add_argument("--qc_log", default=QC_LOG)
    args = ap.parse_args()

    movie = args.movie
    out_dir = args.out_dir or f"../outputs/{movie}_trimmed"
    os.makedirs(out_dir, exist_ok=True)
 
    csv_path = os.path.join(CSV_PATH, f"{movie}_annot.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing CSV: {csv_path}")
 
    df = pd.read_csv(csv_path, encoding="latin-1")
    for col in ("start_time_seconds", "end_time_seconds"):
        if col not in df.columns:
            raise ValueError(f"{movie}: CSV lacks required column '{col}'")
 
    annot_times = ((df["start_time_seconds"] + df["end_time_seconds"]) / 2.0).values
    n_annot = len(annot_times)

    spacing = float(np.median(np.diff(np.sort(annot_times))))
    print(f"  n_annot={n_annot}, annot span=[{annot_times.min():.1f}, "
          f"{annot_times.max():.1f}]s, median scene spacing={spacing:.2f}s")
    
    n_tr = get_canonical_n_tr(args.qc_log, movie)
    target_times = np.arange(n_tr) * TR
    print(f"  N_TR={n_tr} (TR={TR}s) -> design spans "
          f"[0, {(n_tr - 1) * TR:.1f}]s of film")

    for L in LAYERS:
        process_layer(L, movie, annot_times, target_times, n_tr, n_annot, spacing, out_dir)

if __name__ == "__main__":
    main()
