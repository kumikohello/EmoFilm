# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 16, 2026 (August 21, 2026 fixed)
"""
Reads BOLD NIFTI for each subject/session/film. Builds that subject's own brain 
mask (intersected across their runs), and slices the volume down to each film
window using events.tsv, dropping the rest-period padding. Z-scores per voxel.

Saves as bold_runY_trimmed/sub-X/ses-Y_Film_Y.npy.

Each subject has their own voxel count and voxel identity so nothing is aligned
across people yet.
"""

import os
import re
import glob
import json
import csv
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

# ===================== CONFIG =====================
FMRIPREP_DERIV = "/project/ycleong/datasets/EmoFilm_prep"
EVENTS_DERIV = "/project/ycleong/datasets/EmoFilm" # raw BIDS root, where *_events.tsv live
OUT_DIR = "/project/ycleong/users/kumiko/EmoFilm/data/bold_runY_trimmed"
QC_LOG = os.path.join(OUT_DIR, "trim_qc_log.csv")

SUBJECTS_DEFAULT = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32']
SESSIONS = ["ses-1", "ses-2", "ses-3", "ses-4"]
FILMS_DEFAULT = ["BigBuckBunny", "FirstBite", "YouAgain", "AfterTheRain", "LessonLearned", "Payload", "TheSecretNumber", "BetweenViewings", "Chatter", "Spaceman", "ToClaireFromSonny", "Sintel", "Superhero", "TearsOfSteel"]

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", default=None, help="Comma-separated subject IDs, e.g. 'S15' or 'S01,S15'. Default: all subjects.")
ap.add_argument("--films", default=None, help="Comma-separated film names. Default: all films.")
args = ap.parse_args()

SUBJECTS = args.subjects.split(",") if args.subjects else SUBJECTS_DEFAULT
FILMS = args.films.split(",") if args.films else FILMS_DEFAULT

SPACE = "MNI152NLin6Asym" # matches your filenames
USE_MNI = True            # use the space-MNI preproc bold + mask

# Optional confound regression (recommended later; start False for speed)
REGRESS_CONFOUNDS = False
CONFOUND_COLS = [
    "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
    "csf","white_matter","global_signal","framewise_displacement"
]

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== HELPERS =====================
def load_mask_bool(mask_path):
    mask_img = nib.load(mask_path)
    M = mask_img.get_fdata() > 0.5
    return M

def load_confounds(tsv_path, n_tr):
    df = pd.read_csv(tsv_path, sep="\t")
    cols = [c for c in CONFOUND_COLS if c in df.columns]
    if len(cols) == 0:
        raise ValueError(f"No requested confounds found in {tsv_path}")
    C = df[cols].copy().fillna(0.0)
    C["intercept"] = 1.0
    C = C.to_numpy(dtype=np.float32)
    return C[:n_tr, :]

def regress_out(Y, C):
    beta, *_ = np.linalg.lstsq(C, Y, rcond=None)  # (K,V)
    return Y - (C @ beta)

def can_open_nifti(path):
    """Lightweight check: can nibabel read header + shape?"""
    try:
        img = nib.load(path)
        _ = img.shape
        return True
    except Exception:
        return False

def get_repetition_time(bold_path):
    """
    Read RepetitionTime (seconds) from the BOLD sidecar JSON.
    """
    json_path = re.sub(r"\.nii(\.gz)?$", ".json", bold_path)
    with open(json_path) as f:
        meta = json.load(f)
    tr = meta.get("RepetitionTime")
    if tr is None:
        raise ValueError(f"No RepetitionTime in {json_path}")
    return float(tr)

def find_events_path(subj, sess, film):
    """
    Events files use a different naming pattern than the BOLD files:
        sub-{subj}_{sess}_task-scan_acq-{film}_events.tsv
    Falls back to a glob in case of naming drift.
    """
    direct = os.path.join(
        EVENTS_DERIV, f"sub-{subj}", sess, "func",
        f"sub-{subj}_{sess}_task-scan_acq-{film}_events.tsv"
    )
    if os.path.exists(direct):
        return direct
    
    pattern = os.path.join(
        EVENTS_DERIV, f"sub-{subj}", sess, "func",
        f"sub-{subj}_{sess}_task-*{film}*_events.tsv"
    )
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def load_film_window(events_path, tr):
    """
    Read the events.tsv, find the trial_type == 'film' row, and convert its
    onset/duration (seconds) into [start_idx, end_idx) TR indices.

    Using onset+duration for the end index (rather than start_idx + n_trs)
    avoids rounding drift over long durations.
    """
    df = pd.read_csv(events_path, sep="\t")
    film_rows = df[df["trial_type"] == "film"]
    if len(film_rows) == 0:
        raise ValueError(f"No 'film' trial_type row in {events_path}")
    if len(film_rows) > 1:
        print(f"    [WARN] {events_path}: multiple 'film' rows, using the first")
    row = film_rows.iloc[0]
    onset, duration = float(row["onset"]), float(row["duration"])

    start_idx = int(round(onset / tr))
    end_idx = int(round((onset + duration) / tr))
    return start_idx, end_idx, onset, duration

# ===================== MAIN =====================
qc_rows = []

for subj in SUBJECTS:
    print(f"\n=== Extracting run-level Y (trimmed to film) for sub-{subj} ===")
    subj_dir = os.path.join(FMRIPREP_DERIV, f"sub-{subj}")
    if not os.path.isdir(subj_dir):
        print(f"Missing subj dir: {subj_dir}")
        continue

    run_info = []

    out_subj = os.path.join(OUT_DIR, f"sub-{subj}")
    os.makedirs(out_subj, exist_ok=True)

    for sess in SESSIONS:
        func_dir = os.path.join(subj_dir, sess, "func")
        if not os.path.isdir(func_dir):
            print(f"Missing func dir: {func_dir}")
            continue

        for film in FILMS:
            bold_path = os.path.join(
                func_dir,
                f"sub-{subj}_{sess}_task-{film}_space-{SPACE}_desc-preproc_bold.nii.gz"
            )
            mask_path = os.path.join(
                func_dir,
                f"sub-{subj}_{sess}_task-{film}_space-{SPACE}_desc-brain_mask.nii.gz"
            )
            conf_path = os.path.join(
                func_dir,
                f"sub-{subj}_{sess}_task-{film}_desc-confounds_timeseries.tsv"
            )

            if not (os.path.exists(bold_path) and os.path.exists(mask_path)):
                print(f"Missing bold/mask for {sess}, film {film}")
                continue

            # If the gzip is corrupted, avoid using it in mask intersection
            if not can_open_nifti(bold_path):
                print(f"Bold not readable {sess}, film {film} (likely corrupted) → skipping")
                continue

            run_info.append((sess, film, bold_path, mask_path, conf_path))

    if len(run_info) == 0:
        print("No usable runs for this subject.")
        continue

    # build intersection mask across runs (skip corrupted masks gracefully)
    M_int = None
    good_run_info = []
    for sess, film, bold_path, mask_path, conf_path in run_info:
        try:
            M = load_mask_bool(mask_path)
        except Exception as e:
            print(f"Mask read failed {sess}, film {film}: {e} (skipping run for mask intersection)")
            continue

        if M_int is None:
            M_int = M
        else:
            M_int = M_int & M
        good_run_info.append((sess, film, bold_path, mask_path, conf_path))

    if M_int is None or len(good_run_info) == 0:
        print("Could not build intersection mask.")
        continue

    V_common = int(M_int.sum())
    print(f"Intersection mask voxels (V_common): {V_common}")

    # Save voxel indices for later debugging/consistency checks
    vox_idx = np.where(M_int.ravel())[0].astype(np.int64)
    np.save(os.path.join(out_subj, "intersection_vox_idx.npy"), vox_idx)

    # 2) Extract Y for each run using the SAME mask
    for sess, film, bold_path, mask_path, conf_path in good_run_info:
        print(f"Run {sess}, {film}: extracting...")

        try:
            tr = get_repetition_time(bold_path)
        except Exception as e:
            print(f"Failed to read RepetitionTime {sess}, film {film}: {e} → skipping")
            continue
        
        events_path = find_events_path(subj, sess, film)
        if events_path is None:
            print(f"Noe events.tsv found for {sess}, film {film} -> skipping (cannot trim safely)")
            continue
        
        try:
            start_idx, end_idx, onset, duration = load_film_window(events_path, tr)
        except Exception as e:
            print(f"Failed to parse film window {sess}, film {film}: {e} -> skipping")
            continue
        
        try:
            bold_img = nib.load(bold_path)
            B_full = bold_img.get_fdata(dtype=np.float32) # (X, Y, Z, T_full)
        except Exception as e:
            print(f"Failed to read bold {sess}, film {film}: {e} -> skipping")
            continue

        T_full = B_full.shape[-1]
        end_idx_clamped = min(end_idx, T_full)
        if start_idx >= T_full or start_idx >= end_idx_clamped:
            print(f"Bad film window for {sess}, film {film}: "
                  f"start={start_idx}, end={end_idx}, T_full={T_full} -> skipping")
            continue
        if end_idx_clamped < end_idx:
            print(f"    [WARN] {sess}, film {film}: requested end TR {end_idx} > "
                  f"scan length {T_full}; clamping to {end_idx_clamped}")

        B = B_full[..., start_idx:end_idx_clamped] # TRIM to film window
        T = B.shape[-1]

        # Mask then reshape to (T, V_common)
        Y = B[M_int].reshape(-1, T).T

        # Optional confound regression
        if REGRESS_CONFOUNDS and os.path.exists(conf_path):
            try:
                # confounds are on the FULL run's timeline - slice them the same way
                C_full = load_confounds(conf_path, n_tr=T_full)
                C = C_full[start_idx:end_idx_clamped]
                Y = regress_out(Y, C)
            except Exception as e:
                print(f"Confound regression failed {sess}, film {film}: {e} (continuing without)")

        # Z-score per voxel (model-space normalization)
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-6)

        out_path = os.path.join(out_subj, f"{sess}_{film}_Y.npy")
        np.save(out_path, Y.astype(np.float32))
        print(f"Saved {out_path} | shape={Y.shape} | TR={tr}s | "
              f"onset={onset:.2f}s duration={duration:.2f}s | TRs [{start_idx}:{end_idx_clamped}]")
        
        qc_rows.append({
            "subject": subj, "session": sess, "film": film,
            "tr_seconds": tr, "onset_seconds": onset, "duration_seconds": duration,
            "start_tr": start_idx, "end_tr_requested": end_idx,
            "end_tr_used": end_idx_clamped, "n_trs_used": T,
            "full_scan_trs": T_full, "clamped": end_idx_clamped < end_idx,
        })

# write QC log so alignment can be checked at a glance, per run
if qc_rows:
    with open(QC_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qc_rows)
    print(f"\nQC log written to {QC_LOG} ({len(qc_rows)} runs)")

print("\nDone extracting run-level Y (trimmed to film window)")
