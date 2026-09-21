# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: July 29, 2026

"""
Extract per-ROI VOXEL data (Schaefer-200), across ALL movies together.

Produces, for each ROI k and movie m:
    roi_{k:03d}_{movie}_data.pkl  =  {sid: (n_voxels_in_roi, n_trs)}

IMPORTANT: for leave-one-movie-out analyses to be valid, voxel v of a given ROI 
must refer to the SAME physical voxel (and the same *count* of voxels) in every 
movie's pickle. That requires computing ONE cross-subject voxel intersection, 
using ONE fixed subject cohort, shared across every movie -- not a separate 
intersection per movie.

Previously (see extract_roi_voxels.py), extraction ran one movie at a time, and 
the cross-subject voxel intersection was computed only among the subjects who 
happened to have usable data for THAT movie. Since different movies have 
different missing-subject lists, the intersection (and thus per-ROI voxel count) 
could differ movie to movie -- e.g. ROI 54 with 179 voxels in one movie's pickle 
and 180 in another's. Concatenating across movies for LOSO x LOMO then fails 
(or, worse, silently misaligns voxels) because column i of the ROI array no longer
refers to the same physical voxel across movies.

This version fixes that by:
  1. Determining, up front, which subjects have usable data in EVERY movie being processed.
  2. Computing ONE global cross-subject voxel intersection using only those subjects 
     (a subject's raw voxel index array doesn't depend on movie, so this is 
     well-defined regardless of which movie you look at it from).
  3. Using that SAME common_idx / column order for every movie's extraction pass, 
     so ROI k has identical voxel count and voxel identity in every movie's pickle.

How it works, without re-reading any BOLD NIfTIs:
  - Each subject's intersection_vox_idx.npy gives their voxels as FLAT indices 
    into the MNI grid, in the same order as their Y columns.
  - The atlas, resampled onto that same grid, tells you which ROI each flat index 
    belongs to.
  - So ROI membership is just a lookup on the flat indices, and slicing each 
    subject's Y by the matching columns gives that ROI's voxels.

CRITICAL: the atlas is resampled with nearest-neighbour interpolation.
Labels are categorical -- linear interpolation would invent nonexistent
ROI numbers.

Usage:
    python 0_parcellate_shread_voxels.py --movies chatter,between_viewings,big_buck_bunny,the_secret_number,sintel
    python 0_parcellate_shared_voxels.py --movies chatter,sintel --atlas-path /path/to/schaefer.nii.gz
"""

import os
import ast
import glob
import pickle
import argparse
from pathlib import Path
 
import numpy as np
import nibabel as nib
from nilearn import datasets, image
from scipy.stats import zscore

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BOLD_DIR    = "/project/ycleong/users/kumiko/EmoFilm/data/bold_by_movie_trimmed"
RUNY_DIR    = "/project/ycleong/users/kumiko/EmoFilm/data/bold_runY_trimmed"
FMRIPREP    = "/project/ycleong/datasets/EmoFilm_prep"
SUBJECT_DIR = "../data/subjects"
OUT_DIR     = "../data/roi_data_trimmed"

GROUP  = "all"
N_ROI  = 200          # Schaefer-200

# Map lowercase movie key -> CamelCase BOLD filename stem
BOLD_NAME = {
    "between_viewings":  "BetweenViewings",
    "big_buck_bunny":    "BigBuckBunny",
    "chatter":           "Chatter",
    "secret_number":     "TheSecretNumber",
    "the_secret_number": "TheSecretNumber",
    "sintel":            "Sintel",
}

def load_subject_ids(txt_path):
    with open(txt_path) as f:
        return ast.literal_eval(f.read())

def find_reference_mask(sid):
    """A subject's MNI brain mask defines the grid the flat indices refer to."""
    for sess in ["ses-1", "ses-2", "ses-3", "ses-4"]:
        pattern = os.path.join(
            FMRIPREP, f"sub-{sid}", sess, "func",
            f"sub-{sid}_{sess}_task-*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz"
        )
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None

def load_atlas_on_grid(ref_img, atlas_path=None):
    """Load Schaefer-200 and resample it onto the BOLD grid.
 
    Nearest-neighbour only: labels are categorical.
    Returns a flat int array, one label per voxel of the reference grid
    (0 = outside any parcel).
    """
    if atlas_path:
        atlas_img = image.load_img(atlas_path)
        print(f"  atlas: {atlas_path}")
    else:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROI)
        atlas_img = image.load_img(atlas["maps"])
        print(f"  atlas: fetched Schaefer-{N_ROI}")
 
    print(f"  atlas native shape: {atlas_img.shape}")
    print(f"  reference shape:    {ref_img.shape[:3]}")

    atlas_rs = image.resample_to_img(
        atlas_img, ref_img, interpolation="nearest"
    )
    labels = np.asarray(atlas_rs.get_fdata()).astype(np.int32)
    print(f"  resampled shape:    {labels.shape}")

    present = np.unique(labels)
    present = present[present > 0]
    print(f"  parcels present after resampling: {present.size} "
          f"(expected {N_ROI})")
    if present.size < N_ROI:
        missing = sorted(set(range(1, N_ROI + 1)) - set(present.tolist()))
        print(f"  NOTE: {len(missing)} parcels vanished at this resolution: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    return labels.ravel()

def find_usable_subjects(ids, movie):
    """Subjects with both a Y file and a voxel-index file for this movie."""
    movie_stem = BOLD_NAME.get(movie, movie)
    usable, missing = {}, []
    for sid in ids:
        y_path   = Path(BOLD_DIR) / f"sub-{sid}" / f"{movie_stem}_Y.npy"
        idx_path = Path(RUNY_DIR) / f"sub-{sid}" / "intersection_vox_idx.npy"
        if not y_path.exists() or not idx_path.exists():
            missing.append(sid)
            continue
        usable[sid] = y_path
    return usable, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True,
                     help="comma-separated movie names, e.g. chatter,sintel,big_buck_bunny")
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--atlas-path", default=None,
                    help="Optional explicit Schaefer NIfTI; otherwise fetched via nilearn.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    movies = args.movies.split(",")
    print(f"=== movies: {movies} ===")

    ids = load_subject_ids(Path(SUBJECT_DIR) / f"{args.group}_IDs.txt")

    # ---- pass 1: which subjects are usable in EVERY movie? ----
    # This is the cohort the whole extraction will be built around, so
    # the voxel intersection below is identical no matter which movie
    # you're looking at.
    y_paths_by_movie = {}
    usable_sets = []
    for movie in movies:
        usable, missing = find_usable_subjects(ids, movie)
        y_paths_by_movie[movie] = usable
        usable_sets.append(set(usable.keys()))
        print(f"  [{movie}] {len(usable)} usable subjects"
              f"{f', {len(missing)} missing: {missing}' if missing else ''}")
 
    sids = sorted(set.intersection(*usable_sets))
    dropped = set.union(*usable_sets) - set(sids)
    if dropped:
        print(f"  dropping {len(dropped)} subjects not usable in every movie: {sorted(dropped)}")
    if len(sids) < 3:
        raise RuntimeError("Fewer than 3 subjects usable in every movie.")
    print(f"  {len(sids)} subjects usable in ALL {len(movies)} movies -- "
          f"this fixed cohort is used for every movie's voxel intersection below")

    # ---- subject voxel-index arrays (movie-independent) ----
    subj_vox = {
        sid: np.load(Path(RUNY_DIR) / f"sub-{sid}" / "intersection_vox_idx.npy").astype(np.int64)
        for sid in sids
    }

    # ---- ONE global cross-subject voxel intersection, shared by all movies ----
    common = None
    for sid in sids:
        s = set(subj_vox[sid].tolist())
        common = s if common is None else (common & s)
    common_idx = np.array(sorted(common), dtype=np.int64)
    print(f"  common voxels across the shared cohort: {common_idx.size}")

    # ---- atlas on the BOLD grid (same grid for every subject/movie) ----
    ref_path = find_reference_mask(sids[0])
    if ref_path is None:
        raise FileNotFoundError(f"No reference mask for sub-{sids[0]}")
    ref_img = nib.load(ref_path)
    atlas_flat = load_atlas_on_grid(ref_img, args.atlas_path)

    # ROI label of each common voxel, in common_idx order -- shared by
    # every movie, so ROI k has the same voxel count/identity everywhere.
    common_labels = atlas_flat[common_idx]
    in_atlas = int(np.sum(common_labels > 0))
    print(f"  common voxels inside a parcel: {in_atlas} "
          f"({100.0 * in_atlas / common_idx.size:.1f}%)")

    # ---- per movie: load, align to the SHARED common_idx, slice per ROI ----
    for movie in movies:
        print(f"\n### {movie} ###")
        y_paths = y_paths_by_movie[movie]

        aligned = {}
        for sid in sids:
            Y = np.load(y_paths[sid]).astype(np.float32)   # (n_trs, V_subj)
            if Y.shape[1] != subj_vox[sid].size:
                print(f"    [skip] {sid}: Y width {Y.shape[1]} != vox_idx {subj_vox[sid].size}")
                continue
            pos = {v: i for i, v in enumerate(subj_vox[sid].tolist())}
            cols = np.array([pos[v] for v in common_idx], dtype=np.int64)
            Yc = Y[:, cols]                                # (n_trs, V_common) -- same V_common every movie
            # z-score each voxel over time, matching the encoding loader
            Yc = zscore(Yc, axis=0)
            aligned[sid] = np.nan_to_num(Yc, nan=0.0, posinf=0.0, neginf=0.0)

        if not aligned:
            print(f"    [skip movie] no subjects aligned for {movie}")
            continue
        n_trs = next(iter(aligned.values())).shape[0]
        print(f"    aligned: {len(aligned)} subjects, n_trs={n_trs}")

        n_written, n_empty = 0, 0
        for k in range(1, N_ROI + 1):
            cols_k = np.where(common_labels == k)[0]      # columns of the aligned Y
            if cols_k.size == 0:
                n_empty += 1
                continue

            # {sid: (n_voxels_in_roi, n_trs)} -- note the transpose, which
            # is what load_roi_data / load_group_roi_responses expects
            roi_data = {sid: aligned[sid][:, cols_k].T.astype(np.float32)
                        for sid in aligned}

            out_path = os.path.join(OUT_DIR, f"roi_{k:03d}_{movie}_data.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(roi_data, f)
            n_written += 1

            if k % 25 == 0:
                print(f"      roi {k:3d}: {cols_k.size:5d} voxels -> {out_path}")

        print(f"    wrote {n_written} ROI pickles, {n_empty} empty (no voxels at this resolution)")

    print(f"\n  output dir: {OUT_DIR}")
    print("Done. Every movie's ROI pickles now share the same voxel count/order per ROI.")

if __name__ == "__main__":
    main()