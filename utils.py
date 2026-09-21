"""
Shared helpers for building encoding models
"""

import ast
import pickle
from pathlib import Path
import numpy as np
from scipy.stats import zscore


def load_subject_ids(txt_path):
    """Parse subject IDs from a txt file containing a Python list literal."""
    with open(txt_path) as f:
        return ast.literal_eval(f.read())


def load_roi_data(roi_dir, roi, movie):
    """Load the per-ROI, per-movie pkl: {subject_id: (n_voxels, n_trs)}."""
    roi_path = Path(roi_dir) / f"roi_{roi:03d}_{movie}_data.pkl"
    with open(roi_path, "rb") as f:
        return pickle.load(f)
 
 
def load_group_roi_responses(roi_dir, subject_dir, group, roi, movie, n_trs=None):
    """Return {sid: (n_trs, n_voxels)} for available subjects in a group.
 
    The ROI pkl stores (n_voxels, n_trs); transpose to (n_trs, n_voxels) so it
    lines up with the (n_trs, n_features) design matrix.
 
    If n_trs is given, each response is trimmed/padded to that length so it
    aligns with the design row-for-row.
    """
    ids = load_subject_ids(Path(subject_dir) / f"{group}_IDs.txt")
    roi_data = load_roi_data(roi_dir, roi, movie)
 
    responses, missing = {}, []
    for sid in ids:
        if sid not in roi_data:
            missing.append(sid)
            continue
 
        arr = np.asarray(roi_data[sid]).T          # (n_voxels, n_trs) -> (n_trs, n_voxels)
 
        if n_trs is not None:
            if arr.shape[0] > n_trs:
                arr = arr[:n_trs]
            elif arr.shape[0] < n_trs:
                pad = np.zeros((n_trs - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.vstack([arr, pad])
 
        # already z-scored at extraction; just guard against NaNs
        responses[sid] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
 
    return responses, missing


def load_design(design_dir, layer):
    """Load a (n_trs, n_features) Lanczos+FIR design matrix for one layer."""
    return np.load(Path(design_dir) / f"{layer}_pca32_lanczsco_FIR.npy")


# ── ridge via a single SVD of X ──────────────────────────────────────────────

def ridge_weights(U, s, Vt, Y, lam):
    """w(lam) = V diag(s/(s^2+lam)) U^T Y, using precomputed SVD of X."""
    UtY = U.T @ Y
    d = s / (s ** 2 + lam)
    return Vt.T @ (d[:, None] * UtY)


def column_corr(A, B):
    """Per-column Pearson correlation between (n, m) arrays. Returns (m,)."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0))
    den = np.where(den == 0, np.nan, den)
    return num / den