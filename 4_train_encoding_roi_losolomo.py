# Authors: Kumiko Ueda (kumiko@uchicago.edu), Andrea Liu
# Last Edited: August 27, 2026

'''
Training encoding model
Per-ROI (ROI-mean) between-subject encoding model with nested leave-one-subject-out
AND leave-one-movie-out cross-validation.

Uses MNI-space ROI voxel data, collapsed to a single ROI-mean time series per
subject/movie (matching doc16's fixed behavior -- no per-voxel option here).
For each LLM layer (12, 16, 20), fits a stimulus->response encoding model on
the training subjects'/movies' ROI-mean responses and predicts the held-out
subject's response on the held-out movie. Nested LOSO selects the ridge
penalty. Outputs one mean-R per held-out subject, pooled across movies.

Collapsing to ROI-mean also sidesteps the voxel-alignment requirement a
per-voxel version would have: since each movie's mean is computed over
whatever voxels that movie's ROI pickle actually has, small voxel
count/identity mismatches across movies (see 0_parcellate_shared_voxels.py)
don't affect this version.

(August 27, 2026) adds per-TR predictions/responses to the saved pickle for the
chunk-anxiety correlation analysis (9_chunk_anxiety_correlation.py).
Writes to a separate output folder rather than overwriting the already ran
encoding_results_roi_mean_losolomo_trimmed because it is already used for the
DASS_anx correlation/plotting scripts.
'''
 
import os
import argparse
import pickle
import numpy as np
import subprocess
import datetime

from utils import(
    load_group_roi_responses, load_design, 
    ridge_weights, column_corr,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROI_DIR      = "../data/roi_data_trimmed"
SUBJECT_DIR  = "../data/subjects"
LAYERS       = ["layer12", "layer16", "layer20"]
BASE_OUT_DIR = "../outputs/encoding_results_roi_mean_losolomo_trimmed_v2"

GROUP = "all"

LAMBDAS = np.logspace(1, 3, 10) # 10 to 1000, log-spaced

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def load_responses_all_movies(movies, roi):
    """
    Load ROI responses for every movie (layer-independent), collapsed to a
    single ROI-mean time series per subject/movie via nanmean (ignores any
    NaN voxels rather than propagating them across the whole mean -- see the
    fix applied after doc16's plain .mean() produced all-NaN ROI pickles).

    Returns:
        responses_by_movie: {movie: {sid: (n_trs_movie, 1)}}
        n_trs_by_movie: {movie: n_trs}
        subjects: sorted subjects present in every movie
    """
    responses_by_movie = {}
    n_trs_by_movie = {}
    subj_sets = []

    for movie in movies:
        design_dir = f"../outputs/{movie}_trimmed"
        n_trs = load_design(design_dir, LAYERS[0]).shape[0]
        n_trs_by_movie[movie] = n_trs

        responses, missing = load_group_roi_responses(
            ROI_DIR, SUBJECT_DIR, GROUP, roi, movie, n_trs
        )
        if missing:
            print(f"    [{movie}] WARNING: {len(missing)} subjects missing: {missing}")

        responses = {sid: np.nanmean(r, axis=1, keepdims=True) for sid, r in responses.items()}

        responses_by_movie[movie] = responses
        subj_sets.append(set(responses.keys()))

    subjects = sorted(set.intersection(*subj_sets))
    dropped = set.union(*subj_sets) - set(subjects)
    if dropped:
        print(f"    dropping {len(dropped)} subjects not present in all movies: {sorted(dropped)}")

    return responses_by_movie, n_trs_by_movie, subjects

def load_designs_all_movies(movies, layer):
    """
    X_by_movie: {movie: (n_trs_movie, n_features)} for one layer.
    """
    return {movie: load_design(f"../outputs/{movie}_trimmed", layer) for movie in movies}

# Nested LOSO x LOMO for one layer
def nested_loso_lomo(X_by_movie, responses_by_movie, movies, subjects):
    '''
    Between-subject, between-movie encoding with nested LOSO x LOMO,
    one lambda per (held-out subject, held-out movie) outer fold.

    X_by_movie: {movie: (n_trs_movie, n_features)} -- each movie has its
        own design matrix, shared across subjects for that movie.
    responses_by_movie: {movie: {sid: (n_trs_movie, n_voxel_or_1)}} in
        MNI space (voxels aligned across subjects, or ROI-mean).

    Fits X -> stacked training-subject/training-movie responses, predicts
    held-out subject's response on the held-out movie, correlates per
    output column, averages across columns. Leave out both one subject
    and one movie, cycle through all movies, and concatenate predictions
    across movies for the final correlation.

    Returns:
        subject_mean_R: {sid: mean_R across columns, pooled over all movies}.
        subject_predict: {sid: {movie: (n_trs_movie, 1) predicted array}} --
            per-TR predictions, kept for downstream chunk-level analyses.
    '''
    subject_predict = {
        sid: {m: np.zeros_like(responses_by_movie[m][sid]) for m in movies}
        for sid in subjects
    }

    for out_sid in subjects:
        train_sids = [sid for sid in subjects if sid != out_sid]
        #create empty dictionary for each subject and shape like responses (filled with zeros)

        for out_movie in movies:
            train_movies = [m for m in movies if m != out_movie]
            X_test = X_by_movie[out_movie]
            #X_by_movie[m] has shape [n_tr_m, n_features]

            inner = np.zeros(len(LAMBDAS))

            for val_movie in train_movies:
                inner_movies = [m for m in train_movies if m != val_movie]
                X_val = X_by_movie[val_movie]

                X_inner = np.concatenate(
                    [np.tile(X_by_movie[m], (len(train_sids)-1, 1)) for m in inner_movies],
                    axis = 0,
                )
                U_i, s_i, Vt_i = np.linalg.svd(X_inner, full_matrices=False)

                for val_sid in train_sids:
                    inner_train = [sid for sid in train_sids if sid != val_sid]

                    Y_tr = np.concatenate(
                        [np.concatenate([responses_by_movie[m][sid] for sid in inner_train], axis=0)
                        for m in inner_movies], axis=0,
                    )
                    Y_val = responses_by_movie[val_movie][val_sid]

                    for li, lam in enumerate(LAMBDAS):
                        W = ridge_weights(U_i, s_i, Vt_i, Y_tr, lam)
                        pred = X_val @ W
                        inner[li] += np.nanmean(column_corr(pred, Y_val))
            # X_tr_inner = np.concatenate(
            #     [np.tile(X_by_movie[m], (len(train_sids)-1, 1)) for m in train_movies],
            #     axis=0,
            # )
            # U_tr, s_tr, Vt_tr = np.linalg.svd(X_tr_inner, full_matrices = False)

            # inner = np.zeros(len(LAMBDAS))
 
            # for val_sid in train_sids:
            #     inner_train = [sid for sid in train_sids if sid != val_sid]
            #     # stack inner-training subjects across training movies:
            #     # x_tr = x tiled, y_tr = y concatenated
            #     Y_tr = np.concatenate(
            #         [np.concatenate([responses_by_movie[m][sid] for sid in inner_train], axis = 0)
            #         for m in train_movies], axis=0,
            #     )
            #     Y_val = responses_by_movie[out_movie][val_sid] # held out subject, held out movie

            #     for li, lam in enumerate(LAMBDAS):
            #         W = ridge_weights(U_tr, s_tr, Vt_tr, Y_tr, lam)
            #         pred = X_test @ W
            #         inner[li] += np.nanmean(column_corr(pred, Y_val)) 
            best_lam = LAMBDAS[int(np.argmax(inner))]

            # refit on all outer-training subjects x training movies (stacked),
            # eval on held-out subject's held-out movie
            Y_tr_full = np.concatenate(
                [np.concatenate([responses_by_movie[m][sid] for sid in train_sids], axis = 0)
                for m in train_movies], axis=0,
            )
            X_tr_full = np.concatenate(
                [np.tile(X_by_movie[m], (len(train_sids), 1)) for m in train_movies],
                axis=0,
            )
            U_f, s_f, Vt_f = np.linalg.svd(X_tr_full, full_matrices = False)
            W = ridge_weights(U_f, s_f, Vt_f, Y_tr_full, best_lam)
            subject_predict[out_sid][out_movie] = X_test @ W

    # runs once, after every (subject, movie) prediction has been filled in
    subject_mean_R = {}
    for sid in subjects:
        pred_full = np.concatenate([subject_predict[sid][m] for m in movies], axis=0)
        resp_full = np.concatenate([responses_by_movie[m][sid] for m in movies], axis=0)
        subject_mean_R[sid] = float(np.nanmean(column_corr(pred_full, resp_full)))

    return subject_mean_R, subject_predict

def run_diagnostics(X_by_movie, responses_by_movie, movies, subjects):
    '''
    Returns (shuffled_R, groupmean_R), each a {sid: mean_R} dict.
    shuffled-X: randomly shuffle the embedding matrix
    How well the model does when the LLM features are scrambled?
    If LLM features predicts brain activity, then shuffling would destroy
    predictions

    group-mean: predict held-out subject from training-group mean response
    How well a no model baseline predicts each held-out subject?
    Just seeing if a held-out subject correlates with the group means of
    subjects.
    '''
    rng = np.random.default_rng(0)

    # shuffled-X null
    X_shuf_by_movie = {m: X[rng.permutation(X.shape[0])] for m, X in X_by_movie.items()}
    shuffled_R, _ = nested_loso_lomo(X_shuf_by_movie, responses_by_movie, movies, subjects)

    # group-mean baseline /noise ceiling
    groupmean_R = {}
    for out_sid in subjects:
        train_sids = [sid for sid in subjects if sid != out_sid]
        per_movie_R = []
        for m in movies:
            mean_resp = np.mean([responses_by_movie[m][sid] for sid in train_sids], axis = 0)
            per_movie_R.append(column_corr(mean_resp, responses_by_movie[m][out_sid]))
        groupmean_R[out_sid] = float(np.nanmean(np.concatenate(per_movie_R)))

    return shuffled_R, groupmean_R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True,
                     help="comma-separated movie names, e.g. chatter,sherlock,pieman")
    ap.add_argument("--roi", required=True, type=int)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--run-tag", default=None,
                     help="Optional label for this run; results go in "
                          "BASE_OUT_DIR/<run-tag>/. Defaults to a timestamp.")
    args = ap.parse_args()

    OUT_DIR = os.path.join(BASE_OUT_DIR, args.run_tag) if args.run_tag else BASE_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    movies = args.movies.split(",")

    run_config = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "stage": "encoding",
        "response_mode": "roi_mean",
        "movies": movies,
        "roi": args.roi,
        "layers": LAYERS,
        "lambdas": LAMBDAS.tolist(),
        "roi_dir": ROI_DIR,
    }

    print(f"=== Encoding: movies={movies}, roi={args.roi} "
          f"(MNI, no SRM, LOSO x LOMO, ROI-mean) ===")

    print("[1/3] Loading ROI responses ...")
    responses_by_movie, n_trs_by_movie, subjects = load_responses_all_movies(
        movies, args.roi
    )
    print(f"    {len(subjects)} subjects present in all {len(movies)} movies")

    if len(subjects) < 3:
        raise RuntimeError("Need at least 3 subjects in every movie for nested LOSO x LOMO.")

    results = {}
    predictions = {}
    diagnostics = {}
    for layer in LAYERS:
        print(f"    ---Layer: {layer}---")
        X_by_movie = load_designs_all_movies(movies, layer)
        for m in movies:
            assert X_by_movie[m].shape[0] == n_trs_by_movie[m], \
            f"design/response TR mismatch for movie {m}, layer {layer}"
        print("[2/3] Running nested LOSO x LOMO...")

        results[layer], predictions[layer] = nested_loso_lomo(X_by_movie, responses_by_movie, movies, subjects)
        vals = np.array(list(results[layer].values()))
        print(f"    mean R across subjects = {np.nanmean(vals):.4f}"
              f"(n={len(vals)}, sd = {np.nanstd(vals):.4f}, "
              f"min = {np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f})")

        if args.diagnostics:
            shuffled_R, groupmean_R = run_diagnostics(X_by_movie, responses_by_movie, movies, subjects)
            diagnostics[layer] = {"shuffled": shuffled_R, "groupmean": groupmean_R}
            print(f"    [shuffled-X]    mean R = {np.nanmean(list(shuffled_R.values())):.4f}")
            print(f"    [subject-mean]   mean R = {np.nanmean(list(groupmean_R.values())):.4f}")

    print("[3/3] Saving results ...")

    out_path = os.path.join(OUT_DIR, f"roi{args.roi:03d}_losolomo_meanR.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "roi": args.roi,
            "movies": movies,
            "layers": results,
            "predictions": predictions,
            "responses": responses_by_movie,
            "diagnostics": diagnostics,
            "config": run_config,
        }, f)
    print(f"    saved-> {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()
