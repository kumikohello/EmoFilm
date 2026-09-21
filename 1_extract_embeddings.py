# Authors: Alicia Liu, Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: July 1, 2026

"""
extract_embeddings.py

Extracts scene-level contextual embeddings from Llama-3-8B-Instruct, following
the incremental-context approach (Tikochinski et al. 2025), adapted to the
scene level. Embeddings are extracted from multiple intermediate layers, since
the brain-encoding literature consistently finds intermediate layers align
better with brain activity than the final layer.

For each scene i, the model input is:

    "Background: " + summary_i + " Continuation of the story: " + scene_description_i

The embedding for scene i (at a given layer) is the mean of that layer's hidden
states over the tokens of scene_description_i only (the incoming context), not
the background-summary tokens.

For each requested layer, PCA reduces the (n_scenes, hidden_dim) matrix to
(n_scenes, n_components) and is saved as a separate file.

Note on hidden_states indexing: model(..., output_hidden_states=True) returns a
tuple of length (num_layers + 1). Index 0 is the embedding-layer output; index L
is the output AFTER transformer block L. So LAYERS = [12, 16, 20] selects the
outputs after blocks 12, 16, and 20.

Setup (same env as the summary job):
    transformers==4.46.3, torch 2.5.1+cu121, scikit-learn, pandas, numpy
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

# config
MODEL_ID     = "/scratch/midway3/alicial/huggingface_cache/meta-llama/Meta-Llama-3-8B-Instruct"
CSV_DIR      = "../data/csv"
OUT_DIR      = "../outputs/embeddings"
LAYERS       = [12, 16, 20]
N_COMPONENTS = 32

MOVIES = {
    "between_viewings": "between_viewings_with_summaries_gpt4o.csv",
    "big_buck_bunny":   "big_buck_bunny_with_summaries_gpt4o.csv",
    "chatter":          "chatter_with_summaries_gpt4o.csv",
    "the_secret_number":    "the_secret_number_with_summaries_gpt4o.csv",
    "sintel":           "sintel_with_summaries_gpt4o.csv",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_input_ids(tok, summary, scene):
    """
    Builds the tokenized model input following the paper's structure:
        "Background: " + summary + " Continuation of the story: " + scene

    Returns:
        input_ids:   the full token sequence (prefix + scene) as a tensor
        scene_start: index where the scene tokens begin, so the caller can
                     average hidden states over the scene tokens only
    """
    # assemble the background/continuation prefix that frames the scene
    prefix = f"Background: {summary} Continuation of the story: "
    # tokenize the prefix; include the BOS special token at the start
    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    # tokenize the scene; no special tokens, since it continues the sequence
    scene_ids  = tok(scene,  return_tensors="pt", add_special_tokens=False).input_ids
    # concatenate prefix and scene into one sequence along the token axis
    input_ids = torch.cat([prefix_ids, scene_ids], dim=1)
    # the scene begins right after the prefix tokens end
    scene_start = prefix_ids.shape[1]
    return input_ids, scene_start

def extract_layer_embeddings(model, tok, summary, scene, layers):
    """
    Runs one forward pass for a single scene and extracts its embedding at
    each requested layer.

    The embedding for a layer is the mean of that layer's hidden states over
    the scene tokens only (the incoming context), excluding the background
    summary tokens.

    Returns:
        embs: dict mapping each layer index to its (hidden_dim,) numpy vector
    """
    # build the token sequence and find where the scene portion starts
    input_ids, scene_start = build_input_ids(tok, summary, scene)
    # move tokens to the model's device (GPU if available)
    input_ids = input_ids.to(device)
    # attend to all tokens (no padding in a single-sequence input)
    attention_mask = torch.ones_like(input_ids)

    # forward pass without tracking gradients (inference only)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # return hidden states from every layer
        )

    embs = {}
    for L in layers:
        # hidden states after block L for this sequence: (seq_len, hidden_dim)
        hidden = out.hidden_states[L][0]
        # keep only the scene tokens, dropping the prefix: (scene_len, hidden_dim)
        scene_hidden = hidden[scene_start:, :]
        # average over scene tokens -> one vector; cast to float32 and move to CPU
        embs[L] = scene_hidden.mean(dim=0).float().cpu().numpy()
    return embs

def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",      default=OUT_DIR)
    parser.add_argument("--layers",       type=int, nargs="+", default=LAYERS)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--movies", nargs="+", default=list(MOVIES.keys()))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model once ─────────────────────────────────────────────────────────
    print(f"Loading model from {MODEL_ID} ...")
    print(f"Using device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval() # Set model to evaluation mode (as opposed to training)

    # Confirm the number of model layers 
    n_total_layers = model.config.num_hidden_layers
    print(f"Model has {n_total_layers} transformer layers "
          f"({n_total_layers + 1} hidden states incl. embedding layer)")
    for L in args.layers: # Check that request layer does not exceed model layers
        if L > n_total_layers:
            raise ValueError(f"Requested layer {L} exceeds model depth {n_total_layers}.")

    # ── pass 1: extract raw vectors for every movie & layer ─────────────────────
    # raw[movie][L] = (n_scenes_movie, hidden_dim)
    raw = {}
    movie_order = []
    for movie in args.movies:
        if movie not in MOVIES:
            print(f"[skip] unknown movie key: {movie}")
            continue
        csv_path = os.path.join(CSV_DIR, MOVIES[movie])
        if not os.path.exists(csv_path):
            print(f"[skip] missing summaries CSV: {csv_path}")
            continue
 
        df = pd.read_csv(csv_path, encoding="latin1")
        if "background_summary" not in df.columns:
            print(f"[skip] {movie}: CSV lacks background_summary column")
            continue
        df["background_summary"] = df["background_summary"].fillna("")
        scenes    = df["scene_description"].tolist()
        summaries = df["background_summary"].tolist()
        n = len(scenes)
        print(f"\n=== {movie}: {n} scenes ===")
 
        per_layer = {L: [] for L in args.layers}
        for i, (summ, scene) in enumerate(zip(summaries, scenes)):
            scene = "" if not isinstance(scene, str) else scene
            print(f"  scene {i+1}/{n}", flush=True)
            embs = extract_layer_embeddings(model, tok, summ, scene, args.layers)
            for L in args.layers:
                per_layer[L].append(embs[L])
 
        raw[movie] = {L: np.vstack(per_layer[L]) for L in args.layers}
        movie_order.append(movie)
 
    if not movie_order:
        raise RuntimeError("No movies processed; nothing to PCA.")

    # ── pass 2: shared PCA per layer, then project each movie ───────────────────
    # Do PCA for every layer
    for L in args.layers:
        # pool all movies' scenes for this layer
        pooled = np.vstack([raw[m][L] for m in movie_order])

        n_comp = min(args.n_components, pooled.shape[0], pooled.shape[1])
        if n_comp < args.n_components:
            print(f"  Note: only {n_comp} components possible given {pooled.shape[0]} samples.")

        pca = PCA(n_components=n_comp)
        pca.fit(pooled)
        cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
        print(f"\nLayer {L}: shared PCA on pooled {pooled.shape} "
              f"-> {n_comp} comps (cum var {cum_var:.4f})")

        # save the shared PCA model for this layer
        pca_path = os.path.join(args.out_dir, f"shared_pca_layer{L}_pca{n_comp}.npz")
        np.savez_compressed(
            pca_path,
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )
 
        # project each movie through the shared PCA and save per movie
        for m in movie_order:
            Z = pca.transform(raw[m][L]).astype(np.float32)   # (n_scenes_movie, n_comp)
            out_path = os.path.join(
                args.out_dir, f"{m}_annot_word_embeddings_layer{L}.npz"
            )
            np.savez_compressed(out_path, Z=Z)
            print(f"  {m} layer {L}: saved {out_path} | Z {Z.shape}")

    print("\nDone.")

if __name__ == "__main__":
    main()
