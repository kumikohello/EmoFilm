# Authors: Kumiko Ueda (kumiko@uchicago.edu), Andrea Liu
# Last Edited: August 18, 2026

'''
Extract scene-level embeddings from meta-llama/Llama-3.1-8B-Instruct

Using incremental-context approach (Tikochinski et al. 2025), human context window is smaller
than llm's context window.

Extract from multiple intermediate layers- middle layers better predicts brain activity
(Schrimpf et al. 2021)

For each scene i, the model input is:
    "Preprompt + Background: " + summary_i + " Continuation of the story: " + scene_description_i
    Not as instruction or chat formatted, just system input

Embedding for scene i (at given layer) is the mean of the layer's hidden states (activations)
over the tokens of scene_description_i only (the incoming context), excluding the background
summary tokens.

PCA (principle component analysis) reduces (n_scenes, hidden_dim) matrix to (n_scenes, 
n_components) and is saved as a separate file for each requested layer.

Hidden states indexing: model(..., output_hidden_states=TRUE) returns a tuple of length 
(num_layers + 1). Index 0 is the embedding-layer output (token embeddings- input to 
first transformer block); index L is the output after transformer layer block L. 
LAYERS = [12, 16, 20] selects the outputs AFTER blocks 12, 16, and 20.
'''

import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from prompts import prefixes

# configuration
MODEL_ID     = "/scratch/midway3/alicial/huggingface_cache/meta-llama/Meta-Llama-3-8B-Instruct"
CSV_DIR      = "../data/csv"
OUT_DIR      = "../outputs/embeddings"
LAYERS = [12, 16, 20] 
N_COMPONENTS = 32 # can experiment with different number of components for PCA
CONDITIONS_LOC = "/project/ycleong/users/kumiko/EmoFilm/data/conditions.json"

MOVIES = {
    "between_viewings": "between_viewings_with_summaries_gpt4o.csv",
    "big_buck_bunny":   "big_buck_bunny_with_summaries_gpt4o.csv",
    "chatter":          "chatter_with_summaries_gpt4o.csv",
    "the_secret_number":    "the_secret_number_with_summaries_gpt4o.csv",
    "sintel":           "sintel_with_summaries_gpt4o.csv",
}

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== build prompt and tokenize
def build_input_ids(tokenizer, preprompt, summary, scene, prompt_v):
    '''
    Structure of input: 
    "Background: " + summary + "Continuation of the story: " + scene

    Returns:
        input_ids: full token sequence (prefix + scene) as tensor
        scene_start: index where scene tokens begin, so caller can average
                     hidden states over scene tokens only and not background-summary tokens
    '''
    prefix = prefixes[prompt_v].format(preprompt = preprompt, summary = summary)
    prefix_ids = tokenizer(prefix, return_tensors = "pt", add_special_tokens=True).input_ids
    # .input_ids is an attribute of the returned BatchEncoding object, 
    #  containing only the input tokens which is a tensor of shape (1, seq_len)

    # tokenizer is defaulted to no padding, even if inputs are different lengths,
    # we only have one input sequence at a time, so no padding is needed

    scene_ids = tokenizer(scene, return_tensors = "pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prefix_ids, scene_ids], dim = 1)
    #concatenate prefix and scene token ids along the sequence dimension (dim=1), 
    # dim = 0 is the batch dimension (column), dim = 1 is the sequence dimension (row)

    scene_start = prefix_ids.shape[1] 
    # scene_start after the prefix tokens end, prefix_ids has shape (batch_size, seq_len), 
    # so prefix_ids.shape[1] gives the length of the prefix sequence
    # one tensor is one run, batch size is the number of sequences processed together
    # padding is only needed with multiple sequences of different lengths in a batch

    return input_ids, scene_start

def embeddings_output(model, tokenizer, preprompt, summary, scene, layers, prompt_v):
    '''
    Run one forward pass of the a single scene (plus its background summary if there is one)
    and extract embeddings at each requested layer.

    The embedding we need is for the mean of the hidden states over the scene tokens only,
    excluding summary tokens. Hidden states match the input sequence length, so we can 
    index into the hidden states tensor to get only the scene tokens.

    Returns:
    embs: dict mapping each layer index to its (hidden_dim,) numpy vector
    output_text: greedy-decoded continuation generated from the same input
    '''

    # build the token sequence and find where the scene portion starts
    input_ids, scene_start = build_input_ids(tokenizer, preprompt, summary, scene, prompt_v)

    # move tokens to the model's device (GPU if available)
    input_ids = input_ids.to(torch_device) # .to() a method of all torch tensors, moves the tensor to the specified device (CPU or GPU)

    # attend to all tokens(no padding in a single-sequence input)
    attention_mask = torch.ones_like(input_ids) 
    # create a tensor of ones with the same shape as input_ids, 
    # indicating that all tokens should be attended to

    with torch.no_grad(): # context manager to disable gradient calculation, saves memory and computations during inference
        out = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True, # return hidden states from every layer
        )

    embs = {}

    for L in layers:
        # out.hidden_states is a tuple of length (num_layers + 1), where index 0 is the embedding-layer output 
        # (token embeddings- input to first transformer block); index L is the output after transformer layer block L.
    
        # hidden states after block L for this sequence: (seq_len, hidden_dim)
        hidden = out.hidden_states[L][0] # [0] to get the one and only sequence in batch

        # select only the scene tokens, exclude prefix
        scene_hidden = hidden[scene_start:, :] # shape (token_len, hidden_dim)

        # average over scene tokens -> one vector; cast to float32 and move to CPU
        embs[L] = scene_hidden.mean(dim=0).cpu().numpy().astype(np.float32)

    # --- generated response: separate call on the SAME input ---
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,          # how much continuation to read
            do_sample=False,             # greedy = deterministic/reproducible
            pad_token_id=tokenizer.eos_token_id,
        )

    # keep only the newly generated tokens (drop the prompt), then decode
    output_text = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)

    return embs, output_text

def load_movie_data(csv_dir, movie_keys):
    """
    Loads scene_description + background_summary for each requested movie key.
    Returns dict: movie -> {"scenes": [...], "summaries": [...]}
    Skips (with a warning) any movie key not in MOVIES or whose CSV is missing.
    """
    movie_data = {}
    for movie in movie_keys:
        if movie not in MOVIES:
            print(f"[skip] unknown movie key: {movie}")
            continue
        csv_path = os.path.join(csv_dir, MOVIES[movie])
        if not os.path.exists(csv_path):
            print(f"[skip] missing summaries CSV: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path, encoding="utf-8")
        if "scene_description" not in df.columns or "background_summary" not in df.columns:
            print(f"[skip] {movie}: CSV missing scene_description/background_summary column")
            continue
        
        df["background_summary"] = df["background_summary"].fillna("") # replace na with empty string
        movie_data[movie] = {
            "scenes": df["scene_description"].tolist(),
            "summaries": df["background_summary"].tolist(),
        }
        print(f"Loaded {len(movie_data[movie]['scenes'])} scenes for {movie}")

    return movie_data
    
def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default=CSV_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--prompt_version", choices= list(prefixes), default="original")
    parser.add_argument("--movies", nargs="+", default=list(MOVIES.keys()))
    args = parser.parse_args()

    # Open and Read JSON conditions file
    with open(CONDITIONS_LOC, "r") as f:
        conditions = json.load(f)
        
    conditionsCLEAN = {}
    for k, v in conditions.items():
        if isinstance(v, list):
            conditionsCLEAN[k] = " ".join(v)
        else:
            conditionsCLEAN[k] = v

    os.makedirs(args.out_dir, exist_ok=True)

    # Load scenes/summaries for every requested movie
    movie_data = load_movie_data(args.csv_dir, args.movies)
    if not movie_data:
        raise RuntimeError("No movies loaded; nothing to do.")

    print(f"Movies: {list(movie_data.keys())}")
    print(f"Extracting from layers: {args.layers}")

    # Load llama-3-8B-Instruct model
    print(f"Loading model from {MODEL_ID} ...")
    print(f"Using device: {torch_device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32,
        device_map = {"": torch_device},
    )
    model.eval() # eval, not training

    # Confirm the number of model layers
    n_total_layers = model.config.num_hidden_layers
    print(f"Model has {n_total_layers} transformer layers "
          f"({n_total_layers + 1} hidden states include embedding layer)")
    # embedding "layer" does not count as a layer, it's only a hidden state

    for L in args.layers: 
        # check request layer does not exceed model layers
        if L > n_total_layers:
            raise ValueError(f"Requested layer {L} exceeds model depth {n_total_layers}.")
        
    # raw[condition][movie][L] -> list of (hidden_dim,) vectors, one per scene
    raw = {
        condition: {movie: {L: [] for L in args.layers} for movie in movie_data}
        for condition in conditionsCLEAN
    }

    output_rows = []
    for condition, p_prompt in conditionsCLEAN.items():
        print(f"\n=== condition: {condition} ===")
        print(p_prompt)
        for movie, data in movie_data.items():
            scenes = data["scenes"]
            summaries = data["summaries"]
            n = len(scenes)
            print(f"  movie: {movie} ({n} scenes)")
            for i, (summ, scene) in enumerate(zip(summaries, scenes)):
                print(f"  Scene {i+1} / {n}: extracting ...", flush = True)
                embs, output_text = embeddings_output(model, tokenizer, p_prompt, summ, scene, args.layers, args.prompt_version)
                for L in args.layers:
                    raw[condition][movie][L].append(embs[L])
                output_rows.append({
                    "movie": movie,
                    "scene_index": i,
                    "condition": condition,
                    "prompt_version": args.prompt_version,   # the template: original/s1/s2
                    "scene": scene,
                    "output_text": output_text,
                })

    # For each condition & layer: fit PCA on scenes pooled across ALL movies
    # (shared PCA space), then project each movie's scenes through it and save separately
    for condition in conditionsCLEAN:
        for L in args.layers:
            pooled = np.vstack([np.vstack(raw[condition][movie][L]) for movie in movie_data])
            print(f"\nCondition {condition}, Layer {L}: embedding matrix shape {pooled.shape}")

            n_comp = min(args.n_components, pooled.shape[0], pooled.shape[1])
            # getting the number for components, not exceeding number of scenes

            if n_comp < args.n_components:
                print(f"    Note: only {n_comp} components possible give {pooled.shape[0]} samples.")
            
            pca = PCA(n_components = n_comp) # setting up PCA configured for 32 components
            pca.fit(pooled)
            cum_var = np.cumsum(pca.explained_variance_ratio_)[-1]
            # getting the cumulative percent of variance captured, -1 is the cumulative for all ratios

            print(f"    Shared PCA fit: {n_comp} comps (cumulative explained var: {cum_var:.4f})")

            # save the shared PCA model for this condition+layer, so movies can be
            # compared / new movies projected later without refitting
            pca_path = os.path.join(
                args.out_dir,
                f"{args.prompt_version}_condition_{condition}_layer{L}_shared_pca{n_comp}.npz",
            )
            np.savez_compressed(
                pca_path,
                components=pca.components_.astype(np.float32),
                mean=pca.mean_.astype(np.float32),
                explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            )
            print(f"    Save shared PCA -> {pca_path}")

            #project each movie's scenes through the shared PCA and save separately
            for movie in movie_data:
                mat = np.vstack(raw[condition][movie][L])
                reduced = pca.transform(mat).astype(np.float32)
                out_path = os.path.join(args.out_dir, f"{args.prompt_version}_condition_{condition}_{movie}_layer{L}_pca{n_comp}.npy")
                np.save(out_path, reduced)
                print(f"    {movie}: saved -> {out_path} (shape {reduced.shape})")

    #compile output_text csv
    df_out = pd.DataFrame(output_rows)
    print(f"output_rows shape: {df_out.shape}, columns: {df_out.columns.tolist()}")
    df_out.to_csv(os.path.join(args.out_dir, f"all_conditions_{args.prompt_version}_output_texts.csv"),
                index=False, encoding="utf-8")

    print("\nDone.")

if __name__ == "__main__":
    main()
