# Authors: Alicia Liu, Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 30, 2026

"""
generate_summaries_openai.py

Generates rolling incremental summaries for each scene in the Bang annotation CSV,
using GPT-4o via the OpenAI API. Same structure as the Llama version:
for each scene i, the summary is built from summary_{i-1} + scene_description[i].
Scene 1 has no background summary (empty string).

Output: CSV with columns [index, start_TR, end_TR, scene_description, background_summary]
where background_summary[i] is the context to prepend when extracting the
embedding for scene i.

Setup:
    pip install openai pandas
    export OPENAI_API_KEY="sk-..."

Run:
    python generate_summaries_openai.py
"""

import os
import time
import argparse
import pandas as pd
from openai import OpenAI

# ── config ──────────────────────────────────────────────────────────────────────
CSV_DIR    = "../data/csv"
MODEL       = "gpt-4o"
MAX_WORDS   = 80
MOVIES = {
    "between_viewings": {
        "input":   "between_viewings_annot.csv",
        "output":  "between_viewings_with_summaries_gpt4o.csv",
        "context": "A comedy/drama short film following a disillusioned estate agent selling his childhood home.",  # e.g. "A drama short film about ..."
    },
    "big_buck_bunny": {
        "input":   "big_buck_bunny_annot.csv",
        "output":  "big_buck_bunny_with_summaries_gpt4o.csv",
        "context": "An animated short film with no dialogue, following a large rabbit.",
    },
    "chatter": {
        "input":   "chatter_annot.csv",
        "output":  "chatter_with_summaries_gpt4o.csv",
        "context": "A thriller short film following a girl.",
    },
    "sintel": {
        "input":   "sintel_annot.csv",
        "output":  "sintel_with_summaries_gpt4o.csv",
        "context": "An animated fantasy short film following a young woman searching for a dragon.",
    },
    "the_secret_number": {
        "input":   "the_secret_number_annot.csv",
        "output":  "the_secret_number_with_summaries_gpt4o.csv",
        "context": "A drama short film following a psychiatrist compelled by his patient.",
    },
}

client = OpenAI()   # reads OPENAI_API_KEY from environment


def build_messages(previous_summary: str, current_scene: str, movie_context: str):
    """Builds the chat messages for one summary update."""
    context_line = f"{movie_context}\n\n" if movie_context else ""

    if previous_summary:
        user_content = (
            f"You are maintaining a running summary of a movie.\n\n"
            f"{context_line}"
            f"Rewrite the summary below to reflect the latest scene. "
            f"Focus on the current state of the story, "
            f"only keeping background context from earlier that is still DIRECTLY relevant, "
            f"and dropping details that are no longer relevant to the most recent scene. "
            f"Avoid direct quotes from the movie; summarize concisely in your own words. "
            f"Write no more than {MAX_WORDS} words. Return only the summary, no preamble.\n\n"
            f"Here is the most recent summary:\n{previous_summary}\n\n"
            f"Here is the latest scene:\n{current_scene}\n\n"
        )
    else:
        user_content = (
            f"You are maintaining a running summary of a movie.\n\n"
            f"{context_line}"
            f"First scene:\n{current_scene}\n\n"
            f"Write a brief summary of this scene in no more than {MAX_WORDS} words. "
            f"Return only the summary, no preamble or explanation."
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert movie reviewer writing a running summary "
                "of a movie's plot. Always respond with the summary only."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def generate_summary(previous_summary: str, current_scene: str, movie_context: str) -> str:
    messages = build_messages(previous_summary, current_scene, movie_context)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"    API error (attempt {attempt+1}): {e}. Retrying in {wait}s ...")
            time.sleep(wait)

    raise RuntimeError("Failed to generate summary after 5 attempts.")

def process_movie(movie, cfg):
    in_path  = os.path.join(CSV_DIR, cfg["input"])
    out_path = os.path.join(CSV_DIR, cfg["output"])
    movie_context = cfg.get("context", "")
 
    print(f"\n=== {movie} ===")
 
    if not os.path.exists(in_path):
        print(f"  [skip] missing input CSV: {in_path}")
        return
 
    df = pd.read_csv(in_path, encoding="latin1")
    scenes = df["scene_description"].tolist()
    n = len(scenes)
    print(f"  Loaded {n} scenes from {in_path}")
 
    # summaries[i] = background context used when embedding scene i
    # summaries[0] = "" (no prior context for the first scene)
    summaries = [""]
    current_summary = ""
 
    for i, scene in enumerate(scenes):
        # Guard against blank/NaN scenes: carry context forward, don't call API.
        if not isinstance(scene, str) or scene.strip() == "":
            print(f"  Scene {i+1}/{n}: empty, carrying previous summary forward")
            summaries.append(current_summary)
            continue
 
        print(f"  Scene {i+1}/{n}: generating summary ...", flush=True)
        new_summary = generate_summary(current_summary, scene, movie_context)
        summaries.append(new_summary)
        current_summary = new_summary
        print(f"    -> {new_summary[:80]}{'...' if len(new_summary) > 80 else ''}")
 
    # drop the trailing entry so it aligns with the n scenes:
    # saved background_summary[i] is the summary BEFORE scene i.
    summaries = summaries[:n]
 
    keep_cols = [c for c in ["index", "start_TR", "end_TR", "scene_description"] if c in df.columns]
    out_df = df[keep_cols].copy()
    out_df["background_summary"] = summaries
    out_df.to_csv(out_path, index=False)
    print(f"  Done. Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", nargs="+", default=list(MOVIES.keys()),
                        help="Subset of movie keys to process (default: all).")
    args = parser.parse_args()
 
    for movie in args.movies:
        if movie not in MOVIES:
            print(f"[skip] unknown movie key: {movie}")
            continue
        process_movie(movie, MOVIES[movie])

if __name__ == "__main__":
    main()
