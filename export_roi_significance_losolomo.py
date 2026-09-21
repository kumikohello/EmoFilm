# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: August 31, 2026

"""
Flatten pickles into CSV of per-subject values.
"""

import glob, pickle, pandas as pd

rows = []
for path in sorted(glob.glob("../outputs/encoding_results_roi/loso_lomo_trimmed_v2/roi*_losolomo_meanR.pkl")):
    res = pickle.load(open(path, "rb"))
    roi = res["roi"]
    diag = res.get("diagnostics") or {}
    for layer in ["layer12", "layer16", "layer20"]:
        R = res["layers"][layer]
        gm = diag.get(layer, {}).get("groupmean", {})
        sh = diag.get(layer, {}).get("shuffled", {})
        for sid, r in R.items():
            rows.append({
                "roi": roi, "layer": layer, "subject_id": sid, 
                "R": r,
                "ceiling": gm.get(sid, float("nan")),
                "shuffled": sh.get(sid, float("nan")),
            })
pd.DataFrame(rows).to_csv("../outputs/roi_per_subject_R_trimmed.csv", index=False)

rows = []
for path in sorted(glob.glob("../outputs/encoding_results_roi_mean_losolomo_trimmed_v2/roi*_losolomo_meanR.pkl")):
    res = pickle.load(open(path, "rb"))
    roi = res["roi"]
    diag = res.get("diagnostics") or {}
    for layer in ["layer12", "layer16", "layer20"]:
        R = res["layers"][layer]
        gm = diag.get(layer, {}).get("groupmean", {})
        sh = diag.get(layer, {}).get("shuffled", {})
        for sid, r in R.items():
            rows.append({
                "roi": roi, "layer": layer, "subject_id": sid, 
                "R": r,
                "ceiling": gm.get(sid, float("nan")),
                "shuffled": sh.get(sid, float("nan")),
            })
pd.DataFrame(rows).to_csv("../outputs/roi_mean_per_subject_R_trimmed.csv", index=False)
