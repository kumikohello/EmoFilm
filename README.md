# EmoFilm

This repository contains scripts and data for preprocessing and analyzing fMRI data collected during a narrative (movie-watching) experiment. The goal is to investigate cognitive responses to naturalistic stimuli.

---

## 🧪 Scripts Overview

All scripts below are located in `scripts` and are designed to be run in sequential order as part of the fMRI analysis pipeline.

NEED: `utils.py`, `data/conditions.json`, `data/participants.tsv`, `Schaefer2018_200Parcels_7Networks_order.txt`

### 0. `b1_extract_runY.py`
Reads BOLD NIFTI for each subject/film and build that subject's own brain mask and slices the volume down to each film window; z-scores per voxel. 

### 1. `0_gpt_summarize.py`
Extracts rolling incremental summaries of the movies.

### 2. `0_parcellate_shared_voxels.py`
Extracts per-ROI voxel data, across all movies.

### 3. `1_extract_embeddings.py`
Extract scene-level contextual embeddings.

### 4. `3_lanczos_and_FIR.py`
Lanczos resampling + FIR delay stacking for PCA-reduced annotation embeddings.

### 5. `4_train_encoding_roi_losolomo.py`
Per-ROI (ROI-mean) between-subject encoding model with nested leave-one-subject-out (LOSO) and leave-one-movie-out (LOMO) cross-validation.

### 6. `5_build_result_roi_losolomo_csv.py`
Build a per-subject master CSV from the ROI-mean LOSO x LOMO encoding pkls.

### 7. `6_plot_roi_losolomo_surface.py`
Build per-ROI brain maps from the ROI-mean LOSO x LOMO encoding results.

### 8. `correlate_anxiety_roi_losolomo.py`
Correlate LOSO x LOMO ROI-mean encoding results with DASS_anx score, per ROI, per layer.

### 9. `plot_correlate_anxiety_grouped.py`
Plots the anxiety correlation results.

### 10.`export_roi_significance_losolomo.py`
Flattens pkl files to csv files to run significance tests in R.

### 11. `EmoFilm_R.Rmd` - not on cluster
Run significance tests in R.

### 12. `plot_schaefer_betas.py`
Surface maps of the LOSO x LOMO ROI-mean effect with black outlines around
FDR-significant parcels.

---

## Anxiety Models
Scripts to analyze using anxiety models.

### 1. `8_anxious_llama_embeddings.py`
Extract scene-level embeddings using incremental-context approach.

### 2. `3_lanczos_and_FIR_anxious.py`
Lanczos resampling + FIR delay stacking for PCA-reduced annotation embeddings.

### 3. `8_anxious_llama_encoding.py`
Per-ROI (ROI-mean) between-subject encoding model with nested leave-one-subject-out (LOSO) and leave-one-movie-out (LOMO) cross-validation.

### 4. `5_build_result_anxiety_csv.py`
Build a per-subject master CSV from the ROI-mean LOSO x LOMO encoding pkls.

### 5. `6_plot_roimean_anxiety.py`
Per-ROI brain maps from the ROI-mean LOSO x LOMO encoding results.

### 6. `correlate_anxiety_anxiousmodel.py`
Correlate LOSO x LOMO, ROI-mean encoding results with DASS_anx score, per ROI, per layer.

### 7. `export_significance_anxiousmodel.py`
Flattens pkl files to csv files to run significance tests in R.

### 8. `plot_schaefer_betas_anxious.py`
Surface maps of the LOSO x LOMO ROI-mean effect with black outlines around
FDR-significant parcels.

---

### 9. `5_build_result_anxiety_network_csv.py`
Build a per-subject master CSV for the per-network LOSO x LOMO encoding pkls.

### 10. `6_plot_roimean_anxiety_network.py`
Plot per-network (Schaefer 7 networks) brain maps from the LOSO x LOMO encoding results.

### 11. `plot_schaefer_betas_anxious_network.py`
Surface maps of the LOSO x LOMO per-network effect with black outlines around
FDR-significant parcels.

---

### 12. `condition_diff_anxiety.py`
Compute the per-subject, per-ROI, per-layer difference in LOSO x LOMO
ROI-mean encoding R between two anxious-model conditions, correlate that
difference with trait anxiety (DASS_anx), AND plot both as Schaefer-200
surface maps.
