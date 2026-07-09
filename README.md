# BlameBERT — Blame Attribution in the Danish Parliament


A project by **Rune Trust** and **Markus Lundsfryd Jensen**.

This repository contains the full pipeline behind *"BlameBERT — Blame Attribution in the Danish Parliament,"* a study examining how blame rhetoric in the Danish Parliament (Folketinget) has evolved from **1997 to 2026**, and how it relates to party ideology and government status. The project combines:

- **BlameBERT**, a fine-tuned multilingual BERT classifier for detecting blame at the sentence level (Macro F1: **0.80**), built via an annotation-efficient, natural language inference (NLI)-assisted labeling pipeline.
- **Multilevel statistical modeling** (negative binomial mixed-effects regression) quantifying how time, government status, ideological wing, and ideological intensity ("wingness") predict blame attribution across ~5 million parliamentary sentences.

The trained [model](https://huggingface.co/Lundsfryd/DaBERT-BlameDetector) and [dataset](https://huggingface.co/datasets/runetrust/blame-folketinget-dk) are available on Hugging Face, and full code is provided here for reproducibility.

---

## Key Findings

- **A "banana-shaped" temporal trend:** blame attribution declined until roughly **April 2016**, then entered a significant, sustained increase through 2026 — consistent with public perceptions of rising political incivility.
- **Political contrasting:** opposition parties consistently blame more than governing parties; governing parties blame at roughly **60–62%** the rate of the opposition, an effect that holds even after controlling for the topical agenda of debates.
- **Ideological asymmetry:** the blame-dampening effect of being in government is *weaker* for right-wing parties, and increasing ideological intensity ("wingness") amplifies blame *more* on the right than the left — an effect that intensifies further in recent years (2019–2026).
- **Robustness:** core findings hold across multiple classification-confidence thresholds in a dedicated sensitivity analysis.

See the paper for full hypotheses (H1, H1.1, H2, H2.1), model specifications, and discussion.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
  - [1. Data Collection & Preprocessing](#1-data-collection--preprocessing)
  - [2. Translation & Preliminary Labeling (DEBATE)](#2-translation--preliminary-labeling-debate)
  - [3. Gold Test Set & Human Annotation](#3-gold-test-set--human-annotation)
  - [4. Model Training (BlameBERT)](#4-model-training-blamebert)
  - [5. Baseline Model Comparisons](#5-baseline-model-comparisons)
  - [6. Inference](#6-inference)
  - [7. Statistical Analysis](#7-statistical-analysis)
- [Model Performance](#model-performance)
- [Data](#data)
- [Requirements & Installation](#requirements--installation)
- [Citation](#citation)

---

## Repository Structure

```
BlameBERT/
├── data_making/scripts/     # Data collection & preprocessing scripts (XML extraction, merging, cleaning)
├── src/                     # Core Python modules for translation, labeling, training, and inference
├── nbs/                     # Jupyter notebooks, organized by pipeline stage
├── analysis/                # R and Python statistical analysis (GLMMs, sensitivity, topic modeling)
├── data/                    # Raw, training, and inference datasets
└── README.md
```

---

## Pipeline Overview

The full pipeline runs from raw parliamentary transcripts to statistical inference. A visual flowchart is included in the paper (Figure 1 / Appendix A).

### 1. Data Collection & Preprocessing

Parliamentary speech data combines two sources:
- **ParlSpeechV2** (Rauh & Schwalbach), covering Folketinget from **07-10-1997 to 20-12-2018**.
- A custom fetch from the **Danish Parliament's SFTP server**, covering **06-10-2009 to 26-02-2026**.

The two datasets were merged at the 2018/2019 boundary after a consistency check (no evidence of a structural discontinuity — see Appendix A of the paper). Sentence segmentation was performed with **DaCy** (`da_dacy_large_trf`). Sentences containing parentheses or shorter than five characters, and all chairman utterances, were filtered out, reducing ~6.55M sentences to a final working set of ~5.5M unique sentences.

For the political analysis, the dataset is further restricted to the 13 parties active in continental Denmark at time of writing, each annotated with **wing** (left/right) and **wingness** (ideological distance from center), derived from the Chapel Hill Expert Survey's "lrecon" index.

### 2. Translation & Preliminary Labeling (DEBATE)

500,500 sentences were randomly sampled for training/validation/testing. These were:
1. Machine-translated Danish → English using **Opus-MT-da-en**.
2. Passed through **DEBATE** (`Political_DEBATE`), a zero-shot NLI classifier, using five hypothesis templates targeting the labels *blame*, *praise*, and *neutral*.
3. Labeled as **blame** only if DEBATE's blame probability was ≥ 0.80 and exceeded both other labels.

Because agreement across templates varies, five training datasets of increasing label conservatism — **DIAL-1 through DIAL-5** — were constructed (agreement of at least 1 to all 5 templates). Blame prevalence ranges from 1.71% (DIAL-1) to 0.50% (DIAL-5).

### 3. Gold Test Set & Human Annotation

500 sentences (250 blame / 250 non-blame, balanced sampling from DIAL-1) were manually annotated by both authors following the blame definition of Bilotta (2025): a causal utterance carrying negative sentiment. Inter-annotator agreement reached **84.8% (Cohen's κ = 0.676)**. Only sentences with full agreement were retained, yielding a **424-sentence gold test set** (34.9% blame / 65.1% no blame).

### 4. Model Training (BlameBERT)

- **Base model:** `mmBERT` (multilingual BERT, ~307M parameters).
- **Fine-tuning:** full-precision **LoRA** (rank 64, alpha 128) with **focal loss** to address class imbalance.
- **Hyperparameter search:** grid search over the 5 DIALs × 3 learning rates (`1e-5`, `1e-4`, `5e-4`) × 3 focal-loss alpha scalings (raw, ²⁄₃ power, √), with gamma fixed at 2.0. Each DIAL subset used 20,000 sentences, split 80/20 train/validation, selecting the best checkpoint by Matthews Correlation Coefficient (MCC).
- **Final model:** trained on **DIAL-5** (learning rate `1e-4`, √ class-weight alpha), chosen for its balance of precision and recall.

Experiment tracking was done with **Weights & Biases**; see `nbs/4_training/` for training notebooks and `src/` for the underlying training pipeline (LoRA setup, focal loss, grid search, embedding visualization via UMAP).

### 5. Baseline Model Comparisons

BlameBERT was benchmarked against two open-source zero-shot alternatives on the gold test set:
- **Qwen 3 Embedding (0.6B)** — anchor-sentence cosine-similarity classification.
- **Generative Qwen 3.5 (9B)** — structured JSON output via Ollama, Danish system prompt, temperature 0.

BlameBERT outperformed both on macro F1 while requiring a fraction of the inference cost (minutes vs. ~19 hours on an RTX 4070 Super for the generative model).

### 6. Inference

The trained BlameBERT model is applied to all held-out sentences (~5M), producing a `prediction` and `confidence` score per sentence. Minister-to-party mapping accounts for politicians who changed party affiliation over time (e.g., cabinet reshuffles). See `nbs/5_inference/`.

### 7. Statistical Analysis

Conducted in **R** using `glmmTMB`, with models compared via likelihood ratio tests:

- **H1 / H1.1** — Negative binomial mixed-effects models of blame counts (offset by sentence volume, party random intercept) test for linear/quadratic effects of time, both across the full period and restricted to 2019–2026.
- **H2 / H2.1** — Extended models add government status, wing, wingness, and their interactions as predictors, again for both the full period and the 2019–2026 subset.
- **Sensitivity analysis** — All focal models are refit at three increasingly conservative blame-classification thresholds (τ = 0.625, 0.75, 0.875) to test robustness.
- **Topic control** — A supplementary sentence-level logistic mixed model, using `ManifestoBERTa` to classify political topic domain, tests whether the government-status effect on blame survives controlling for parliamentary agenda (it does).

R Markdown files and rendered HTML reports are in `analysis/r/`; supporting Python exploration and power analysis notebooks are in `analysis/python/`.

---

## Model Performance

| Model | Precision | Recall | Macro F1 |
|---|---|---|---|
| Qwen 3 Embedding (0.6B, anchor) | 0.70 | 0.72 | 0.67 |
| Generative Qwen 3.5 (9B) | 0.91 | 0.71 | 0.75 |
| **BlameBERT** | **0.72** | **0.79** | **0.80** |

BlameBERT's final DIAL-5 checkpoint achieves an average precision of 0.80, recall of 0.81, and accuracy of 0.82 on the held-out gold test set. No systematic error pattern was found across parties (see confusion-matrix breakdown in the paper's appendix).

---

## Data

| Path | Description |
|---|---|
| `data/raw_data/` | Raw merged Folketing transcripts (ParlSpeechV2 + SFTP fetch) and government/cabinet metadata |
| `data/training_data/` | DIAL-sampled training data and the gold-labeled 424-sentence validation set |
| `data/inference/` | Sentences prepared for full-corpus inference and resulting predictions |

Full dataset and model card are published on Hugging Face (see project links below); code for all preprocessing and label construction is available under `data_making/scripts/` and `src/`.

---

## Requirements & Installation

### Python

```bash
pip install torch transformers datasets peft accelerate
pip install spacy dacy
pip install pandas tqdm scikit-learn
pip install umap-learn plotly wandb
pip install sentencepiece sacremoses
```

For sentence segmentation:

```bash
python -m spacy download da_dacy_large_trf
```

### R

Statistical analysis requires:

```r
install.packages(c("tidyverse", "lubridate", "zoo", "glmmTMB", "DHARMa", "emmeans", "sjPlot"))
```
