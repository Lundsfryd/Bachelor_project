# BlameBERT: Blame Detection in Danish Parliamentary Debates

A project by **Markus Lundsfryd Jensen** and **Rune Egeskov Trust**.

This project develops **BlameBERT**, a fine-tuned multilingual BERT model for detecting *blame attribution* in transcripts from the Danish Parliament (Folketinget). The full pipeline covers everything from raw parliamentary data to a trained and deployed model, including machine translation, zero-shot preliminary labelling, human annotation, LoRA-based fine-tuning, inference, and statistical analysis.

The trained model is publicly available on Hugging Face:
**[Lundsfryd/blameBERT](https://huggingface.co/Lundsfryd/blameBERT)**

### Reproducability of XML files: Folketingets open data (ODA) FTP server can be accessed by following this guide: https://www.ft.dk/-/media/sites/ft/pdf/dokumenter/aabne-data/oda-browser_brugervejledning.pdf 
 ---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1 — Preprocessing & Sentence Segmentation](#step-1--preprocessing--sentence-segmentation)
  - [Step 2 — Machine Translation & Preliminary Blame Detection (PBD)](#step-2--machine-translation--preliminary-blame-detection-pbd)
  - [Step 3 — Annotation & Validation](#step-3--annotation--validation)
  - [Step 4 — Training BlameBERT](#step-4--training-blamebert)
  - [Step 5 — Inference](#step-5--inference)
- [Source Scripts (src/)](#source-scripts-src)
- [Analysis](#analysis)
- [Data](#data)
- [Requirements & Installation](#requirements--installation)
- [Usage Examples](#usage-examples)

---

## Project Overview

The goal of the project is to automatically detect sentences in which a Danish politician is *blaming* another party, politician, or institution.

1. Raw parliamentary debate transcripts (XML/CSV from Folketing) are cleaned and sentence-segmented.
2. Sentences are translated from Danish to English using the Helsinki-NLP MarianMT model.
3. A zero-shot preliminary blame detection (PBD) step uses the `Political_DEBATE_large_v1.0` model with multiple hypothesis templates to create a weakly labelled dataset.
4. Human annotators validate a subset of this data using Label Studio, producing a gold-labelled validation set.
5. The `jhu-clsp/mmBERT-base` model is fine-tuned with LoRA adapters on the agreed-upon labelled data.
6. The trained BlameBERT model is used to run inference over all Folketing debate transcripts.
7. Statistical analysis is performed in R using mixed-effects models to study blame attribution patterns across parties and time.

---

## Repository Structure

```
BlameBERT/
│
├── src/
│   ├── sentence_segmentation.py # Sentence segmentation using DaCy/spaCy
│   ├── machine_trans.py # Danish->English translation (MarianMT)
│   ├── PBD.py                            # Preliminary Blame Detection (zero-shot NLI)
│   ├── template_manipulation.py          # Hypothesis template processing
│   ├── json_formatting.py                # Label Studio I/O, inter-annotator agreement
│   ├── danish_minister_party_pipeline.py # Minister->party assignment by date
│   ├── blame_detection.py                # BlameDetector class (inference & validation)
│   ├── training_setup.py                 # BERT fine-tuning setup (LoRA, Trainer)
│   ├── training_pipeline_with_viz.py     # Training pipeline with embedding visualisation
│   ├── full_training.py                  # Final training loop (multiple LRs)
│   ├── gridsearch_training.py            # Hyperparameter sweep (LR x data x alpha)
│   ├── subset_training_pipe.py           # Training on subsets of data
│   ├── embedding_viz.py                  # Embedding visualisation callbacks
│   ├── plot_embeddings.py                # Standalone embedding plots
│   ├── xml_extract.py                    # XML extraction from Folketing raw files
│   └── wandb_api_test.py                 # Weights & Biases integration test
│
├── nbs/                          # Jupyter notebooks (ordered by pipeline step)
│   ├── 1_preprocessing_and_sentence_segmentation/
│   │   ├── PLS_rds_to_csv.Rmd                # Convert .rds Folketing data to CSV
│   │   ├── initial_data_formatting.ipynb     # Initial data cleaning and formatting
│   │   └── sent_seg_and_datasplit.ipynb      # Sentence segmentation and train/test split
│   │
│   ├── 2_machine_translation_and_PBD/
│   │   ├── machine_translation.ipynb         # Danish->English translation
│   │   ├── PBD.ipynb                         # Preliminary Blame Detection runs
│   │   └── hypothesis_templates.txt          # NLI hypothesis templates used for PBD
│   │
│   ├── 3_annotation_and_validation/
│   │   ├── label_studio_in.ipynb             # Format data for Label Studio import
│   │   └── label_studio_out.ipynb            # Process Label Studio annotations
│   │
│   ├── 4_training/
│   │   ├── Training_pipeline_final_model.ipynb   # Final model training notebook
│   │   ├── looping_training.ipynb                # Iterative training experiments
│   │   └── wandb_attempt.ipynb                   # W&B experiment tracking
│   │
│   ├── 5_inference/
│   │   └── Blame_inference.ipynb             # Run BlameBERT on full Folketing data
│   │
│   └── requirements.txt                      # Python dependencies for notebooks
│
├── analysis/
│   ├── python/
│   │   ├── power_analysis.ipynb              # Statistical power analysis
│   │   └── inference_data.ipynb              # Post-inference data exploration
│   └── r/
│       ├── blame_analysis.Rmd                # Main statistical analysis (GLMMs)
│       ├── blame_analysis.html               # Rendered HTML report
│       ├── leaderboard_party.Rmd             # Party-level blame leaderboard analysis
│       └── leaderboard_party.html            # Rendered HTML report
│
├── data/
│   ├── raw_data/
│   │   ├── Corp_Folketing_V2.csv             # Raw Folketing corpus
│   │   └── danish_govs.csv                   # Danish government/cabinet metadata
│   ├── training_data/
│   │   └── validation_set/
│   │       ├── validation_set.jsonl           # Final gold-labelled validation set
│   │       └── intermediate_files/            # Intermediate annotator files
│   │           ├── markus_annotated.json
│   │           ├── markus_gold_annotated.jsonl
│   │           ├── rune_annotated.json
│   │           └── rune_gold_annotated.jsonl
│   └── inference/
│       └── inference_data.jsonl              # Data prepared for final inference run
│
└── skraldespand/                 # Archived/experimental work (Danish: "trash can")
    ├── failed_llm_identifications/   # Abandoned LLM-based approaches (Llama, DeepSeek)
    ├── data_making/scripts/          # Earlier versions of data pipeline scripts
    ├── training/                     # Exploratory training notebooks
    ├── json_files/                   # Intermediate JSON data files
    └── Model_data/                   # Early validation set versions
```

---

## Pipeline Walkthrough

### Step 1 — Preprocessing & Sentence Segmentation

**Notebooks:** `nbs/1_preprocessing_and_sentence_segmentation/`  
**Script:** `src/sentence_segmentation.py`

Raw debate transcripts from the Danish Parliament are loaded from a CSV file. Each paragraph is sentence-segmented using [DaCy](https://github.com/centre-for-humanities-computing/DaCy)'s large transformer model (`da_dacy_large_trf`) via spaCy. Each output sentence is stored as a JSONL record together with metadata (date, speaker, party, paragraph number, sentence number).

```bash
python src/sentence_segmentation.py \
  --inpath_to_csv data/raw_data/Corp_Folketing_V2.csv \
  --outpath_to_jsonl data/training_data/sentences.jsonl \
  --batch_size 4 \
  --n_process 1
```

| Argument | Default | Description |
|---|---|---|
| `--inpath_to_csv` | required | Path to the input CSV with parliamentary text |
| `--outpath_to_jsonl` | required | Output path for segmented sentences |
| `--n_rows` | `-1` (all) | Number of paragraphs to process |
| `--batch_size` | `2` | Batch size for spaCy pipeline |
| `--n_process` | `1` | Number of parallel processes |

---

### Step 2 — Machine Translation & Preliminary Blame Detection (PBD)

**Notebooks:** `nbs/2_machine_translation_and_PBD/`  
**Scripts:** `src/machine_trans.py`, `src/PBD.py`

#### Machine Translation

Danish sentences are translated to English using the `Helsinki-NLP/opus-mt-da-en` MarianMT model. Translations are appended to each JSONL record under the `translated_text` key.

```bash
python src/machine_trans.py \
  --input_path_jsonl data/training_data/sentences.jsonl \
  --output_path_jsonl data/training_data/sentences_translated.jsonl \
  --batch_size 8
```

| Argument | Default | Description |
|---|---|---|
| `--input_path_jsonl` | required | Input JSONL with `text` field |
| `--output_path_jsonl` | required | Output JSONL with added `translated_text` field |
| `--batch_size` | `2` | Translation batch size |

#### Preliminary Blame Detection (PBD)

Translated sentences are passed through `mlburnham/Political_DEBATE_large_v1.0`, a zero-shot natural language inference (NLI) model. Multiple hypothesis templates (defined in `hypothesis_templates.txt`) are tested against each sentence. A sentence is labelled as blame (1) if the blame label scores highest among {blame, praise, neutral} *and* exceeds 0.8. Each template's binary verdict is added as a separate field (e.g., `Hyp_1_blame`, `Hyp_2_blame`, ...).

```bash
python src/PBD.py \
  --input_path_jsonl data/training_data/sentences_translated.jsonl \
  --output_path_jsonl data/training_data/pbd_output.jsonl \
  --hyp_temp_path nbs/2_machine_translation_and_PBD/hypothesis_templates.txt \
  --batch_size 4
```

| Argument | Default | Description |
|---|---|---|
| `--input_path_jsonl` | required | Input JSONL with `translated_text` field |
| `--output_path_jsonl` | required | Output JSONL with hypothesis blame fields |
| `--hyp_temp_path` | required | Path to text file with one hypothesis template per line |
| `--batch_size` | `2` | Inference batch size |

---

### Step 3 — Annotation & Validation

**Notebooks:** `nbs/3_annotation_and_validation/`  
**Script:** `src/json_formatting.py`

A subset of the PBD-labelled data was exported to [Label Studio](https://labelstud.io/) for manual annotation by both authors. The `Formatter` class in `json_formatting.py` handles:

- Converting JSONL to Label Studio's import format
- Processing Label Studio exports back to JSONL
- Computing inter-annotator agreement (Cohen's Kappa, classification reports)

Annotated files went through multiple reconciliation rounds, with gold-labelled versions stored under `data/training_data/validation_set/intermediate_files/`. The final agreed-upon validation set is `data/training_data/validation_set/validation_set.jsonl`.

---

### Step 4 — Training BlameBERT

**Notebooks:** `nbs/4_training/`  
**Scripts:** `src/training_setup.py`, `src/training_pipeline_with_viz.py`, `src/full_training.py`, `src/gridsearch_training.py`

The base model `jhu-clsp/mmBERT-base` (multilingual BERT) is fine-tuned for binary sequence classification (blame / not blame) using LoRA adapters via the `peft` library and Hugging Face `Trainer`.

Training features:
- **LoRA fine-tuning** for parameter-efficient training
- **Weighted Binary Cross-Entropy** loss to handle class imbalance, with configurable alpha modes (`sqrt`, `two_thirds`, `raw`)
- **Early stopping** based on validation loss
- **Inverse square root** or **linear** learning rate scheduling
- **Embedding visualisation** callbacks using UMAP
- **Weights & Biases** integration for experiment tracking

#### Running the final training loop

`full_training.py` trains three variants (different learning rates) and selects the best by weighted BCE/MCC score:

```bash
python src/full_training.py
```

#### Running the hyperparameter grid search

`gridsearch_training.py` sweeps over datasets x learning rates x LR schedulers x alpha modes:

```bash
python src/gridsearch_training.py
```

#### Key training arguments (via `training_setup.py` CLI)

| Argument | Default | Description |
|---|---|---|
| `--data_input_path` | required | Path to training JSONL |
| `--validation_input_path` | optional | Path to validation JSONL |
| `--output_path` | required | Directory for model checkpoints and saved models |
| `--model_name` | required | Name used for saving and W&B logging |
| `--save_model` | `False` | Flag to save the final model |
| `--learning_rate` | `1e-5` | Initial learning rate |
| `--batch_size` | `64` | Training batch size |
| `--subset` | `None` | Train on a fixed-size random subset |

---

### Step 5 — Inference

**Notebook:** `nbs/5_inference/Blame_inference.ipynb`  
**Script:** `src/blame_detection.py`

The trained BlameBERT model is loaded via the `BlameDetector` class and run on all Folketing debate sentences. Results are written back to JSONL with `prediction` (0/1) and `confidence` fields. The `danish_minister_party_pipeline.py` script enriches records with the correct party affiliation for each minister based on the debate date, accounting for ministers who changed parties over time (e.g., Lars Løkke Rasmussen switching from Venstre to Moderaterne).

```python
from blame_detection import BlameDetector

detector = BlameDetector(
    model_path="path/to/saved/model",  # or "Lundsfryd/blameBERT"
    max_length=512,
    batch_size=64
)

# Run on a JSONL file and save results
detector.predict_from_jsonl_to_file(
    input_path="data/inference/inference_data.jsonl",
    output_path="data/inference/predictions.jsonl"
)

# Run validation against gold labels
metrics = detector.run_validation(
    validation_path="data/training_data/validation_set/validation_set.jsonl",
    report_output_path="validation_report.txt"
)
print(metrics)
```

`BlameDetector` also supports single-text and batch prediction:

```python
# Single prediction
label, confidence, probs = detector.predict("Regeringen har svigtet borgerne.")

# Batch prediction
labels, confidences = detector.run_batch_prediction(list_of_texts)
```

---

## Source Scripts (`src/`)

| Script | Description |
|---|---|
| `sentence_segmentation.py` | Sentence-segments Danish parliamentary paragraphs using DaCy |
| `machine_trans.py` | Translates Danish JSONL text fields to English using MarianMT |
| `PBD.py` | Zero-shot preliminary blame detection using NLI with multiple hypothesis templates |
| `template_manipulation.py` | Utilities for processing and aggregating hypothesis template results |
| `json_formatting.py` | Label Studio I/O conversion and inter-annotator agreement computation |
| `danish_minister_party_pipeline.py` | Maps Danish ministers to their party affiliation by date across three cabinets (2016-present) |
| `blame_detection.py` | `BlameDetector` class: batch inference, JSONL I/O, validation, metrics |
| `training_setup.py` | Core BERT/LoRA fine-tuning logic |
| `training_pipeline_with_viz.py` | Extended training pipeline with UMAP embedding callbacks |
| `full_training.py` | Final training loop over multiple learning rates with automatic best-model selection |
| `gridsearch_training.py` | Full hyperparameter sweep (datasets x LR x scheduler x alpha mode) |
| `subset_training_pipe.py` | Training on controlled data subsets |
| `embedding_viz.py` | UMAP-based embedding visualisation and W&B logging callbacks |
| `plot_embeddings.py` | Standalone embedding plot generation |
| `xml_extract.py` | Extraction and parsing of Folketing raw XML transcripts |
| `wandb_api_test.py` | Weights & Biases connection test |

---

## Analysis

Statistical analysis is performed in R and Python after inference is complete.

### R Analysis (`analysis/r/`)

- **`blame_analysis.Rmd`** - Primary analysis using Generalised Linear Mixed-Effects Models (GLMMs via `glmmTMB`). Models blame counts over time, controlling for party and cabinet period. Includes overdispersion checks, residual diagnostics, and post-hoc comparisons using `emmeans`.
- **`leaderboard_party.Rmd`** - Party-level aggregation and ranking of blame attribution rates.

Both reports are pre-rendered as HTML files.

### Python Analysis (`analysis/python/`)

- **`power_analysis.ipynb`** - Statistical power analysis for the annotation study.
- **`inference_data.ipynb`** - Exploratory analysis of the raw inference output.

---

## Data

| Path | Description |
|---|---|
| `data/raw_data/Corp_Folketing_V2.csv` | Raw Folketing parliamentary corpus |
| `data/raw_data/danish_govs.csv` | Metadata on Danish governments and cabinet compositions |
| `data/training_data/validation_set/validation_set.jsonl` | Final gold-labelled validation set (blame / not blame) |
| `data/inference/inference_data.jsonl` | Processed sentences ready for inference |

Each JSONL record contains fields for `date`, `speaker`, `party`, `paragraph_nr`, `sentence_nr`, and `text`, with additional fields added at each pipeline stage (`translated_text`, `Hyp_N_blame`, `prediction`, `confidence`).

---

## Requirements & Installation

### Python

Install the main dependencies:

```bash
pip install -r nbs/requirements.txt
```

For the spaCy/DaCy sentence segmentation model:

```bash
python -m spacy download da_dacy_large_trf
```

### R

The R analysis uses the following packages (installed automatically via `pacman`):

```r
tidyverse, lubridate, zoo, glmmTMB, DHARMa, emmeans, sjPlot
```

### Hardware

A CUDA-capable GPU is strongly recommended for:
- Machine translation (MarianMT)
- Preliminary Blame Detection (NLI inference)
- BERT fine-tuning

All scripts automatically detect and use GPU if available, falling back to CPU.

---

## Usage Examples

**Running the full pipeline end-to-end:**

```bash
# 1. Segment sentences
python src/sentence_segmentation.py \
  --inpath_to_csv data/raw_data/Corp_Folketing_V2.csv \
  --outpath_to_jsonl data/training_data/sentences.jsonl

# 2. Translate to English
python src/machine_trans.py \
  --input_path_jsonl data/training_data/sentences.jsonl \
  --output_path_jsonl data/training_data/sentences_translated.jsonl

# 3. Preliminary blame detection
python src/PBD.py \
  --input_path_jsonl data/training_data/sentences_translated.jsonl \
  --output_path_jsonl data/training_data/pbd_output.jsonl \
  --hyp_temp_path nbs/2_machine_translation_and_PBD/hypothesis_templates.txt

# 4. Train BlameBERT
python src/full_training.py

# 5. Run inference with trained model
python -c "
from src.blame_detection import BlameDetector
d = BlameDetector('Lundsfryd/blameBERT', batch_size=64)
d.predict_from_jsonl_to_file('data/inference/inference_data.jsonl', 'data/inference/predictions.jsonl')
"
```

**Using the pre-trained model directly from Hugging Face:**

```python
from src.blame_detection import BlameDetector

detector = BlameDetector(model_path="Lundsfryd/blameBERT", batch_size=32)
label, confidence, probs = detector.predict("Regeringen er ansvarlig for denne krise.")
print(f"Blame: {label}, Confidence: {confidence:.2f}")
```
