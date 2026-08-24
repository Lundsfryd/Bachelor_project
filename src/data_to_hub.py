import os
import json
import pandas as pd
from pathlib import Path
from datasets import DatasetDict, Dataset, Value, Features

parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(root_dir, "data")

test_data_path = os.path.join(data_dir, "training_data", "validation_set", "validation_set.jsonl")

training_data_path = Path(os.path.join(data_dir, "training_data", "5_agreement.jsonl" ))

inference_path = Path(os.path.join(data_dir, "inference", "inference_data.jsonl"))

def align_dataset_dict(dataset_dict: DatasetDict) -> DatasetDict:
    # Collect all unique features, preferring non-null types
    all_features = {}
    for split, ds in dataset_dict.items():
        for col, dtype in ds.features.items():
            if col not in all_features or str(all_features[col]) == "Value('null')":
                all_features[col] = dtype

    aligned = {}
    for split, ds in dataset_dict.items():
        missing = {col: dtype for col, dtype in all_features.items() if col not in ds.features}
        for col, dtype in missing.items():
            ds = ds.add_column(col, [None] * len(ds))
        
        # Cast to the unified feature schema to fix Value('null') columns
        ds = ds.select_columns(list(all_features.keys()))
        ds = ds.cast(Features(all_features))
        aligned[split] = ds

    return DatasetDict(aligned)

def read_jsonl(file_path):
    """Read a jsonl file and return a list of records."""
    print(f'\n\n#### Reading "{file_path}" ####\n\n')
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

test_data_records = read_jsonl(test_data_path)
test_data = pd.DataFrame(test_data_records)
test_data = test_data[["text", "label", "speaker","date","party", "paragraph_nr", "sentence_nr"]]
test_data.rename(columns={'label': 'labels'}, inplace=True)
test_dataset = Dataset.from_pandas(test_data)

data_records = read_jsonl(training_data_path)
data = pd.DataFrame(data_records)
data = data[["text", "label", "speaker","date","party", "paragraph_nr", "sentence_nr"]]
data.rename(columns={'label': 'labels'}, inplace=True)
dataset = Dataset.from_pandas(data)

print(dataset)
inference_records = read_jsonl(inference_path)
inf_data = pd.DataFrame(inference_records)
inf_dataset = Dataset.from_pandas(inf_data)

# 90/10 train/eval split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = dataset["test"]
train_dataset = dataset["train"]

dataset["validation"] = eval_dataset
dataset["train"] = train_dataset
dataset["test"] = test_dataset
dataset["inference"] = inf_dataset





dataset = align_dataset_dict(dataset)
print(dataset)

dataset.push_to_hub("runetrust/blame-folketinget-dk")
print("pushed to hub")