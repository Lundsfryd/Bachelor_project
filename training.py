import torch
import os
import transformers
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes
import accelerate
import datasets
#import scikit-learn
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import Conv1D, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

## DEFINING FUNCTIONS 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro') # Macro is better suited for imbalanced data
    }

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

## CHUNKING MODELS TO ALL AVAILABLE GPUS FOR QLORA ADAPT
local_rank = int(os.environ.get("LOCAL_RANK", 0))

## SPLITTING AMONG GPUS
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")

## MODEL CALL IN QLORA ADAPTATION

model_name = "jhu-clsp/mmBERT-base"

quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True, # 4 bit precision
                                         bnb_4bit_compute_dtype=torch.bfloat16, # B-float 16
                                         bnb_4bit_quant_type="nf4", # Quantization nf4 data type
                                         bnb_4bit_use_double_quant=True,
                                         )

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map={"": torch.cuda.current_device()},  # map to the current GPU
    # recommended bnb args:
    #device_map="auto",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=16,
    lora_dropout=0.05, # Default drop-out rate
    target_modules=["Wqkv"],  # Fine-tuning the attention layer specifically
)

lora_model = get_peft_model(model, lora_config)

## SETTING TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=8e-4, # Learning rate copied from mmBERT paper as they found this to perform best
    num_train_epochs=3,
    per_device_train_batch_size=8,
    auto_find_batch_size=True, # Allows for auto adjusting of batch to avoid OOM
    gradient_accumulation_steps=12,  # Simulate larger batch size

    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision
    dataloader_pin_memory=False,
    remove_unused_columns=True, # Avoiding manual handling of residual text columns
    max_grad_norm=1.0,
)

## LOADING AND PROCESSING DATA

dataframe = pd.read_json("/work/RuneEgeskovTrust#9638/Bachelor/training_data/subset_cleaned_training_data.json")

val_dataframe = pd.read_json("/work/RuneEgeskovTrust#9638/Bachelor/Bachelor_project/Model_data/validation_set.json")

val_dataframe = val_dataframe[['preceding_sentence', 'text', 'succeeding_sent', 'label']]

dataframe = dataframe[['preceding_sentence', 'text', 'succeeding_sent', 'label']]

val_dataset = Dataset.from_pandas(val_dataframe)

dataset = Dataset.from_pandas(dataframe)

tokenized_val = val_dataset.map(tokenize_function)

tokenized_dataset = dataset.map(tokenize_function)

## TRAINING

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val,#split_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

## EVALUATING AND SAVING TO FILE
eval_results = trainer.evaluate()
with open("/work/RuneEgeskovTrust#9638/Bachelor/Evalresult5templates.txt", "w") as f:
    f.write(eval_results)