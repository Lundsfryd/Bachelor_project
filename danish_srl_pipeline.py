#!/usr/bin/env python3
"""
Danish Semantic Role Labeling (SRL) Annotation and Training Pipeline
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import spacy
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score

# Configuration
@dataclass
class Config:
    model_name: str = "Maltehb/danish-bert-botxo"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    output_dir: str = "./danish_srl_model"
    data_dir: str = "./data"
    
    # SRL Label scheme (BIO format)
    labels: List[str] = None
    use_coreference: bool = False  # Enable coreference-aware labeling
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = [
                'O',           # Outside
                'B-A0',        # Agent (subject)
                'I-A0',        # Inside Agent
                'B-A1',        # Patient (direct object/theme)
                'I-A1',        # Inside Patient
                'B-A2',        # Recipient/Beneficiary
                'I-A2',        # Inside Recipient
                'B-A3',        # Instrument
                'I-A3',        # Inside Instrument
                'B-V',         # Verb/Predicate
                'I-V',         # Inside Verb (for multi-word verbs)
                'B-AM-TMP',    # Time adjunct
                'I-AM-TMP',    # Inside Time
                'B-AM-LOC',    # Location adjunct
                'I-AM-LOC',    # Inside Location
                'B-AM-MNR',    # Manner adjunct
                'I-AM-MNR',    # Inside Manner
                'B-AM-CAU',    # Cause adjunct
                'I-AM-CAU',    # Inside Cause
            ]
            
            # Add coreference labels if enabled
            if self.use_coreference:
                coref_labels = [
                    'B-A0-PRON',    # Pronoun acting as agent
                    'I-A0-PRON',    # Inside pronoun agent
                    'B-A1-PRON',    # Pronoun acting as patient
                    'I-A1-PRON',    # Inside pronoun patient
                    'B-A2-PRON',    # Pronoun acting as recipient
                    'I-A2-PRON',    # Inside pronoun recipient
                ]
                self.labels.extend(coref_labels)

class DanishSRLAnnotator:
    """Tool for creating and managing Danish SRL annotations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp = None
        self.load_spacy_model()
        
    def load_spacy_model(self):
        """Load Danish spaCy model for tokenization and initial processing"""
        try:
            self.nlp = spacy.load("da_core_news_sm")
            print("✓ Danish spaCy model loaded")
        except OSError:
            print("❌ Danish spaCy model not found. Install with:")
            print("python -m spacy download da_core_news_sm")
            print("Using basic tokenization as fallback")
    
    def create_annotation_template(self, text: str, output_file: str):
        """Create an annotation template from raw text"""
        sentences = []
        
        if self.nlp:
            doc = self.nlp(text)
            for sent in doc.sents:
                tokens = []
                for token in sent:
                    if not token.is_space:
                        tokens.append((token.text, 'O'))  # Default to 'O' label
                if tokens:
                    sentences.append(tokens)
        else:
            # Fallback tokenization
            for line in text.split('\n'):
                if line.strip():
                    tokens = [(token, 'O') for token in line.split() if token.strip()]
                    if tokens:
                        sentences.append(tokens)
        
        # Write to CoNLL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                for token, label in sentence:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Empty line between sentences
        
        print(f"✓ Annotation template created: {output_file}")
        print(f"  - {len(sentences)} sentences")
        print(f"  - Total tokens: {sum(len(sent) for sent in sentences)}")
    
    def validate_annotations(self, annotation_file: str) -> bool:
        """Validate annotation file format and labels"""
        valid_labels = set(self.config.labels)
        errors = []
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_num = 0
        for line in lines:
            line_num += 1
            line = line.strip()
            
            if not line:  # Empty line (sentence separator)
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                errors.append(f"Line {line_num}: Expected 2 columns, got {len(parts)}")
                continue
            
            token, label = parts
            if label not in valid_labels:
                errors.append(f"Line {line_num}: Invalid label '{label}'")
        
        if errors:
            print(f"❌ Validation failed for {annotation_file}:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            return False
        else:
            print(f"✓ Validation passed for {annotation_file}")
            return True
    
    def annotation_statistics(self, annotation_file: str):
        """Show statistics about annotations"""
        label_counts = {}
        total_tokens = 0
        total_sentences = 0
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            current_sentence_length = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence_length > 0:
                        total_sentences += 1
                        current_sentence_length = 0
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    label_counts[label] = label_counts.get(label, 0) + 1
                    total_tokens += 1
                    current_sentence_length += 1
            
            # Handle last sentence if file doesn't end with empty line
            if current_sentence_length > 0:
                total_sentences += 1
        
        print(f"\n📊 Annotation Statistics for {annotation_file}:")
        print(f"  Total sentences: {total_sentences}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average sentence length: {total_tokens / total_sentences:.1f}")
        print(f"\n  Label distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_tokens) * 100
            print(f"    {label}: {count} ({percentage:.1f}%)")
    
    def split_dataset(self, input_file: str, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15, 
                     output_dir: str = None):
        """Split annotated dataset into train/val/test sets"""
        if output_dir is None:
            output_dir = self.config.data_dir
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        # Read all sentences
        sentences = []
        current_sentence = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    current_sentence.append(line)
        
        # Handle last sentence
        if current_sentence:
            sentences.append(current_sentence)
        
        # Shuffle sentences for random split
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(sentences)
        
        # Calculate split indices
        n_sentences = len(sentences)
        train_end = int(n_sentences * train_ratio)
        val_end = train_end + int(n_sentences * val_ratio)
        
        # Split data
        train_sentences = sentences[:train_end]
        val_sentences = sentences[train_end:val_end]
        test_sentences = sentences[val_end:]
        
        # Write splits to files
        def write_sentences(sentences_list, filename):
            with open(filename, 'w', encoding='utf-8') as f:
                for sentence in sentences_list:
                    for line in sentence:
                        f.write(line + '\n')
                    f.write('\n')  # Empty line between sentences
        
        train_file = f"{output_dir}/train.conll"
        val_file = f"{output_dir}/val.conll"
        test_file = f"{output_dir}/test.conll"
        
        write_sentences(train_sentences, train_file)
        write_sentences(val_sentences, val_file)
        write_sentences(test_sentences, test_file)
        
        print(f"✓ Dataset split completed:")
        print(f"  Train: {len(train_sentences)} sentences → {train_file}")
        print(f"  Val:   {len(val_sentences)} sentences → {val_file}")
        print(f"  Test:  {len(test_sentences)} sentences → {test_file}")
        
        return train_file, val_file, test_file

class TripletExtractor:
    """Extract semantic triplets from SRL predictions with coreference resolution"""
    
    def __init__(self, config: Config):
        self.config = config
        self.label2id = {label: i for i, label in enumerate(config.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def extract_triplets_from_sentence(self, tokens: List[str], labels: List[str], 
                                     sentence_id: int = 0) -> List[Dict]:
        """Extract agent-verb-patient triplets from a single sentence"""
        triplets = []
        
        # Find spans for each role
        spans = self._extract_spans(tokens, labels)
        
        # Group by verb to form triplets
        verbs = spans.get('V', [])
        agents = spans.get('A0', [])
        patients = spans.get('A1', [])
        recipients = spans.get('A2', [])
        
        for verb in verbs:
            # Try to find the closest agent and patient to this verb
            verb_pos = verb['start']
            
            # Find closest agent
            closest_agent = self._find_closest_span(agents, verb_pos)
            # Find closest patient  
            closest_patient = self._find_closest_span(patients, verb_pos)
            
            triplet = {
                'sentence_id': sentence_id,
                'verb': verb,
                'agent': closest_agent,
                'patient': closest_patient,
                'recipient': self._find_closest_span(recipients, verb_pos) if recipients else None
            }
            triplets.append(triplet)
        
        return triplets
    
    def _extract_spans(self, tokens: List[str], labels: List[str]) -> Dict[str, List[Dict]]:
        """Extract labeled spans from BIO tags"""
        spans = {}
        current_label = None
        current_span_start = None
        current_tokens = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # Save previous span if exists
                if current_label and current_span_start is not None:
                    role = current_label.split('-')[0] if '-' in current_label else current_label
                    if role not in spans:
                        spans[role] = []
                    spans[role].append({
                        'text': ' '.join(current_tokens),
                        'start': current_span_start,
                        'end': i - 1,
                        'tokens': current_tokens.copy(),
                        'is_pronoun': 'PRON' in current_label
                    })
                
                # Start new span
                current_label = label[2:]  # Remove 'B-'
                current_span_start = i
                current_tokens = [token]
                
            elif label.startswith('I-') and current_label:
                # Continue current span
                if label[2:] == current_label:
                    current_tokens.append(token)
                else:
                    # Label mismatch - start new span
                    current_label = label[2:]
                    current_span_start = i
                    current_tokens = [token]
            
            else:  # 'O' or other
                # End current span
                if current_label and current_span_start is not None:
                    role = current_label.split('-')[0] if '-' in current_label else current_label
                    if role not in spans:
                        spans[role] = []
                    spans[role].append({
                        'text': ' '.join(current_tokens),
                        'start': current_span_start,
                        'end': i - 1,
                        'tokens': current_tokens.copy(),
                        'is_pronoun': 'PRON' in current_label
                    })
                current_label = None
                current_span_start = None
                current_tokens = []
        
        # Handle final span
        if current_label and current_span_start is not None:
            role = current_label.split('-')[0] if '-' in current_label else current_label
            if role not in spans:
                spans[role] = []
            spans[role].append({
                'text': ' '.join(current_tokens),
                'start': current_span_start,
                'end': len(tokens) - 1,
                'tokens': current_tokens.copy(),
                'is_pronoun': 'PRON' in current_label
            })
        
        return spans
    
    def _find_closest_span(self, spans: List[Dict], verb_pos: int) -> Optional[Dict]:
        """Find the span closest to the verb position"""
        if not spans:
            return None
        
        closest_span = None
        min_distance = float('inf')
        
        for span in spans:
            # Calculate distance from verb to span
            distance = min(abs(span['start'] - verb_pos), abs(span['end'] - verb_pos))
            if distance < min_distance:
                min_distance = distance
                closest_span = span
        
        return closest_span
    
    def resolve_coreferences_simple(self, triplets: List[Dict], all_sentences: List[List[str]]) -> List[Dict]:
        """Simple coreference resolution for pronouns"""
        # Danish pronouns and their likely referents
        danish_pronouns = {
            'han': 'masculine_singular',
            'hun': 'feminine_singular', 
            'den': 'neuter_singular',
            'det': 'neuter_singular',
            'de': 'plural',
            'dem': 'plural'
        }
        
        resolved_triplets = []
        
        # Keep track of recent entities for simple recency-based resolution
        recent_entities = []
        
        for triplet in triplets:
            resolved_triplet = triplet.copy()
            
            # Try to resolve agent pronoun
            if triplet['agent'] and triplet['agent']['is_pronoun']:
                agent_text = triplet['agent']['text'].lower()
                if agent_text in danish_pronouns:
                    # Find recent compatible entity
                    referent = self._find_referent(agent_text, recent_entities, danish_pronouns)
                    if referent:
                        resolved_triplet['agent']['resolved_text'] = referent
                        resolved_triplet['agent']['original_text'] = agent_text
            
            # Try to resolve patient pronoun
            if triplet['patient'] and triplet['patient']['is_pronoun']:
                patient_text = triplet['patient']['text'].lower()
                if patient_text in danish_pronouns:
                    referent = self._find_referent(patient_text, recent_entities, danish_pronouns)
                    if referent:
                        resolved_triplet['patient']['resolved_text'] = referent
                        resolved_triplet['patient']['original_text'] = patient_text
            
            # Add non-pronoun entities to recent entities list
            if triplet['agent'] and not triplet['agent']['is_pronoun']:
                recent_entities.append(triplet['agent']['text'])
            if triplet['patient'] and not triplet['patient']['is_pronoun']:
                recent_entities.append(triplet['patient']['text'])
            
            # Keep only last 10 entities
            recent_entities = recent_entities[-10:]
            
            resolved_triplets.append(resolved_triplet)
        
        return resolved_triplets
    
    def _find_referent(self, pronoun: str, recent_entities: List[str], 
                      pronoun_types: Dict[str, str]) -> Optional[str]:
        """Simple heuristic to find referent for pronoun"""
        # This is a very basic implementation
        # In practice, you'd want more sophisticated coreference resolution
        
        if not recent_entities:
            return None
        
        # For now, just return the most recent entity
        # In a real system, you'd check gender/number agreement
        return recent_entities[-1]
    
    def format_triplets_for_output(self, triplets: List[Dict]) -> str:
        """Format triplets for readable output"""
        output = []
        
        for triplet in triplets:
            agent_text = "Unknown"
            if triplet['agent']:
                if 'resolved_text' in triplet['agent']:
                    agent_text = f"{triplet['agent']['resolved_text']} ({triplet['agent']['original_text']})"
                else:
                    agent_text = triplet['agent']['text']
            
            verb_text = triplet['verb']['text'] if triplet['verb'] else "Unknown"
            
            patient_text = "Unknown"
            if triplet['patient']:
                if 'resolved_text' in triplet['patient']:
                    patient_text = f"{triplet['patient']['resolved_text']} ({triplet['patient']['original_text']})"
                else:
                    patient_text = triplet['patient']['text']
            
            triplet_str = f"({agent_text}, {verb_text}, {patient_text})"
            output.append(triplet_str)
        
        return '\n'.join(output)

class DanishSRLDataset:
    """Handle dataset loading and preprocessing for Danish SRL"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label2id = {label: i for i, label in enumerate(config.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def load_conll_file(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Load CoNLL format file"""
        sentences = []
        labels = []
        
        current_sentence = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line - end of sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    current_sentence.append(token)
                    current_labels.append(label)
        
        # Handle last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
        
        return sentences, labels
    
    def tokenize_and_align_labels(self, sentences: List[List[str]], labels: List[List[str]]):
        """Tokenize sentences and align labels with subword tokens"""
        tokenized_inputs = []
        aligned_labels = []
        
        for sentence, label_seq in zip(sentences, labels):
            # Join sentence for tokenization
            text = " ".join(sentence)
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_offsets_mapping=True,
                is_split_into_words=False
            )
            
            # Align labels
            word_ids = encoding.word_ids()
            aligned_label_seq = []
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP], [PAD])
                    aligned_label_seq.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word
                    if word_idx < len(label_seq):
                        aligned_label_seq.append(self.label2id[label_seq[word_idx]])
                    else:
                        aligned_label_seq.append(self.label2id['O'])
                else:
                    # Subsequent subwords of the same word
                    if word_idx < len(label_seq):
                        # Convert B- to I- for subsequent subwords
                        original_label = label_seq[word_idx]
                        if original_label.startswith('B-'):
                            sub_label = 'I-' + original_label[2:]
                            aligned_label_seq.append(self.label2id.get(sub_label, self.label2id['O']))
                        else:
                            aligned_label_seq.append(self.label2id[original_label])
                    else:
                        aligned_label_seq.append(self.label2id['O'])
                
                previous_word_idx = word_idx
            
            tokenized_inputs.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': aligned_label_seq
            })
        
        return tokenized_inputs
    
    def create_dataset(self, file_path: str) -> Dataset:
        """Create HuggingFace dataset from CoNLL file"""
        sentences, labels = self.load_conll_file(file_path)
        tokenized_data = self.tokenize_and_align_labels(sentences, labels)
        
        # Convert to HuggingFace dataset format
        dataset_dict = {
            'input_ids': [item['input_ids'] for item in tokenized_data],
            'attention_mask': [item['attention_mask'] for item in tokenized_data],
            'labels': [item['labels'] for item in tokenized_data]
        }
        
        return Dataset.from_dict(dataset_dict)

class DanishSRLTrainer:
    """Training pipeline for Danish SRL model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dataset_handler = DanishSRLDataset(config)
        self.model = None
        self.tokenizer = self.dataset_handler.tokenizer
    
    def load_model(self):
        """Load pre-trained Danish BERT model for token classification"""
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.labels),
            id2label=self.dataset_handler.id2label,
            label2id=self.dataset_handler.label2id
        )
        print(f"✓ Model loaded: {self.config.model_name}")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_labels = [[self.dataset_handler.id2label[l] for l in label if l != -100] 
                      for label in labels]
        true_predictions = [[self.dataset_handler.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                           for prediction, label in zip(predictions, labels)]
        
        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    def train(self, train_file: str, val_file: Optional[str] = None):
        """Train the SRL model"""
        if not self.model:
            self.load_model()
        
        # Load datasets
        train_dataset = self.dataset_handler.create_dataset(train_file)
        val_dataset = None
        if val_file:
            val_dataset = self.dataset_handler.create_dataset(val_file)
        
        print(f"✓ Training dataset: {len(train_dataset)} examples")
        if val_dataset:
            print(f"✓ Validation dataset: {len(val_dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if val_dataset else None,
        )
        
        # Train
        print("\n🚀 Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"✓ Model saved to {self.config.output_dir}")
    
    def evaluate(self, test_file: str):
        """Evaluate trained model"""
        if not self.model:
            self.load_model()
        
        test_dataset = self.dataset_handler.create_dataset(test_file)
        
        trainer = Trainer(
            model=self.model,
            eval_dataset=test_dataset,
            data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
        
        results = trainer.evaluate()
        print(f"\n📈 Test Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results

def main():
    """Example usage of the Danish SRL pipeline"""
    config = Config()
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize annotator
    annotator = DanishSRLAnnotator(config)
    
    # Example: Create annotation template from raw text
    sample_text = """
    Manden købte en bog i butikken i går.
    Hun gav ham bogen efter mødet.
    Politiet arresterede tyven på hovedbanegården.
    Lægen undersøgte patienten meget grundigt.
    """
    
    template_file = f"{config.data_dir}/template.conll"
    annotator.create_annotation_template(sample_text, template_file)
    
    print(f"\n📝 Next steps:")
    print(f"1. Edit {template_file} to add correct SRL labels")
    print(f"2. Validate your annotations:")
    print(f"   annotator.validate_annotations('{template_file}')")
    print(f"3. Create train/val/test splits")
    print(f"4. Train the model:")
    print(f"   trainer = DanishSRLTrainer(config)")
    print(f"   trainer.train('train.conll', 'val.conll')")
    
    # Show example of manual annotation
    print(f"\n💡 Example annotation (edit {template_file}):")
    print("Manden\tB-A0")
    print("købte\tB-V") 
    print("en\tB-A1")
    print("bog\tI-A1")
    print("i\tB-AM-LOC")
    print("butikken\tI-AM-LOC")
    print("i\tB-AM-TMP")
    print("går\tI-AM-TMP")
    print(".\tO")
    print()

if __name__ == "__main__":
    main()