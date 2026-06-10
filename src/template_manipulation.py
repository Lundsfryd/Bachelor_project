#import modules
import json
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import random


class TemplateManipulation(object):

    """
    This class serves two purposes after PBD has been implemented.

    1) It should create a number of datasets equivalent to the number 
        of hypothesis templates passed for evaluation of differentiated 
        agreement, i.e. at least five templates agreee on entailment to 
        classify as such, at least four templates agree on entailment... 
        and so on.

    2) Compute hypothesis-specific stats regarding amount of labels as true.
    
    
    """

    # --------------------------------------------------------------------#

    def __init__(self, input_path, output_path, validation_size = 500, clean_keys = []):

        self.input_path = input_path
        self.output_path = output_path
        self.outdir = os.path.dirname(output_path)
        self.validation_size = validation_size
        self.clean_keys = set(clean_keys)
        self.validation_path = os.path.join(self.outdir,
                                            "validation_set.jsonl")

        self.setup()

        return
    
    def setup(self):
        os.makedirs(os.path.dirname(self.validation_path), exist_ok=True)

        #detect n hypothesis entries in data
        self.detect_n_hypotheses()

        #add a key denoting how many hypothesis labelled true
        self.add_n_hyp_entail()
    
    # --------------------------------------------------------------------#

    def detect_n_hypotheses(self):
        """Detect number of Hyp_*_blame fields from the first non-empty line."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    first_record = json.loads(line)
                    self.n_hyp = sum(
                        1 for k in first_record
                        if k.startswith("Hyp_") and k.endswith("_blame")
                    )
                    break
            else:
                raise ValueError("Input file is empty")
            
            print(f"number of hypothesis templates detected: {self.n_hyp}...\n")
            return self.n_hyp
        
    # --------------------------------------------------------------------#
    
    def add_n_hyp_entail(self):

        self.blame_keys = [f"Hyp_{i}_blame" for i in range(1, self.n_hyp + 1)]

        with open(self.input_path, "r", encoding="utf-8") as fin, \
            open(self.output_path, "w", encoding="utf-8") as fout:


            for line in tqdm(fin, desc= f"Adding key 'n_hyp entail' to: {self.output_path}..."):
                if not line.strip():
                    continue

                record = json.loads(line)

                record["n_hyp_entail"] = sum(
                    record.get(k, 0) for k in self.blame_keys
                )

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # --------------------------------------------------------------------#

    def setup_writers(self):

        writers = {}
        print("Setting up writers for labeling by level of agreement...\n")
        for threshold in range(1, self.n_hyp + 1):
            fname = (
                f"{threshold}_agreement.jsonl"
                if threshold == self.n_hyp
                else f"{threshold}_{self.n_hyp}_agreement.jsonl"
            )
            writers[threshold] = open(
                os.path.join(self.hyp_agree_outdir, fname),
                "w",
                encoding="utf-8"
            )
        return writers
    
    # --------------------------------------------------------------------#

    def label_writer(self, writers):

        try:
            with open(self.output_path, "r", encoding="utf-8") as fin:
                for line in tqdm(fin, desc= f"Reading input and writing output files for threshold of agreement..."):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    
                    if "n_hyp_entail" not in record:
                        raise KeyError("Missing 'n_hyp_entail' in record")
                    
                    entail = record["n_hyp_entail"]

                    for threshold, fout in writers.items():
                        record_out = record.copy()
                        record_out["label"] = int(entail >= threshold)

                        validation, record_out = self.validate_clean_record(record_out)

                        if validation == True:
                            fout.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                        else:
                            continue

        finally:
            for f in writers.values():
                f.close()
        
    # --------------------------------------------------------------------#
    def validate_clean_record(self, record):

        record = self.clean_keys_in_record(record)

        #evaluate in record is in self.validation_set

        record_id = (record.get("paragraph_nr"), record.get("sentence_nr"))
        if record_id in self.validation_ids:
            validation = False
        else:
            validation = True

        return validation, record
    
    
    def clean_keys_in_record(self, record):
        for key in self.clean_keys:
                    record.pop(key, None)  # safe if key missing

        return record
    
    def create_dir(self):


        # if the directory is not present 
        # then create it.
        self.hyp_agree_outdir = os.path.join(self.outdir,
                                     "diff_hypothesis_agreemen")
        
        print(f"Making output directory for datasets of different levels of agreement:\n{self.hyp_agree_outdir}\n")

        os.makedirs(self.hyp_agree_outdir, exist_ok=True)
        return
    
    # --------------------------------------------------------------------#

    def create_datasets(self):

        self.create_dir()

        #make validation set
        self.make_validation_set()

        # Open all output files at once
        writers = self.setup_writers()

        #write to those files
        self.label_writer(writers)

        return
    
    # --------------------------------------------------------------------#
    def read_labels(self):
        true_labels = []
        false_labels = []
        
        with open(self.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record['n_hyp_entail'] > 0:
                        true_labels.append(record)
                    else:
                        false_labels.append(record)

        return true_labels, false_labels


    def make_validation_set(self):
        """
        Create a balanced validation set from a jsonl file.
        
        Args:
            input_file: path to input jsonl file
            validation_size: desired size of the validation set
        
        Returns:
            List of records forming the validation set
        """
        #read and append true/false records from output file
        true_labels, false_labels = self.read_labels()
                
        n_true, n_false = self.evaluate_nr_labels(true_labels, false_labels)

        sampled_true = random.sample(true_labels, n_true)
        sampled_false = random.sample(false_labels, n_false)

        self.validation_set = sampled_true + sampled_false
        random.shuffle(self.validation_set)

        print(f"Validation set created with {len(self.validation_set)} records: "
            f"{n_true} true labels, {n_false} false labels.")
        
        self.write_validation_set(self.validation_set)

        self.validation_ids = {
        (r["paragraph_nr"], r["sentence_nr"])
        for r in self.validation_set
}
        
        return self.validation_set
    # --------------------------------------------------------------------#

    def evaluate_nr_labels(self,true_labels, false_labels):
        
        n_true_available = len(true_labels)
        n_false_available = len(false_labels)
        n_per_class = self.validation_size // 2

        if n_true_available < n_per_class:
            print(f"Warning: Only {n_true_available} true labels available, "
                f"which is fewer than the desired {n_per_class}. "
                f"Including all true labels and matching with false labels.")
            n_true = n_true_available
            n_false = min(n_true_available, n_false_available)
        else:
            n_true = n_per_class
            n_false = min(n_per_class, n_false_available)

        return n_true, n_false

    def write_validation_set(self, validation_set):
        """
        Write a validation set to a jsonl file.
        
        Args:
            validation_set: list of records to write
            validation_path: path to output jsonl file
        """
        with open(self.validation_path, 'w', encoding='utf-8') as f:
            for record in validation_set:
                record["label"] = 1 if record["n_hyp_entail"] > 0 else 0
                record = self.clean_keys_in_record(record)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Validation set written to {self.validation_path}")

    def compute_agreement_statistics_per_hyp(self):
        """
        Compute per-threshold agreement statistics and per-hypothesis percentages.
        """
        counts_threshold = Counter()  # for n_hyp_entail thresholds
        counts_hyp = Counter()        # for each individual hypothesis
        total = 0

        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc= "Reading lines of output file for statistics..."):
                if not line.strip():
                    continue

                record = json.loads(line)
                entail = record.get("n_hyp_entail", 0)

                total += 1

                # per-threshold counts
                for k in range(1, self.n_hyp + 1):
                    if entail >= k:
                        counts_threshold[k] += 1

                # per-hyp counts
                for key in self.blame_keys:
                    if record.get(key, 0) == 1:
                        counts_hyp[key] += 1

        # compute percentages
        pct_threshold = {
            k: 100 * counts_threshold[k] / total for k in range(1, self.n_hyp + 1)
        }

        pct_hyp = {
            key: 100 * counts_hyp[key] / total for key in self.blame_keys
        }

        return pct_threshold, pct_hyp, total
    

    def print_agreement_statistics_per_hyp(self, pct_threshold, pct_hyp, total):
        print(f"\nTotal samples: {total:,}\n")

        print("=== Per-threshold agreement percentages ===")
        for k, pct in pct_threshold.items():
            print(f"Agreement ≥ {k} hypotheses: {pct:.2f}%")

        print("\n=== Per-hypothesis percentages ===")
        for key, pct in pct_hyp.items():
            print(f"{key}: {pct:.2f}%")



    def plot_agreement_statistics_per_hyp(self, pct_threshold, pct_hyp):
        
        # per-threshold bar plot
        plt.figure(figsize=(8, 4))
        plt.bar(pct_threshold.keys(), pct_threshold.values())
        plt.xlabel("Minimum hypothesis agreement")
        plt.ylabel("Percentage of samples labelled 1 (%)")
        plt.title("Percentage of 1s by Agreement thresholds")
        plt.xticks(list(pct_threshold.keys()))
        plt.ylim(0, self.ylim)
        plt.savefig(os.path.join(self.hyp_agree_outdir, "per_threshold_bar_plot.png"))
        plt.show()

        # per-hypothesis bar plot
        plt.figure(figsize=(10, 4))
        plt.bar(pct_hyp.keys(), pct_hyp.values())
        plt.xlabel("Hypothesis key")
        plt.ylabel("Percentage of samples labeled 1 (%)")
        plt.title("Percentage of 1s per hypothesis")
        plt.ylim(0, self.ylim)
        plt.savefig(os.path.join(self.hyp_agree_outdir, "per_hypothesis_bar_plot.png"))
        plt.show()

    def run_statistics(self, ylim = 10):

        self.ylim = ylim

        pct_threshold, pct_hyp, total = self.compute_agreement_statistics_per_hyp()
        self.print_agreement_statistics_per_hyp(pct_threshold, pct_hyp, total)
        self.plot_agreement_statistics_per_hyp(pct_threshold, pct_hyp)

        return

        
    


def main():

    parser = argparse.ArgumentParser(
                    prog='Template manipulation',
                    description='Creates templates based on level of agreement. Also prodices template statistics')
    
    parser.add_argument("--input_path", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--output_path", 
                        type=Path,
                        required=True) # add argument
    
    parser.add_argument("--ylim", 
                        type=int,
                        default=10) # add argument
    
    parser.add_argument("--validation_size", 
                        type=int,
                        default=300) # add argument
    
    parser.add_argument("--clean_keys",
                        nargs="*",
                        default=[],
                        help="Keys to remove from JSONL records")


    args = parser.parse_args()

    TM = TemplateManipulation(input_path = args.input_path,
                            output_path = args.output_path,
                            validation_size=args.validation_size,
                            clean_keys=args.clean_keys)

    TM.create_datasets()
    TM.run_statistics(ylim=args.ylim)



if __name__ == "__main__":

    main()


    

    