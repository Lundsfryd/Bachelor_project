#import stuff

import random
import spacy
import pandas as pd

class BlameAnnotater(object):

    def __init__(self, path):

        self.data = self.read_data(path)
        self.nlp = spacy.load("da_core_news_sm")

        return


    def read_data(self, path):

        df = pd.read_csv(path)
        return df

    
    def shuffler(self):
        # Example: annotation_data is your original DataFrame
        paragraphs = self.data['text'].tolist()
        original_indices = list(self.data.index)  # keep original index for para_id

        # Shuffle paragraphs (zip together with original index)
        zipped = list(zip(original_indices, paragraphs))
        random.shuffle(zipped)

        # Unzip
        self.shuffled_indices, self.shuffled_paragraphs = zip(*zipped)
        return

    
    def split_paragraph(self, paragraph: str):
        doc = self.nlp(paragraph)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences


    def build_samples(self,paragraph):
        sentences = self.split_paragraph(paragraph)
        samples = []
        
        for i, sent in enumerate(sentences):
            prev_sent = sentences[i-1] if i > 0 else ""
            next_sent = sentences[i+1] if i < len(sentences)-1 else ""

            sent_id = i

            # Three-part input with [SEP] handled by tokenizer later
            samples.append({
                "prev": prev_sent,
                "target": sent,
                "next": next_sent
            })
            
        return samples


    def annotate_sentence(self, sentence, para_id, sent_id):
        """
        Annotate a sentence interactively.
        
        Returns:
            label (int) or None if escaped
        """
        print(f"\nPara {para_id}, Sent {sent_id}:")
        print(f"\"{sentence}\"")
        
        while True:
            user_input = input("Label this sentence (0 = no blame, 1 = blame, 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                return None  # escape
            elif user_input in ['0', '1']:
                return int(user_input)
            else:
                print("Invalid input. Please enter 0, 1, or 'q'.")
        
    
    def label_sentences(self):

        labelled_sentences = []

        for para_id, para in zip(self.shuffled_indices, self.shuffled_paragraphs):
            samples = self.build_samples(para)  # build prev/target/next windows

            for sent_idx, s in enumerate(samples):
                # annotate interactively
                label = self.annotate_sentence(s['target'], para_id, sent_idx)
                if label is None:  # user pressed 'q'
                    print("Annotation interrupted. Saving progress...")
                    break
                
                labelled_sentences.append({
                    "para_id": para_id,   # original index
                    "sent_id": sent_idx,
                    "prev": s['prev'],
                    "target": s['target'],
                    "next": s['next'],
                    "label": label
                })
            else:
                continue  # only if inner loop wasn't broken
            break  # breaks outer loop if user quits

        return labelled_sentences


    def main(self):

        self.shuffler()

        annotated_data = self.label_sentences()

        #save data
        data = pd.DataFrame(annotated_data)
        data.to_csv("/work/MarkusLundsfrydJensen#1865/Bachelor_project/annotated_sentences.csv")

        return




BA = BlameAnnotater("/work/MarkusLundsfrydJensen#1865/Bachelor_project/annotation_data.csv")
BA.main()
    