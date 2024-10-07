import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
import spacy
nlp = spacy.load('en_core_web_sm')

import os
import json
from collections import defaultdict
from PIL import Image


import os
import json
from collections import defaultdict
from PIL import Image

class CustomAzureDataset:
    def __init__(self, data_dir=""):
        self.data_dir = data_dir
        self.splits = defaultdict(list)  # For train/test split data
    
    def load_image(self, image_path):
        """Load an image from a given path."""
        image = Image.open(image_path)
        return image

    def parse_annotation(self, annotation_path):
        """Parse the JSON annotation file."""
        with open(annotation_path, "r", encoding="utf8") as f:
            data = json.load(f)
        return data

    def load_dataset(self, filepath):
        """Load and store each sample in the dataset."""
        ann_dir = os.path.join(filepath, "azure_results")
        img_dir = os.path.join(filepath, "images")

        samples = []
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            boxes = []
            line_boxes = [] 
            handwritings = []
            

            # Load annotation
            file_path = os.path.join(ann_dir, file)
            data = self.parse_annotation(file_path)

            # Load corresponding image
            image_file = file.replace("json", "png")
            image_path = os.path.join(img_dir, image_file)
            image = self.load_image(image_path)

            # Extract tokens, boxes, and NER tags from annotation
            for item in data["text_lines"]:
                words = item["words"]
                handwriting = item['handwriting']
                line_bbox = item['line_bbox']

                if len(words) == 0:
                    continue
                
                for word in words:
                    tokens.append(word['text'])
                    boxes.append(word['bbox'])
                    handwritings.append(handwriting)
                    line_boxes.append(line_bbox)
                    
                
                
            

            assert len(tokens) == len(boxes) == len(handwritings) == len(line_boxes) , "Lengths of ner_tags, tokens, and boxes must be equal."
            samples.append({
                "id": str(guid),
                "tokens": tokens,
                'line_boxes': line_boxes,
                "bboxes": boxes,
                'handwritings': handwritings,
                "image": image,
                'image_name':file
            })

        return samples

    def split_generators(self):
        """Return train and test splits."""
        train_dir = os.path.join(self.data_dir, "training_data")
        test_dir = os.path.join(self.data_dir, "testing_data")

        # Load train and test data
        self.splits["train"] = self.load_dataset(train_dir)
        self.splits["test"] = self.load_dataset(test_dir)

    def __repr__(self):
        """Customize the printed representation of the dataset."""
        train_size = len(self.splits["train"])
        test_size = len(self.splits["test"])

        return (
            f"CustomFunsdDataset:\n"
            f"DatasetDict({{\n"
            f"    train: Dataset({{features: ['id', 'tokens', 'line_boxes', 'bboxes','handwritings' ,'image','image_name'], num_rows: {train_size}}}),\n"
            f"    test: Dataset({{features: ['id', 'tokens', 'line_boxes', 'bboxes', 'handwritings','image','image_name'], num_rows: {test_size}}})\n"
            f"}})"
        )

    def __getitem__(self, split):
        """Allow access to train or test splits like dataset['train']."""
        return self.splits[split]


