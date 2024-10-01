import os
import json
from collections import defaultdict
from PIL import Image

class CustomFunsdDataset:
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
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        samples = []
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            boxes = []
            ner_tags = []
            line_boxes = [] 
            line_ids = []
            linkings = []

            # Load annotation
            file_path = os.path.join(ann_dir, file)
            data = self.parse_annotation(file_path)

            # Load corresponding image
            image_file = file.replace("json", "png")
            image_path = os.path.join(img_dir, image_file)
            image = self.load_image(image_path)

            # Extract tokens, boxes, and NER tags from annotation
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                
                linking = []
                if len(item['linking'])>0:
                    for x in item['linking']:
                        linking.extend([it for it in x if it != item['id']])
                        
                    if len(linking)==0: print(item['linking'],linking,file, words)
                else : linking = None

                
                line_id =  item['id']



                if len(words) == 0:
                    continue
                
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        boxes.append(w["box"])
                        line_boxes.append(item['box'])
                        linkings.append(linking)
                        line_ids.append(line_id)

                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    boxes.append(words[0]["box"])
                    line_boxes.append(item['box'])
                    linkings.append(linking)
                    line_ids.append(line_id)

                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        boxes.append(w["box"])
                        line_boxes.append(item['box'])
                        linkings.append(linking)
                        line_ids.append(line_id)

            
            label_map = {
                "O": 0,
                "B-HEADER": 1,
                "I-HEADER": 2,
                "B-QUESTION": 3,
                "I-QUESTION": 4,
                "B-ANSWER": 5,
                "I-ANSWER": 6
                 }
            ner_tags = [label_map[tag] for tag in ner_tags]

            assert len(ner_tags) == len(tokens) == len(boxes) == len(linkings) == len(line_ids) , "Lengths of ner_tags, tokens, and boxes must be equal."
            samples.append({
                "id": str(guid),
                "tokens": tokens,
                'ner_boxes':line_boxes,
                "bboxes": boxes,
                "ner_tags": ner_tags,
                'line_ids': line_ids,
                'linkings': linkings,
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
            f"    train: Dataset({{features: ['id', 'tokens', 'ner_boxes', 'bboxes', 'ner_tags','line_ids','linkings','image','image_name'], num_rows: {train_size}}}),\n"
            f"    test: Dataset({{features: ['id', 'tokens', 'ner_boxes', 'bboxes', 'ner_tags','line_ids','linkings','image','image_name'], num_rows: {test_size}}})\n"
            f"}})"
        )

    def __getitem__(self, split):
        """Allow access to train or test splits like dataset['train']."""
        return self.splits[split]


