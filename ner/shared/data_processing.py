"""
Shared data processing utilities for Serbian Legal NER project.

This module contains classes and functions for loading, preprocessing, and converting
LabelStudio annotations to BIO format for NER training.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class LabelStudioToBIOConverter:
    """Convert LabelStudio annotations to BIO format"""

    def __init__(
        self,
        judgments_dir: str = None,
        labelstudio_files_dir: str = None
    ):
        self.judgments_dir = judgments_dir
        self.labelstudio_files_dir = labelstudio_files_dir
        self.entity_types = set()

    def load_text_file(self, filename: str) -> Optional[str]:
        """Load text content from LabelStudio files or judgment files"""

        # Extract the actual filename from the path
        if "/" in filename:
            actual_filename = filename.split("/")[-1]  # Get last part after /
        else:
            actual_filename = filename

        # Try loading from LabelStudio files directory first
        if self.labelstudio_files_dir:
            labelstudio_path = Path(self.labelstudio_files_dir) / actual_filename
            if labelstudio_path.exists():
                try:
                    with open(labelstudio_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"Error reading from LabelStudio files: {e}")

        # Try loading from judgments directory
        if self.judgments_dir:
            judgment_path = Path(self.judgments_dir) / actual_filename
            if judgment_path.exists():
                try:
                    with open(judgment_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"Error reading from judgments: {e}")

        print(f"Warning: Could not find file {actual_filename}")
        return None

    def convert_to_bio(self, labelstudio_data: List[Dict]) -> List[Dict]:
        """Convert LabelStudio data to BIO format"""
        bio_examples = []

        for item in labelstudio_data:
            # Get text content from LabelStudio data structure
            file_path = item.get("file_upload", "")

            text_content = self.load_text_file(file_path)
            annotations = item.get("annotations", [])

            if text_content and annotations:
                for annotation in annotations:
                    bio_example = self._create_bio_example(text_content, annotation.get("result", []))
                    if bio_example:
                        bio_examples.append(bio_example)

        return bio_examples

    def _create_bio_example(self, text: str, annotations: List[Dict]) -> Optional[Dict]:
        """Create a BIO example from text and annotations"""
        # Example:
        # sudija Babovic Dragan uz ucesce namjestenika suda Dragovic Katarine kao zapisnica
        # ["sudija" "Babovic" "Dragan" "uz" "ucesce" "namjestenika" "suda" "Dragovic" "Katarine" "kao" "zapisnica"]
        # labels = ["O", "B-JUDGE", "I-JUDGE", "O", "O", "O", "O", "B-REGISTRAR", "I-REGISTRAR", "O", "O"]
        tokens = text.split()
        labels = ["O"] * len(tokens)

        # Create character to token mapping
        char_to_token = {}
        current_pos = 0
        for token_idx, token in enumerate(tokens):
            # Find the token in the text starting from current_pos
            token_start = text.find(token, current_pos)
            if token_start != -1:
                for char_idx in range(token_start, token_start + len(token)):
                    char_to_token[char_idx] = token_idx
                current_pos = token_start + len(token)

        # Process annotations
        for annotation in annotations:
            if annotation.get("type") == "labels":
                value = annotation["value"]
                start = value["start"]
                end = value["end"]
                entity_labels = value["labels"]

                # Find tokens that overlap with this annotation
                overlapping_tokens = set()
                for char_idx in range(start, end):
                    if char_idx in char_to_token:
                        overlapping_tokens.add(char_to_token[char_idx])

                # Apply BIO labeling
                if overlapping_tokens:
                    sorted_tokens = sorted(overlapping_tokens)
                    for i, token_idx in enumerate(sorted_tokens):
                        for entity_label in entity_labels:
                            self.entity_types.add(entity_label)
                            if i == 0:
                                labels[token_idx] = f"B-{entity_label}"
                            else:
                                labels[token_idx] = f"I-{entity_label}"
                            break  # Use only the first label if multiple

        return {
            "tokens": tokens,
            "labels": labels,
            "text": text
        }


def load_labelstudio_data(json_path: str) -> List[Dict]:
    """Load LabelStudio annotations from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} annotated documents from {json_path}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: {json_path} not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return []


def analyze_labelstudio_data(data: List[Dict]) -> Dict:
    """Analyze the structure of LabelStudio annotations"""
    total_annotations = 0
    entity_counts = {}

    for item in data:
        annotations = item.get("annotations", [])
        total_annotations += len(annotations)

        for annotation in annotations:
            result = annotation.get("result", [])
            for res in result:
                if res.get("type") == "labels":
                    labels = res["value"]["labels"]
                    for label in labels:
                        entity_counts[label] = entity_counts.get(label, 0) + 1

    analysis = {
        "total_documents": len(data),
        "total_annotations": total_annotations,
        "entity_counts": entity_counts,
        "unique_entities": len(entity_counts)
    }

    print(f"📊 Analysis Results:")
    print(f"Total documents: {analysis['total_documents']}")
    print(f"Total annotations: {analysis['total_annotations']}")
    print(f"Unique entity types: {analysis['unique_entities']}")
    print(f"\nEntity distribution:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity}: {count}")

    return analysis


def load_mlm_documents(data_dir: str) -> List[str]:
    """Load all text documents for MLM pretraining"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ MLM data directory not found: {data_dir}")
        return documents
    
    # Load all .txt files
    txt_files = list(data_path.glob("*.txt"))
    print(f"📂 Found {len(txt_files)} text files in {data_dir}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty documents
                    documents.append(content)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    print(f"✅ Loaded {len(documents)} documents for MLM training")
    return documents


def validate_bio_examples(bio_examples: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Validate BIO examples and return statistics"""
    valid_examples = []
    stats = {
        "total_examples": len(bio_examples),
        "valid_examples": 0,
        "invalid_examples": 0,
        "empty_examples": 0,
        "entity_counts": {}
    }
    
    for example in bio_examples:
        tokens = example.get("tokens", [])
        labels = example.get("labels", [])
        
        # Check if example is valid
        if not tokens or not labels:
            stats["empty_examples"] += 1
            continue
            
        if len(tokens) != len(labels):
            stats["invalid_examples"] += 1
            continue
            
        # Count entities
        for label in labels:
            if label != "O":
                entity_type = label.split("-")[1] if "-" in label else label
                stats["entity_counts"][entity_type] = stats["entity_counts"].get(entity_type, 0) + 1
        
        valid_examples.append(example)
        stats["valid_examples"] += 1
    
    print(f"📊 BIO Validation Results:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Valid examples: {stats['valid_examples']}")
    print(f"Invalid examples: {stats['invalid_examples']}")
    print(f"Empty examples: {stats['empty_examples']}")
    
    return valid_examples, stats
