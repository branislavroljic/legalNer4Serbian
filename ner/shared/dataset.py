"""
Shared dataset utilities for Serbian Legal NER project.

This module contains classes and functions for creating NER datasets,
tokenization, and sequence analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset


class NERDataset:
    """Dataset class for NER training"""

    def __init__(self, bio_examples: List[Dict]):
        self.examples = bio_examples
        self.label_to_id = self._create_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from labels to IDs"""
        all_labels = set(['O'])  # Start with 'O' label

        for example in self.examples:
            all_labels.update(example['labels'])

        # Sort labels to ensure consistent ordering
        sorted_labels = sorted(list(all_labels))
        return {label: idx for idx, label in enumerate(sorted_labels)}

    def encode_labels(self, labels: List[str]) -> List[int]:
        """Convert labels to IDs"""
        return [self.label_to_id[label] for label in labels]

    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """Convert IDs back to labels"""
        return [self.id_to_label[label_id] for label_id in label_ids]

    def get_num_labels(self) -> int:
        """Get number of unique labels"""
        return len(self.label_to_id)

    def prepare_for_training(self) -> List[Dict]:
        """Prepare examples for training"""
        prepared_examples = []

        for example in self.examples:
            prepared_examples.append({
                'tokens': example['tokens'],
                'labels': example['labels'],
                'text': example.get('text', '')
            })

        return prepared_examples

    def get_label_statistics(self) -> Dict:
        """Get statistics about label distribution"""
        label_counts = {}
        total_tokens = 0
        
        for example in self.examples:
            for label in example['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
                total_tokens += 1
        
        # Calculate entity type counts (B- and I- combined)
        entity_counts = {}
        for label, count in label_counts.items():
            if label != 'O':
                entity_type = label.split('-')[1] if '-' in label else label
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + count
        
        return {
            'label_counts': label_counts,
            'entity_counts': entity_counts,
            'total_tokens': total_tokens,
            'num_examples': len(self.examples)
        }


def split_dataset(examples: List[Dict], test_size: float = 0.2, val_size: float = 0.1, 
                 random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train, validation, and test sets"""
    
    # First split: separate test set
    train_val_examples, test_examples = train_test_split(
        examples, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    # Adjust val_size to account for the reduced dataset size
    adjusted_val_size = val_size / (1 - test_size)
    train_examples, val_examples = train_test_split(
        train_val_examples, test_size=adjusted_val_size, random_state=random_state
    )
    
    print(f"📊 Dataset split:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    return train_examples, val_examples, test_examples


def tokenize_and_align_labels_with_sliding_window(
    examples: List[Dict], tokenizer: AutoTokenizer, label_to_id: Dict[str, int],
    max_length: int = 512, stride: int = 128
) -> List[Dict]:
    """
    Tokenize text and align labels with subword tokens using sliding windows.

    Uses HuggingFace tokenizer's built-in sliding window support via
    return_overflowing_tokens=True and stride parameters for cleaner,
    more reliable implementation.

    Args:
        examples: List of examples with 'tokens' and 'labels'
        tokenizer: HuggingFace tokenizer
        label_to_id: Mapping from labels to IDs
        max_length: Maximum sequence length (default: 512)
        stride: Overlap between windows (default: 128)

    Returns:
        List of tokenized chunks with input_ids, attention_mask, and labels
    """
    tokenized_inputs = []

    for example in examples:
        tokens = example['tokens']
        labels = example['labels']

        # Use built-in tokenizer with sliding window support
        # is_split_into_words=True tells tokenizer that input is pre-tokenized
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding='max_length',
            return_overflowing_tokens=True,  # Creates multiple chunks for long sequences
            return_offsets_mapping=False,    # We don't need character offsets
        )

        # Process each chunk (window) created by the tokenizer
        num_chunks = len(tokenized['input_ids'])

        for chunk_idx in range(num_chunks):
            # Get word_ids for this chunk to align labels
            word_ids = tokenized.word_ids(batch_index=chunk_idx)

            # Align labels with subword tokens
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD) get -100 (ignored in loss)
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word gets the original label
                    aligned_labels.append(label_to_id[labels[word_idx]])
                else:
                    # Subsequent subwords of the same word get -100 (ignored)
                    aligned_labels.append(-100)

                previous_word_idx = word_idx

            # Add this chunk to results
            tokenized_inputs.append({
                'input_ids': tokenized['input_ids'][chunk_idx],
                'attention_mask': tokenized['attention_mask'][chunk_idx],
                'labels': aligned_labels
            })

    return tokenized_inputs


def analyze_sequence_lengths(examples: List[Dict], tokenizer: AutoTokenizer) -> Tuple[Dict, List[int]]:
    """
    Analyze the distribution of sequence lengths after WordPiece tokenization.

    Args:
        examples: List of examples with 'tokens' and 'labels'
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (statistics_dict, list_of_lengths)
    """
    lengths = []

    for example in examples:
        tokens = example['tokens']
        
        # Tokenize and count subword tokens
        subword_count = 0
        for token in tokens:
            subword_tokens = tokenizer.tokenize(token)
            subword_count += len(subword_tokens)
        
        # Add 2 for [CLS] and [SEP] tokens
        total_length = subword_count + 2
        lengths.append(total_length)

    # Calculate statistics
    lengths_array = np.array(lengths)
    stats = {
        'mean': float(np.mean(lengths_array)),
        'median': float(np.median(lengths_array)),
        'std': float(np.std(lengths_array)),
        'min': int(np.min(lengths_array)),
        'max': int(np.max(lengths_array)),
        'percentile_95': float(np.percentile(lengths_array, 95)),
        'percentile_99': float(np.percentile(lengths_array, 99)),
        'over_512': int(np.sum(lengths_array > 512)),
        'over_256': int(np.sum(lengths_array > 256)),
        'total_sequences': len(lengths)
    }

    return stats, lengths


def print_sequence_analysis(examples: List[Dict], tokenizer: AutoTokenizer):
    """
    Print detailed analysis of sequence lengths.
    """
    stats, lengths = analyze_sequence_lengths(examples, tokenizer)

    print(f"📏 Sequence Length Analysis:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Mean length: {stats['mean']:.1f}")
    print(f"  Median length: {stats['median']:.1f}")
    print(f"  Std deviation: {stats['std']:.1f}")
    print(f"  Min length: {stats['min']}")
    print(f"  Max length: {stats['max']}")
    print(f"  95th percentile: {stats['percentile_95']:.1f}")
    print(f"  99th percentile: {stats['percentile_99']:.1f}")
    print(f"  Sequences > 512 tokens: {stats['over_512']} ({stats['over_512']/stats['total_sequences']*100:.1f}%)")
    print(f"  Sequences > 256 tokens: {stats['over_256']} ({stats['over_256']/stats['total_sequences']*100:.1f}%)")


# Legacy function for backward compatibility
def tokenize_and_align_labels(examples: List[Dict], tokenizer: AutoTokenizer,
                            label_to_id: Dict[str, int], max_length: int = 512) -> List[Dict]:
    """
    Legacy tokenization function - now calls the sliding window version.
    Kept for backward compatibility.

    Note: Uses default stride of 128 tokens for sliding windows.
    """
    return tokenize_and_align_labels_with_sliding_window(
        examples, tokenizer, label_to_id, max_length, stride=128
    )


def create_huggingface_datasets(train_tokenized: List[Dict], val_tokenized: List[Dict], 
                               test_tokenized: List[Dict]) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """Create HuggingFace datasets from tokenized data"""
    train_dataset = HFDataset.from_list(train_tokenized)
    val_dataset = HFDataset.from_list(val_tokenized)
    test_dataset = HFDataset.from_list(test_tokenized)
    
    print(f"📦 Created HuggingFace datasets:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    return train_dataset, val_dataset, test_dataset
