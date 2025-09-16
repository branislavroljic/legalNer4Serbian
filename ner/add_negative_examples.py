"""
Add Negative Examples (Non-NE Sentences) to Serbian Legal NER Dataset

This script helps add sentences without named entities to improve NER performance by:
1. Reducing class imbalance
2. Improving model generalization
3. Reducing false positives
4. Better boundary detection

Usage:
    python add_negative_examples.py
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_sentences_from_judgments(judgments_dir: str, max_sentences_per_file: int = 10) -> List[str]:
    """
    Extract sentences from judgment files that don't contain named entities.
    
    Args:
        judgments_dir: Directory containing judgment text files
        max_sentences_per_file: Maximum sentences to extract per file
    
    Returns:
        List of sentences without named entities
    """
    judgments_path = Path(judgments_dir)
    all_sentences = []
    
    # Common patterns that might indicate named entities (to avoid)
    entity_patterns = [
        r'\b[A-Z]\.[A-Z]\.?\b',  # Initials like M.E., K.E.
        r'\b\d{1,2}\.\d{1,2}\.\d{4}\.?\b',  # Dates
        r'\bK\.?\s*br\.?\s*\d+',  # Case numbers
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Proper names (basic pattern)
        r'\b(?:Sud|Osnovni|Viši|Vrhovni)\s+sud\b',  # Court names
        r'\b(?:čl|član|stav)\.\s*\d+',  # Legal provisions
        r'\b\d+\s*(?:dinara|eura|KM)\b',  # Monetary amounts
    ]
    
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in entity_patterns]
    
    print(f"Scanning judgment files in {judgments_dir}...")
    
    for file_path in judgments_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Filter sentences
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip very short or very long sentences
                if len(sentence) < 20 or len(sentence) > 200:
                    continue
                
                # Skip sentences with potential named entities
                has_entities = False
                for pattern in compiled_patterns:
                    if pattern.search(sentence):
                        has_entities = True
                        break
                
                if not has_entities:
                    # Additional checks for common legal text without entities
                    if any(word in sentence.lower() for word in [
                        'krivično', 'zakon', 'postupak', 'pravilo', 'odredba',
                        'član', 'stav', 'tačka', 'paragraf', 'procedura'
                    ]):
                        clean_sentences.append(sentence)
                
                if len(clean_sentences) >= max_sentences_per_file:
                    break
            
            all_sentences.extend(clean_sentences)
            print(f"  {file_path.name}: {len(clean_sentences)} clean sentences")
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    print(f"\nTotal clean sentences extracted: {len(all_sentences)}")
    return all_sentences


def create_negative_examples(sentences: List[str], max_examples: int = 100) -> List[Dict]:
    """
    Create negative examples (all tokens labeled as 'O') from sentences.
    
    Args:
        sentences: List of sentences without named entities
        max_examples: Maximum number of examples to create
    
    Returns:
        List of examples with tokens and 'O' labels
    """
    examples = []
    
    # Shuffle sentences for variety
    random.shuffle(sentences)
    
    for sentence in sentences[:max_examples]:
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        
        # Create all 'O' labels
        labels = ['O'] * len(tokens)
        
        # Create example
        example = {
            'tokens': tokens,
            'labels': labels,
            'text': sentence,
            'source': 'negative_example'
        }
        
        examples.append(example)
    
    print(f"Created {len(examples)} negative examples")
    return examples


def augment_dataset_with_negatives(
    original_examples: List[Dict],
    negative_examples: List[Dict],
    negative_ratio: float = 0.3
) -> List[Dict]:
    """
    Augment the original dataset with negative examples.
    
    Args:
        original_examples: Original training examples
        negative_examples: Negative examples to add
        negative_ratio: Ratio of negative examples to add (0.3 = 30%)
    
    Returns:
        Augmented dataset
    """
    num_negatives_to_add = int(len(original_examples) * negative_ratio)
    num_negatives_to_add = min(num_negatives_to_add, len(negative_examples))
    
    # Randomly sample negative examples
    selected_negatives = random.sample(negative_examples, num_negatives_to_add)
    
    # Combine datasets
    augmented_examples = original_examples + selected_negatives
    
    # Shuffle the combined dataset
    random.shuffle(augmented_examples)
    
    print(f"\nDataset augmentation summary:")
    print(f"  Original examples: {len(original_examples)}")
    print(f"  Added negative examples: {len(selected_negatives)}")
    print(f"  Total augmented examples: {len(augmented_examples)}")
    print(f"  Negative ratio: {len(selected_negatives)/len(augmented_examples)*100:.1f}%")
    
    return augmented_examples


def analyze_augmented_dataset(augmented_examples: List[Dict]):
    """
    Analyze the class distribution in the augmented dataset.
    """
    total_tokens = 0
    o_tokens = 0
    entity_tokens = 0
    
    for example in augmented_examples:
        labels = example['labels']
        total_tokens += len(labels)
        
        for label in labels:
            if label == 'O':
                o_tokens += 1
            else:
                entity_tokens += 1
    
    print(f"\nAugmented dataset analysis:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  'O' tokens: {o_tokens:,} ({o_tokens/total_tokens*100:.1f}%)")
    print(f"  Entity tokens: {entity_tokens:,} ({entity_tokens/total_tokens*100:.1f}%)")
    print(f"  Imbalance ratio: {o_tokens/max(entity_tokens, 1):.1f}:1")
    
    if o_tokens / total_tokens < 0.9:
        print("  ✅ Better class balance achieved!")
    else:
        print("  ⚠️  Still high class imbalance - consider adding more negative examples")


def save_augmented_dataset(augmented_examples: List[Dict], output_path: str):
    """
    Save the augmented dataset to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\nAugmented dataset saved to: {output_path}")


def main():
    """
    Main function to demonstrate negative example augmentation.
    """
    print("="*80)
    print("🔧 ADDING NEGATIVE EXAMPLES TO SERBIAN LEGAL NER DATASET")
    print("="*80)
    
    # Configuration
    JUDGMENTS_DIR = "labelstudio_files"  # or your judgment directory
    OUTPUT_PATH = "augmented_training_data.json"
    MAX_SENTENCES_PER_FILE = 15
    MAX_NEGATIVE_EXAMPLES = 200
    NEGATIVE_RATIO = 0.4  # Add 40% negative examples
    
    print(f"\nConfiguration:")
    print(f"  Judgments directory: {JUDGMENTS_DIR}")
    print(f"  Max sentences per file: {MAX_SENTENCES_PER_FILE}")
    print(f"  Max negative examples: {MAX_NEGATIVE_EXAMPLES}")
    print(f"  Negative ratio: {NEGATIVE_RATIO}")
    
    # Step 1: Extract clean sentences
    print(f"\n" + "="*60)
    print("STEP 1: EXTRACTING CLEAN SENTENCES")
    print("="*60)
    
    clean_sentences = extract_sentences_from_judgments(
        JUDGMENTS_DIR, 
        MAX_SENTENCES_PER_FILE
    )
    
    if not clean_sentences:
        print("❌ No clean sentences found. Check your judgment directory.")
        return
    
    # Step 2: Create negative examples
    print(f"\n" + "="*60)
    print("STEP 2: CREATING NEGATIVE EXAMPLES")
    print("="*60)
    
    negative_examples = create_negative_examples(
        clean_sentences, 
        MAX_NEGATIVE_EXAMPLES
    )
    
    # Step 3: Show example usage
    print(f"\n" + "="*60)
    print("STEP 3: USAGE EXAMPLE")
    print("="*60)
    
    print("\nTo use these negative examples in your training:")
    print("\n1. Load your original training examples:")
    print("   # original_train_examples = your_existing_train_examples")
    print("\n2. Augment with negative examples:")
    print("   from add_negative_examples import augment_dataset_with_negatives")
    print("   augmented_train = augment_dataset_with_negatives(")
    print("       original_train_examples, negative_examples, negative_ratio=0.3")
    print("   )")
    print("\n3. Use augmented_train for training instead of original_train_examples")
    
    # Show sample negative examples
    print(f"\n" + "="*60)
    print("SAMPLE NEGATIVE EXAMPLES")
    print("="*60)
    
    for i, example in enumerate(negative_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Text: {example['text'][:100]}...")
        print(f"  Tokens: {example['tokens'][:10]}...")
        print(f"  Labels: {example['labels'][:10]}...")
    
    # Save negative examples for later use
    with open("negative_examples.json", 'w', encoding='utf-8') as f:
        json.dump(negative_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Negative examples saved to: negative_examples.json")
    print(f"📊 Total negative examples created: {len(negative_examples)}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
