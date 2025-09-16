# ============================================================================
# SMART FILTERING: DOWNSAMPLE SENTENCES WITH ONLY 'O' LABELS
# Add this code right after your train/test split in your notebook
# ============================================================================

import random

print("🔍 APPLYING SMART FILTERING TO REDUCE CLASS IMBALANCE")
print("=" * 60)

# Make sure we know what O is
O_id = ner_dataset.label_to_id["O"]
print(f"'O' label ID: {O_id}")

def analyze_split_distribution(examples, split_name):
    """Analyze the distribution of examples in a split."""
    positive = [ex for ex in examples if not all(l == O_id for l in ex["labels"])]
    negative = [ex for ex in examples if all(l == O_id for l in ex["labels"])]
    
    total_tokens = sum(len(ex["labels"]) for ex in examples)
    o_tokens = sum(ex["labels"].count(O_id) for ex in examples)
    entity_tokens = total_tokens - o_tokens
    
    print(f"\n{split_name} split analysis:")
    print(f"  Total examples: {len(examples):,}")
    print(f"  Examples with entities: {len(positive):,} ({len(positive)/len(examples)*100:.1f}%)")
    print(f"  Examples with only 'O': {len(negative):,} ({len(negative)/len(examples)*100:.1f}%)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Entity tokens: {entity_tokens:,} ({entity_tokens/total_tokens*100:.1f}%)")
    print(f"  'O' tokens: {o_tokens:,} ({o_tokens/total_tokens*100:.1f}%)")
    print(f"  Class imbalance ratio: {o_tokens/max(entity_tokens, 1):.1f}:1")
    
    return len(positive), len(negative)

def filter_all_O(examples, keep_ratio=0.2, split_name=""):
    """Remove sentences with only O, but keep some (keep_ratio)."""
    positive = [ex for ex in examples if not all(l == O_id for l in ex["labels"])]
    negative = [ex for ex in examples if all(l == O_id for l in ex["labels"])]
    
    print(f"\n{split_name} filtering:")
    print(f"  Examples with entities: {len(positive):,} (keeping all)")
    print(f"  Examples with only 'O': {len(negative):,}")
    
    # Downsample negative examples
    if keep_ratio > 0 and len(negative) > 0:
        keep_n = max(1, int(len(negative) * keep_ratio))
        negative_sample = random.sample(negative, keep_n)
        print(f"  Keeping {len(negative_sample):,} 'O'-only examples ({keep_ratio*100:.0f}%)")
        print(f"  Removing {len(negative) - len(negative_sample):,} 'O'-only examples")
    else:
        negative_sample = []
        print(f"  Removing all 'O'-only examples")
    
    filtered_examples = positive + negative_sample
    print(f"  Final size: {len(filtered_examples):,} examples")
    
    return filtered_examples

# Analyze original distribution
print("\n📊 ORIGINAL DISTRIBUTION")
print("=" * 40)
analyze_split_distribution(train_examples, "Training")
analyze_split_distribution(val_examples, "Validation") 
analyze_split_distribution(test_examples, "Test")

# Set random seed for reproducibility
random.seed(42)

# Apply filtering to all splits
print(f"\n🔧 APPLYING FILTERING (keeping {20}% of 'O'-only examples)")
print("=" * 60)

original_train_size = len(train_examples)
original_val_size = len(val_examples)
original_test_size = len(test_examples)

train_examples = filter_all_O(train_examples, keep_ratio=0.2, split_name="Training")
val_examples = filter_all_O(val_examples, keep_ratio=0.2, split_name="Validation")
test_examples = filter_all_O(test_examples, keep_ratio=0.2, split_name="Test")

# Analyze filtered distribution
print("\n📊 FILTERED DISTRIBUTION")
print("=" * 40)
analyze_split_distribution(train_examples, "Training")
analyze_split_distribution(val_examples, "Validation")
analyze_split_distribution(test_examples, "Test")

# Summary
print(f"\n✨ FILTERING SUMMARY")
print("=" * 30)
print(f"Training: {original_train_size:,} → {len(train_examples):,} ({(original_train_size-len(train_examples))/original_train_size*100:.1f}% reduction)")
print(f"Validation: {original_val_size:,} → {len(val_examples):,} ({(original_val_size-len(val_examples))/original_val_size*100:.1f}% reduction)")
print(f"Test: {original_test_size:,} → {len(test_examples):,} ({(original_test_size-len(test_examples))/original_test_size*100:.1f}% reduction)")

total_original = original_train_size + original_val_size + original_test_size
total_filtered = len(train_examples) + len(val_examples) + len(test_examples)
print(f"Overall: {total_original:,} → {total_filtered:,} ({(total_original-total_filtered)/total_original*100:.1f}% reduction)")

print(f"\n🎯 BENEFITS:")
print("✅ Reduced class imbalance while keeping some negative examples")
print("✅ Faster training with smaller dataset")
print("✅ Better model focus on entity-rich content")
print("✅ Maintained dataset diversity for generalization")

# ============================================================================
# ALTERNATIVE: Different keep ratios for experimentation
# ============================================================================

# You can experiment with different keep_ratio values:
# keep_ratio=0.0   # Remove all 'O'-only examples (most aggressive)
# keep_ratio=0.1   # Keep 10% of 'O'-only examples
# keep_ratio=0.2   # Keep 20% of 'O'-only examples (balanced)
# keep_ratio=0.3   # Keep 30% of 'O'-only examples (conservative)
# keep_ratio=1.0   # Keep all examples (no filtering)

# Example of trying different ratios:
# print("\n🧪 EXPERIMENTING WITH DIFFERENT RATIOS")
# print("=" * 50)
# 
# for ratio in [0.0, 0.1, 0.2, 0.3]:
#     temp_train = filter_all_O(train_examples_original, keep_ratio=ratio, split_name=f"Train (ratio={ratio})")
#     print(f"  Ratio {ratio}: {len(temp_train)} examples")

print(f"\n🚀 Ready to continue with training using filtered dataset!")
print("   Your existing training code will now use the balanced dataset.")
