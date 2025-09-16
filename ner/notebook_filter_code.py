# ============================================================================
# ADD THIS CODE TO YOUR JUPYTER NOTEBOOK TO FILTER SENTENCES WITHOUT ENTITIES
# ============================================================================

# Cell 1: Import the filtering functions
from filter_sentences import (
    analyze_entity_distribution, 
    print_analysis, 
    filter_examples_with_entities,
    compare_datasets,
    save_filtered_dataset,
    quick_filter_demo
)

# Cell 2: Quick analysis and filtering (replace bio_examples with your actual data)
print("🔍 ANALYZING AND FILTERING YOUR DATASET")
print("=" * 60)

# Option 1: Quick demo (analyzes, filters, and compares automatically)
filtered_examples, removed_examples = quick_filter_demo(bio_examples)

# Cell 3: Save the filtered dataset
save_filtered_dataset(filtered_examples, removed_examples, 'filtered_bio_examples.json')

# Cell 4: Use filtered dataset in your existing pipeline
print("\n💻 USING FILTERED DATASET IN YOUR PIPELINE")
print("=" * 50)
print("Replace 'bio_examples' with 'filtered_examples' in your existing code:")
print()
print("# OLD:")
print("# ner_dataset = NERDataset(bio_examples)")
print()
print("# NEW:")
print("# ner_dataset = NERDataset(filtered_examples)")
print()
print("This will train your model only on sentences that contain entities!")

# ============================================================================
# ALTERNATIVE: Manual step-by-step approach
# ============================================================================

# If you prefer more control, use this step-by-step approach instead:

# Step 1: Analyze your current dataset
# original_stats = analyze_entity_distribution(bio_examples)
# print_analysis(original_stats)

# Step 2: Filter examples (basic filtering - remove sentences with no entities)
# filtered_examples, removed_examples = filter_examples_with_entities(bio_examples, min_entities=1)

# Step 3: Analyze filtered dataset
# if filtered_examples:
#     filtered_stats = analyze_entity_distribution(filtered_examples)
#     compare_datasets(original_stats, filtered_stats)

# Step 4: Save filtered dataset
# save_filtered_dataset(filtered_examples, removed_examples)

# ============================================================================
# ADVANCED FILTERING OPTIONS
# ============================================================================

# Option 1: Keep only examples with at least 2 entities
# from filter_sentences import filter_examples_by_criteria
# advanced_filtered, advanced_removed = filter_examples_by_criteria(
#     bio_examples,
#     min_entities=2,
#     min_tokens=5,
#     max_tokens=100
# )

# Option 2: Keep only examples with specific entity types
# important_entity_types = ['JUDGE', 'DEFENDANT', 'COURT']  # Adjust based on your entity types
# entity_specific_filtered, entity_specific_removed = filter_examples_by_criteria(
#     bio_examples,
#     min_entities=1,
#     required_entity_types=important_entity_types
# )

# ============================================================================
# INTEGRATION WITH YOUR EXISTING PIPELINE
# ============================================================================

# After filtering, simply replace bio_examples with filtered_examples in your existing code:

# OLD CODE:
# ner_dataset = NERDataset(bio_examples)
# prepared_examples = ner_dataset.prepare_for_training()

# NEW CODE:
# ner_dataset = NERDataset(filtered_examples)  # Use filtered data
# prepared_examples = ner_dataset.prepare_for_training()

# The rest of your pipeline remains exactly the same!

# ============================================================================
# BENEFITS OF FILTERING
# ============================================================================

print("\n🎯 BENEFITS OF FILTERING SENTENCES WITHOUT ENTITIES:")
print("=" * 60)
print("✅ Reduced class imbalance (fewer 'O' labels)")
print("✅ Faster training (smaller dataset)")
print("✅ Better model focus (entity-rich content)")
print("✅ Improved performance (less noise)")
print("✅ More efficient use of computational resources")
print()
print("📈 Expected improvements:")
print("   - Better precision and recall")
print("   - Faster convergence during training")
print("   - Reduced overfitting on 'O' labels")
print("   - Better entity boundary detection")
