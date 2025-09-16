# Debug cell to check the label format
print("Checking label format in train_examples:")
print(f"Number of examples: {len(train_examples)}")

if train_examples:
    first_example = train_examples[0]
    print(f"First example keys: {first_example.keys()}")
    print(f"First few tokens: {first_example['tokens'][:10]}")
    print(f"First few labels: {first_example['labels'][:10]}")
    print(f"Label types: {[type(label) for label in first_example['labels'][:5]]}")
    
    # Check if labels are integers or strings
    if first_example['labels']:
        first_label = first_example['labels'][0]
        print(f"First label: {first_label} (type: {type(first_label)})")
        
        if isinstance(first_label, int):
            print("Labels are integers - need to convert back to strings")
            print(f"Label to ID mapping: {ner_dataset.label_to_id}")
            print(f"ID to label mapping: {ner_dataset.id_to_label}")
        else:
            print("Labels are strings - should work fine")
