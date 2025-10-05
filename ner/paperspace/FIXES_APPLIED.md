# Fixes Applied to bertic_base_class_weights.ipynb

This document summarizes all the fixes applied to resolve errors in the class weights notebook.

## Fix #1: ImportError for `compute_metrics`

### Error
```
ImportError: cannot import name 'compute_metrics' from 'shared.evaluation'
```

### Root Cause
The `compute_metrics` function is defined in `shared/model_utils.py` but not exported in `shared/__init__.py`.

### Fix Applied
Changed the import in `create_model_and_weighted_trainer()` function:

**Before:**
```python
from shared.evaluation import compute_metrics  # ❌ Wrong module
```

**After:**
```python
from shared.model_utils import compute_metrics  # ✅ Correct module
from transformers import EarlyStoppingCallback
```

**Complete Fixed Code:**
```python
def create_model_and_weighted_trainer(fold_num, train_dataset, val_dataset, data_collator, 
                                     tokenizer, ner_dataset, class_weights, device):
    # ... model creation code ...
    
    # Create metrics callback for comprehensive tracking
    metrics_callback = PerClassMetricsCallback(id_to_label=ner_dataset.id_to_label)

    # Import compute_metrics from model_utils (not exported in __init__.py)
    from shared.model_utils import compute_metrics
    from transformers import EarlyStoppingCallback
    
    # Create compute_metrics function with id_to_label bound
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, ner_dataset.id_to_label)

    # Build callbacks list
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3), metrics_callback]

    # Create weighted trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks
    )
    
    return model, trainer, metrics_callback, fold_output_dir
```

### Location in Notebook
- **Cell**: Section 8 - K-Fold Cross-Validation Helper Functions
- **Function**: `create_model_and_weighted_trainer()`
- **Lines**: ~503-554

---

## Fix #2: OSError - File name too long

### Error
```
OSError: [Errno 36] File name too long: "{0: 'B-CASE_NUMBER', 1: 'B-COURT', ...}/classification_report_foldFold 1 - Class Weights.txt"
```

### Root Cause
The `generate_detailed_classification_report()` function was called with incorrect argument order. The `id_to_label` dictionary was passed as the `output_dir` parameter, creating an invalid filename.

### Function Signature
```python
def generate_detailed_classification_report(
    true_labels: List[List[str]],
    predictions: List[List[str]],
    output_dir: str,              # ← Directory path, not dictionary!
    fold_num: int = None,
    dataset_name: str = "Validation"
)
```

### Fix Applied

**Before:**
```python
fold_report = generate_detailed_classification_report(
    fold_result['true_labels'],
    fold_result['true_predictions'],
    ner_dataset.id_to_label,  # ❌ Wrong! This is a dictionary
    f"Fold {fold_num} - Class Weights"  # ❌ Wrong parameter position
)
```

**After:**
```python
fold_report = generate_detailed_classification_report(
    fold_result['true_labels'],
    fold_result['true_predictions'],
    fold_output_dir,  # ✅ Correct: directory path
    fold_num=fold_num,  # ✅ Correct: fold number as named parameter
    dataset_name="Class Weights Validation"  # ✅ Correct: dataset name
)
```

### Location in Notebook
- **Cell**: Section 9 - K-Fold Cross-Validation Training Loop
- **Lines**: ~677-685

---

---

## Fix #3: ValueError - classes should have valid labels that are in y

### Error
```
ValueError: classes should have valid labels that are in y
```

### Root Cause
The `compute_class_weight` function from sklearn expects all classes to be present in the training data. However, in k-fold cross-validation, some BIO labels may not appear in certain folds' training sets (especially rare entity types or their I- variants).

The original code tried to calculate weights for all possible labels (29 BIO labels) even if some didn't appear in the current fold's training data.

### Fix Applied

**Before:**
```python
# Get unique classes (as integers)
classes = np.array(list(range(len(label_to_id))))  # All 29 labels

# Calculate class weights using sklearn's balanced approach
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,  # ❌ Some of these may not be in training data!
    y=np.array(all_label_ids)
)
```

**After:**
```python
# Get unique classes that actually appear in the training data
unique_labels_in_data = np.array(sorted(list(set(all_label_ids))))

# Calculate class weights using sklearn's balanced approach for labels that appear
class_weights_for_present = compute_class_weight(
    class_weight='balanced',
    classes=unique_labels_in_data,  # ✅ Only labels present in training
    y=np.array(all_label_ids)
)

# Create full weight array for all possible labels
num_labels = len(label_to_id)
class_weights = np.ones(num_labels)  # Default weight of 1.0 for unseen labels

# Fill in calculated weights for labels that appear in training data
for label_id, weight in zip(unique_labels_in_data, class_weights_for_present):
    class_weights[label_id] = weight

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights)
```

### Strategy
1. **Calculate weights only for present labels**: Use sklearn's `compute_class_weight` only for labels that actually appear in the training data
2. **Default weight for absent labels**: Assign weight of 1.0 to labels not in training data (neutral weight)
3. **Full weight tensor**: Create a complete weight tensor covering all possible labels

### Why This Works
- Labels not in training data get neutral weight (1.0), so they don't affect loss calculation
- Labels in training data get balanced weights based on their frequency
- The model can still predict any label, but loss is only weighted for labels seen during training

### Location in Notebook
- **Cell**: Section 6 - Class Weights Implementation
- **Function**: `calculate_class_weights_from_tokenized()`
- **Lines**: ~275-325

---

---

## Fix #4: TypeError - Missing Data for Aggregate Report

### Error
```
TypeError: 'NoneType' object is not subscriptable
```

And warnings:
```
⚠️  No entity distributions found in fold results
⚠️  No training histories found in fold results
⚠️  No per-class metrics found in fold results
⚠️  No confusion matrices found in fold results
```

### Root Cause
The `create_aggregate_report_across_folds()` function expects `fold_results` to contain comprehensive data including:
- Entity distributions (`distributions`)
- Training histories (`training_history`)
- Per-class metrics (`per_class_metrics`)
- Confusion matrices (`confusion_matrix`)
- Label lists (`labels`)

The class weights notebook was only storing basic metrics (precision, recall, F1, accuracy) and not collecting this additional data.

Additionally, the function was being called with incorrect parameters:
```python
# ❌ INCORRECT
create_aggregate_report_across_folds(
    fold_results,
    ner_dataset.id_to_label,  # Wrong! Should be model_name (string)
    "BERTić Base with Class Weights - 5-Fold CV"  # Wrong position
)
```

### Fix Applied

**1. Enhanced data collection in training loop:**

```python
# After training and evaluation
fold_result = train_and_evaluate_fold(fold_num, trainer, val_dataset, ner_dataset)

# Get predictions and labels for aggregation
predictions, labels, _ = trainer.predict(val_dataset)
predictions = np.argmax(predictions, axis=2)

# Convert to label names
true_labels = [[ner_dataset.id_to_label[l] for l in label if l != -100] for label in labels]
pred_labels = [[ner_dataset.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
               for prediction, label in zip(predictions, labels)]

# Generate per-class metrics and confusion matrix
from sklearn.metrics import confusion_matrix
flat_true = [label for seq in true_labels for label in seq]
flat_pred = [label for seq in pred_labels for label in seq]
all_labels = sorted(list(set(flat_true + flat_pred)))
cm = confusion_matrix(flat_true, flat_pred, labels=all_labels)

# Generate classification report
per_class_metrics = generate_detailed_classification_report(
    true_labels, pred_labels, fold_output_dir, fold_num, "Class Weights Validation"
)

# Store comprehensive data for aggregation
fold_result['distributions'] = {'train': train_dist, 'val': val_dist}
fold_result['per_class_metrics'] = per_class_metrics
fold_result['confusion_matrix'] = cm
fold_result['labels'] = all_labels
fold_result['training_history'] = metrics_callback.get_training_history()
fold_results.append(fold_result)
```

**2. Corrected function call:**

```python
# ✅ CORRECT
aggregate_report = create_aggregate_report_across_folds(
    fold_results=fold_results,
    model_name="BERTić Base with Class Weights",  # Correct: string parameter
    display=True  # Display visualizations in notebook
)
```

### What This Enables
With the complete data collection, the aggregate report now includes:
- 📊 Entity distribution plots across folds
- 📉 Training and validation loss curves
- 📈 Macro/micro-averaged metrics over iterations
- 🔥 Aggregated confusion matrices
- 📊 Per-class F1 scores over training iterations
- 📈 Comprehensive statistical analysis

### Location in Notebook
- **Training Loop**: Section 9 - Lines ~679-714
- **Aggregate Report**: Section 10 - Lines ~733-791

---

## Summary of Changes

| Issue | Module/Function | Change Type | Status |
|-------|----------------|-------------|--------|
| ImportError | `create_model_and_weighted_trainer()` | Import statement | ✅ Fixed |
| OSError | Training loop | Function call arguments | ✅ Fixed |
| ValueError | `calculate_class_weights_from_tokenized()` | Class weight calculation logic | ✅ Fixed |
| TypeError | Training loop & aggregate report | Data collection & function call | ✅ Fixed |

## Testing Checklist

After applying these fixes, verify:

- [ ] All import cells run without errors
- [ ] Class weights are calculated and printed for each fold
- [ ] WeightedTrainer is created successfully
- [ ] Training runs without errors
- [ ] Classification reports are saved to correct locations
- [ ] No "File name too long" errors
- [ ] Aggregate report is generated at the end

## Expected Output Structure

After successful run, you should have:

```
/storage/models/bertic_base_class_weights_5fold_cv/
├── fold_1/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── classification_report_fold1.txt  ← Generated by generate_detailed_classification_report
│   ├── classification_report.json       ← Generated by training loop
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── aggregate_classification_report.json
```

## Additional Notes

### Why These Errors Occurred

1. **ImportError**: The notebook was initially modeled after other notebooks that might have had `compute_metrics` in a different location or defined it inline. The shared module structure doesn't export this function.

2. **OSError**: The function signature for `generate_detailed_classification_report` was misunderstood. It expects a directory path, not a label mapping dictionary.

### Prevention

To avoid similar issues in the future:

1. Always check function signatures in shared modules before calling them
2. Use named parameters when calling functions with multiple optional parameters
3. Verify that imported functions exist in the specified module
4. Test each section of the notebook incrementally rather than running all cells at once

## Related Files

- **Main Notebook**: `ner/paperspace/bertic_base_class_weights.ipynb`
- **Troubleshooting Guide**: `ner/paperspace/TROUBLESHOOTING.md`
- **Implementation Guide**: `ner/paperspace/CLASS_WEIGHTS_IMPLEMENTATION.md`
- **Comparison Guide**: `ner/paperspace/NOTEBOOK_COMPARISON.md`

## Version

- **Notebook Version**: 1.0.1 (with fixes)
- **Date**: 2025-10-04
- **Status**: Ready for use

