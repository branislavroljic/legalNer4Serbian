# Troubleshooting Guide for bertic_base_class_weights.ipynb

## Issue: ImportError - cannot import name 'compute_metrics'

### Error Message
```
ImportError: cannot import name 'compute_metrics' from 'shared.evaluation'
```

### Root Cause
The `compute_metrics` function is defined in `shared/model_utils.py` but is **not exported** in `shared/__init__.py`. It's an internal function used by `create_trainer()`.

### Solution
Import `compute_metrics` directly from `shared.model_utils` instead of `shared.evaluation`:

```python
# ✅ CORRECT
from shared.model_utils import compute_metrics

# ❌ INCORRECT
from shared.evaluation import compute_metrics
```

### Fixed Code in Notebook

The `create_model_and_weighted_trainer()` function now includes:

```python
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
```

### Why This Works

1. **Direct import**: We import `compute_metrics` directly from `model_utils.py` where it's defined
2. **Callback handling**: We manually create the callbacks list (including EarlyStoppingCallback) since we're not using `create_trainer()`
3. **Bound function**: We create `compute_metrics_fn` that binds `id_to_label` to the `compute_metrics` function

### Alternative Approach (Not Recommended)

You could also add `compute_metrics` to the exports in `shared/__init__.py`, but this is not recommended because:
- It's an internal utility function
- It requires `id_to_label` parameter binding
- The current approach keeps the shared module clean

## Issue: OSError - File name too long

### Error Message
```
OSError: [Errno 36] File name too long: "{0: 'B-CASE_NUMBER', 1: 'B-COURT', ...}/classification_report_foldFold 1 - Class Weights.txt"
```

### Root Cause
The `generate_detailed_classification_report()` function was being called with incorrect argument order. The function signature is:

```python
generate_detailed_classification_report(
    true_labels,      # List[List[str]]
    predictions,      # List[List[str]]
    output_dir,       # str - directory path
    fold_num=None,    # int - fold number
    dataset_name="Validation"  # str - dataset name
)
```

But it was being called with `id_to_label` dictionary as the `output_dir` parameter.

### Solution
Use the correct argument order and pass the fold output directory:

```python
# ✅ CORRECT
fold_report = generate_detailed_classification_report(
    fold_result['true_labels'],
    fold_result['true_predictions'],
    fold_output_dir,  # output_dir parameter (directory path)
    fold_num=fold_num,  # fold_num parameter
    dataset_name="Class Weights Validation"  # dataset_name parameter
)

# ❌ INCORRECT
fold_report = generate_detailed_classification_report(
    fold_result['true_labels'],
    fold_result['true_predictions'],
    ner_dataset.id_to_label,  # Wrong! This is a dictionary, not a directory
    f"Fold {fold_num} - Class Weights"  # Wrong position for dataset_name
)
```

### What the Function Does
1. Generates a classification report from true labels and predictions
2. Saves the report to a text file in `output_dir`
3. Returns a dictionary with per-class metrics

---

## Issue: ValueError - classes should have valid labels that are in y

### Error Message
```
ValueError: classes should have valid labels that are in y
```

### Root Cause
This error occurs in the `calculate_class_weights_from_tokenized()` function when using sklearn's `compute_class_weight`. The function expects all classes passed to it to be present in the training data.

In k-fold cross-validation, some BIO labels may not appear in certain folds' training sets. For example:
- Rare entity types (e.g., `I-CASE_NUMBER`) might not appear in a small training fold
- Some folds might not have multi-token entities of certain types
- The 'O' (Outside) label is usually present, but specific B-/I- tags might be missing

### Solution
The fix modifies `calculate_class_weights_from_tokenized()` to:

1. **Only calculate weights for labels present in training data**
2. **Assign default weight (1.0) to absent labels**
3. **Create a complete weight tensor for all possible labels**

```python
# ✅ CORRECT - Only use labels that appear in training data
unique_labels_in_data = np.array(sorted(list(set(all_label_ids))))

class_weights_for_present = compute_class_weight(
    class_weight='balanced',
    classes=unique_labels_in_data,  # Only present labels
    y=np.array(all_label_ids)
)

# Create full weight array
num_labels = len(label_to_id)
class_weights = np.ones(num_labels)  # Default: 1.0 for unseen labels

# Fill in calculated weights
for label_id, weight in zip(unique_labels_in_data, class_weights_for_present):
    class_weights[label_id] = weight
```

### What This Means
- **Labels in training data**: Get balanced weights based on frequency
- **Labels not in training data**: Get neutral weight (1.0)
- **Model behavior**: Can still predict any label, but loss only weighted for seen labels

### Expected Output
After the fix, you should see output like:
```
📊 Class weights calculated:
  Total label types: 29
  Labels present in training: 26
  Labels absent from training: 3
  Total valid tokens: 185432
  Weight range: 0.0234 - 15.4321
  Mean weight: 1.8765
```

This is normal and expected in k-fold CV!

---

## Other Common Issues

### Issue: Class weights not being applied

**Symptom**: Training runs but results are identical to baseline

**Check**:
1. Verify `WeightedTrainer` is being used (not standard `Trainer`)
2. Check that `class_weights` parameter is passed to `WeightedTrainer`
3. Verify class weights are calculated correctly (check printed output)

### Issue: CUDA out of memory

**Symptom**: RuntimeError about CUDA memory

**Solutions**:
1. Reduce batch size in model config
2. Reduce max_length in tokenization
3. Add memory cleanup between folds:
   ```python
   del model, trainer, train_dataset, val_dataset
   torch.cuda.empty_cache()
   ```

### Issue: Different results between folds

**Symptom**: High variance in F1-scores across folds

**This is expected** because:
- Each fold has different training/validation splits
- Class weights are calculated per-fold based on training data
- Some folds may have more/less balanced entity distributions

**Check**:
- Look at entity distribution per fold (printed during training)
- Compare standard deviations in aggregate report
- Ensure random seed is set consistently

## Verification Steps

After fixing the import error, verify the notebook works by:

1. **Check imports cell runs without errors**
2. **Verify class weights calculation**:
   - Should print weight range and mean
   - Weights should be > 1.0 for rare classes
3. **Confirm WeightedTrainer creation**:
   - Should print "Creating WeightedTrainer with class weights"
4. **Monitor training**:
   - Loss should decrease normally
   - Evaluation metrics should be computed at eval_steps

## Getting Help

If you encounter other issues:

1. Check the original `bertic_base.ipynb` works correctly
2. Compare the two notebooks section by section
3. Verify all shared modules are up to date
4. Check that paths are correctly configured for Paperspace environment

