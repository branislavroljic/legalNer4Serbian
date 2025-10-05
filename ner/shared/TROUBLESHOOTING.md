# Troubleshooting Guide for New Visualizations

## Problem: Empty Loss Plots or "No evaluation metrics found"

### Symptoms
```
✅ Entity distribution across folds visualization displayed
📉 Plotting training and validation loss...
   [Empty graph shown]
   
📈 Plotting macro/micro-averaged metrics over iterations...
⚠️  No evaluation metrics found in training histories
```

### Diagnosis

Run this in your notebook after training to inspect the training history:

```python
from shared import inspect_training_history

# After K-fold training completes
inspect_training_history(fold_results)
```

This will show you:
- How many entries are in the training history
- What types of entries exist (training loss vs evaluation metrics)
- Sample entry structure
- Missing fields

### Common Causes & Solutions

---

## Cause 1: No Evaluation During Training

**Problem**: `TrainingArguments` doesn't have `eval_steps` or `evaluation_strategy` set.

**Check**:
```python
# In your notebook, look for TrainingArguments
training_args = TrainingArguments(
    ...
    evaluation_strategy="steps",  # ← Must be set!
    eval_steps=100,                # ← Must be set!
    ...
)
```

**Solution**:
Add evaluation configuration to `TrainingArguments`:

```python
training_args = TrainingArguments(
    output_dir=fold_output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    
    # ADD THESE:
    evaluation_strategy="steps",  # Evaluate every N steps
    eval_steps=100,               # Evaluate every 100 steps
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    logging_dir=f'{fold_output_dir}/logs',
    logging_steps=50,
    save_total_limit=2,
)
```

---

## Cause 2: Callback Not Receiving Evaluation Metrics

**Problem**: The `PerClassMetricsCallback` is not getting the evaluation metrics from the Trainer.

**Check**:
Look at the debug output from `inspect_training_history()`:
```
Entries with 'eval_loss': 0
Entries with 'eval_precision': 0
```

**Solution**:
Make sure you're passing the validation dataset to the Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # ← Must be provided!
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # ← Must be provided!
    callbacks=[metrics_callback]
)
```

---

## Cause 3: `compute_metrics` Function Not Defined

**Problem**: The Trainer doesn't know how to compute precision, recall, F1.

**Check**:
Look for `compute_metrics` function in your notebook.

**Solution**:
Define and pass `compute_metrics` to the Trainer:

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding (-100 labels)
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:  # Ignore padding
                true_seq.append(l)
                pred_seq_clean.append(p)
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)
    
    # Flatten
    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in pred_labels for label in seq]
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='macro', zero_division=0
    )
    accuracy = accuracy_score(flat_true, flat_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Then pass it to Trainer
trainer = Trainer(
    ...
    compute_metrics=compute_metrics,  # ← Add this!
    ...
)
```

---

## Cause 4: Training History Not Being Stored

**Problem**: The callback's `get_training_history()` returns empty list.

**Check**:
```python
# After training
print(f"History length: {len(metrics_callback.get_training_history())}")
```

**Solution**:
Make sure the callback is properly initialized and added:

```python
# Create callback
metrics_callback = PerClassMetricsCallback(id_to_label=ner_dataset.id_to_label)

# Add to trainer
trainer = Trainer(
    ...
    callbacks=[metrics_callback]  # ← Must be in a list!
)

# After training
trainer.train()

# Store history
fold_result['training_history'] = metrics_callback.get_training_history()
```

---

## Cause 5: Metrics Keys Don't Match

**Problem**: The callback stores metrics with different key names than expected.

**Check**:
Run `inspect_training_history()` and look at the sample entry keys.

**Expected keys**:
- `step`, `epoch`, `loss` (for training entries)
- `step`, `epoch`, `eval_loss`, `eval_precision`, `eval_recall`, `eval_f1`, `eval_accuracy` (for eval entries)

**Solution**:
If keys are different, the callback might need updating. Check `shared/model_utils.py` line 359-404.

---

## Quick Fix Checklist

Run through this checklist:

- [ ] `TrainingArguments` has `evaluation_strategy="steps"` and `eval_steps=100`
- [ ] `Trainer` receives `eval_dataset=val_dataset`
- [ ] `Trainer` receives `compute_metrics=compute_metrics`
- [ ] `compute_metrics` function returns dict with `precision`, `recall`, `f1`, `accuracy`
- [ ] `PerClassMetricsCallback` is created and added to `Trainer`
- [ ] `fold_result['training_history'] = metrics_callback.get_training_history()` is called
- [ ] Training history is not empty: `len(metrics_callback.get_training_history()) > 0`

---

## Example: Complete Working Setup

Here's a complete example that should work:

```python
from shared import (
    PerClassMetricsCallback,
    create_aggregate_report_across_folds,
    inspect_training_history
)

# ... data preparation ...

for fold_num, (train_idx, val_idx) in enumerate(kfold.split(examples_array), 1):
    # ... prepare datasets ...
    
    # Create callback
    metrics_callback = PerClassMetricsCallback(id_to_label=ner_dataset.id_to_label)
    
    # Training arguments with evaluation
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",  # ← Important!
        eval_steps=100,                # ← Important!
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )
    
    # Create trainer with all required components
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,      # ← Important!
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # ← Important!
        callbacks=[metrics_callback]      # ← Important!
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Store results
    fold_result = {
        'fold': fold_num,
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'f1': eval_results['eval_f1'],
        'accuracy': eval_results['eval_accuracy'],
        'distributions': {'train': train_dist, 'val': val_dist},
        'training_history': metrics_callback.get_training_history(),  # ← Important!
        # ... other fields ...
    }
    fold_results.append(fold_result)

# After all folds, inspect if needed
inspect_training_history(fold_results)

# Generate aggregate report
aggregate_report = create_aggregate_report_across_folds(
    fold_results=fold_results,
    model_name="Base BERT",
    display=True
)
```

---

## Still Having Issues?

If you've checked everything above and still have problems:

1. **Run the debug utility**:
   ```python
   from shared import inspect_training_history
   inspect_training_history(fold_results)
   ```

2. **Check a single fold's history**:
   ```python
   history = fold_results[0]['training_history']
   print(f"Total entries: {len(history)}")
   print(f"First 3 entries: {history[:3]}")
   ```

3. **Manually check for eval entries**:
   ```python
   eval_entries = [e for e in history if 'eval_loss' in e]
   print(f"Evaluation entries: {len(eval_entries)}")
   if eval_entries:
       print(f"Sample eval entry: {eval_entries[0]}")
   ```

4. **Check if training actually ran evaluation**:
   ```python
   # Look for evaluation logs during training
   # You should see lines like:
   # {'eval_loss': 0.285, 'eval_precision': 0.315, ...}
   ```

---

## Understanding the Debug Output

When you run `inspect_training_history()`, you'll see output like:

```
================================================================================
INSPECTING TRAINING HISTORY STRUCTURE
================================================================================

================================================================================
FOLD 1
================================================================================
✅ Training history found with 156 entries

Entry types:
  - Training loss only: 144
  - Evaluation metrics only: 12
  - Both: 0
  - Other: 0

Sample entries:

  Training loss entry:
    Keys: ['step', 'epoch', 'loss']
    Values: {'step': 50, 'epoch': 0.17, 'loss': 1.234}

  Evaluation entry:
    Keys: ['step', 'epoch', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy', 'per_class_metrics']
    Sample values:
      step: 100
      epoch: 0.33
      eval_loss: 0.285
      eval_precision: 0.315
      eval_recall: 0.199
      eval_f1: 0.244

  ✅ All required evaluation fields present
```

**Good signs**:
- ✅ "Evaluation metrics only" > 0
- ✅ "All required evaluation fields present"
- ✅ Evaluation entries have `eval_precision`, `eval_recall`, `eval_f1`

**Bad signs**:
- ❌ "Evaluation metrics only" = 0
- ❌ "WARNING: Evaluation entries missing fields"
- ❌ No evaluation entry shown

---

## Contact

If none of these solutions work, please provide:
1. Output from `inspect_training_history(fold_results)`
2. Your `TrainingArguments` configuration
3. Your `Trainer` initialization code
4. Any error messages or warnings

