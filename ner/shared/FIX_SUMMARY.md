# Fix Summary: Empty Training History Issue

## Problem Identified

The training history was completely empty (0 entries) because the `PerClassMetricsCallback` was **not being added to the Trainer**.

### Root Cause

The `create_trainer()` function in `shared/model_utils.py` was hardcoding the callbacks list:

```python
# OLD CODE (line 143)
callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
```

This meant that even though the notebook created `metrics_callback`, it was never passed to the Trainer, so the callback's `on_log()` and `on_evaluate()` methods were never triggered.

## Solution

### 1. Modified `create_trainer()` in `shared/model_utils.py`

Added an `additional_callbacks` parameter to accept custom callbacks:

```python
def create_trainer(model, training_args, train_dataset, val_dataset, 
                  data_collator, tokenizer, id_to_label: Dict[int, str],
                  early_stopping_patience: int = 3,
                  additional_callbacks: list = None) -> Trainer:  # ← NEW PARAMETER
    """Create and configure the Trainer with optional additional callbacks."""
    
    # Build callbacks list
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    if additional_callbacks:
        if isinstance(additional_callbacks, list):
            callbacks.extend(additional_callbacks)
        else:
            callbacks.append(additional_callbacks)
    
    trainer = Trainer(
        ...
        callbacks=callbacks  # ← Now includes both EarlyStopping AND custom callbacks
    )
    
    return trainer
```

### 2. Updated Notebook to Pass Callback

Modified `serbian_legal_ner_pipeline_base_5fold_cv.ipynb` to pass the callback:

```python
# Create metrics callback
metrics_callback = PerClassMetricsCallback(id_to_label=ner_dataset.id_to_label)

# Create trainer with metrics callback
trainer = create_trainer(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    id_to_label=ner_dataset.id_to_label,
    early_stopping_patience=3,
    additional_callbacks=[metrics_callback]  # ← ADDED THIS LINE
)
```

## What This Fixes

✅ **Training history will now be populated** with entries during training
✅ **Loss plots will show data** instead of being empty
✅ **Macro/micro metrics plots will work** instead of showing "No evaluation metrics found"
✅ **All three new visualizations will work properly**

## Expected Behavior After Fix

When you run the notebook now, `inspect_training_history()` should show:

```
================================================================================
FOLD 1
================================================================================
✅ Training history found with 156 entries  ← NOT 0!

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
    Keys: ['step', 'epoch', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1', ...]
    Sample values:
      step: 100
      epoch: 0.33
      eval_loss: 0.285
      eval_precision: 0.315
      eval_recall: 0.199
      eval_f1: 0.244

  ✅ All required evaluation fields present
```

And the visualizations will show:

1. ✅ **Entity Distribution** - Grouped bar chart (already working)
2. ✅ **Training & Validation Loss** - Loss curves with data (now fixed!)
3. ✅ **Macro/Micro Metrics** - Performance dynamics over iterations (now fixed!)

## Files Modified

1. **`ner/shared/model_utils.py`**
   - Added `additional_callbacks` parameter to `create_trainer()`
   - Modified to accept and include custom callbacks

2. **`ner/notebook_base_k-fold/serbian_legal_ner_pipeline_base_5fold_cv.ipynb`**
   - Added `additional_callbacks=[metrics_callback]` to `create_trainer()` call

## Next Steps

1. **Re-run the notebook** - The training history should now be populated
2. **Check with debug utility**:
   ```python
   from shared import inspect_training_history
   inspect_training_history(fold_results)
   ```
3. **View the visualizations** - All three new plots should now work!

## Backward Compatibility

✅ The changes are backward compatible:
- If `additional_callbacks` is not provided, it defaults to `None`
- The function works exactly as before with just `EarlyStoppingCallback`
- Existing notebooks that don't pass callbacks will continue to work

## For Other Notebooks

If you want to apply this fix to other 5-fold CV notebooks (BERT-CRF, XLM-R, etc.), make the same change:

```python
# In the create_model_and_trainer function
trainer = create_trainer(
    ...
    additional_callbacks=[metrics_callback]  # ← Add this line
)
```

That's it! The fix is simple but critical for the new visualizations to work.

