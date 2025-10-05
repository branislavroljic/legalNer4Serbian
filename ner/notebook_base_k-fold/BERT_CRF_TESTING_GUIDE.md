# BERT-CRF Testing Guide: Isolating BERT vs CRF Issues

## Overview

The BERT-CRF notebook now supports **toggling the CRF layer on/off** to help diagnose whether performance issues are caused by:
1. **BERT implementation** (tokenization, model architecture, training)
2. **CRF layer** (transition matrix, decoding, masking)

## Quick Start

### Test 1: BERT Only (No CRF - Argmax Decoding)

This tests if the BERT part is implemented the same as the base notebook.

**Set in the configuration cell:**
```python
USE_CRF = False  # Disable CRF, use argmax decoding (same as base model)
```

**Expected Result:**
- Should produce **similar results** to the base BERTić notebook
- If results are similar → BERT implementation is correct
- If results are different → Issue is in BERT part (tokenization, training, etc.)

### Test 2: BERT + CRF (Full Model)

This tests the complete BERT-CRF model.

**Set in the configuration cell:**
```python
USE_CRF = True  # Enable CRF layer
```

**Expected Result:**
- Should produce **better or similar** results to BERT-only
- If results are worse → Issue is in CRF implementation
- If results are better → CRF is working correctly!

## What Changed

### 1. Model Class (`BertCrfForTokenClassification`)

**Added `use_crf` parameter:**
```python
def __init__(self, config, num_labels, use_crf=True):
    self.use_crf = use_crf
    
    # CRF layer (optional)
    if self.use_crf:
        self.crf = CRF(num_labels, batch_first=True)
    else:
        self.crf = None
```

### 2. Forward Pass (Training)

**Conditional loss calculation:**
```python
if self.use_crf:
    # CRF loss (negative log-likelihood)
    log_likelihood = self.crf(logits, labels_for_crf, mask=mask, reduction='mean')
    loss = -log_likelihood
else:
    # Standard cross-entropy loss (same as base BERT)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(active_logits, active_labels)
```

### 3. Prediction (Inference)

**Conditional decoding:**
```python
if self.use_crf:
    # CRF Viterbi decoding
    predictions = self.crf.decode(logits, mask=mask)
    return predictions  # List of lists
else:
    # Argmax decoding (same as base BERT)
    predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy()  # Numpy array
```

### 4. Evaluation Function

**Handles both prediction formats:**
```python
if USE_CRF:
    # CRF returns list of lists
    for pred_seq, label_seq, attention_seq in zip(predictions, labels, batch['attention_mask']):
        # Filter and process...
else:
    # Argmax returns numpy array
    for pred_seq, label_seq in zip(predictions, labels.cpu().numpy()):
        # Filter and process...
```

## Diagnostic Workflow

### Step 1: Test BERT Implementation

1. **Set `USE_CRF = False`**
2. **Run the notebook** (can use `N_FOLDS = 2` for quick test)
3. **Compare results** with base BERTić notebook

**If results match base notebook:**
✅ BERT implementation is correct
→ Proceed to Step 2

**If results don't match:**
❌ Issue is in BERT part
→ Check:
- Tokenization (sliding window parameters)
- Model architecture (dropout, classifier)
- Training arguments (learning rate, batch size, epochs)
- Data preparation (train/val split)

### Step 2: Test CRF Layer

1. **Set `USE_CRF = True`**
2. **Run the notebook**
3. **Compare results** with BERT-only (Step 1)

**If results improve:**
✅ CRF is working correctly!
→ CRF is learning valid transition patterns

**If results are similar:**
⚠️ CRF is not helping but not hurting
→ May need more data or better hyperparameters

**If results are catastrophically worse (F1 drops from 0.85 → 0.02):**
❌ CRF implementation has critical bugs
→ Check:
- Mask alignment (training vs inference)
- Label handling (-100 replacement)
- Transition matrix initialization
- Decoding logic

## Known Issues and Fixes Applied

### Issue 1: Dropout During Inference ✅ FIXED
**Problem:** `self.dropout()` was called during prediction
**Fix:** Removed dropout from `predict()` method
```python
# Before (WRONG):
sequence_output = self.dropout(outputs.last_hidden_state)

# After (CORRECT):
sequence_output = outputs.last_hidden_state
```

### Issue 2: Inconsistent Masking ✅ FIXED
**Problem:** Different masking logic in training vs inference
**Fix:** Consistent masking in both forward and predict methods

### Issue 3: Label Replacement ⚠️ POTENTIAL ISSUE
**Current approach:**
```python
labels_masked = labels.clone()
labels_masked[labels == -100] = 0  # Replace with 'O' tag
```

**Concern:** This treats padding/special tokens as 'O' labels during CRF training, which may corrupt the transition matrix.

**Better approach (to test):**
- Properly mask out these positions so CRF never sees them
- Don't include them in transition probability calculations

## Comparison Checklist

When comparing BERT-only vs base notebook, check:

- [ ] **Dataset sizes match** (train/val examples after tokenization)
- [ ] **Entity distributions match** (same entities in train/val splits)
- [ ] **Hyperparameters match** (learning rate, batch size, epochs)
- [ ] **Evaluation metrics match** (precision, recall, F1 per class)
- [ ] **Overall F1 is within ±2%** (small variance is normal)

## Expected Performance

Based on your observations:

### Base BERTić (Argmax Decoding)
```
B-CASE_NUMBER:     F1 = 0.85
B-DECISION_DATE:   F1 = 0.91
B-DEFENDANT:       F1 = 0.69
I-DEFENDANT:       F1 = 0.65
I-PROCEDURE_COSTS: F1 = 0.96
```

### BERT-CRF with `USE_CRF = False` (Should Match Base)
```
Should be similar to base BERTić
```

### BERT-CRF with `USE_CRF = True` (Current - Broken)
```
B-CASE_NUMBER:     F1 = 0.02  ❌ Catastrophic drop
B-DECISION_DATE:   F1 = 0.05  ❌ Catastrophic drop
B-DEFENDANT:       F1 = 0.47  ❌ Significant drop
I-DEFENDANT:       F1 = 0.08  ❌ Catastrophic drop
I-PROCEDURE_COSTS: F1 = 0.00  ❌ Complete collapse
```

## Next Steps

1. **Run with `USE_CRF = False`** to confirm BERT implementation matches base
2. **If BERT matches:** The problem is definitely in the CRF layer
3. **If BERT doesn't match:** Fix BERT implementation first, then test CRF

## Output Directory

The output directory automatically changes based on `USE_CRF`:
- `USE_CRF = False` → `bertic_no_crf_5fold_cv/`
- `USE_CRF = True` → `bertic_crf_5fold_cv/`

This keeps results separate for easy comparison.

## Configuration Cell Location

Look for this cell in the notebook (around line 331):

```python
# ============================================================================
# CRF CONFIGURATION - Set to False to test BERT logits alone (argmax decoding)
# ============================================================================
USE_CRF = False  # Set to True to enable CRF layer, False for standard BERT (argmax)
```

Just change `False` to `True` or vice versa, then run all cells.

## Debugging Tips

### If BERT-only results don't match base:

1. **Check tokenization output:**
   - Look for the debug print: `🔍 DEBUG - After tokenization:`
   - Compare train/val tokenized counts with base notebook

2. **Check model architecture:**
   - Verify dropout rate matches base
   - Verify classifier layer is identical

3. **Check training arguments:**
   - Learning rate, batch size, epochs should match

### If CRF results are catastrophic:

1. **Check mask alignment:**
   - Add debug prints in forward() and predict()
   - Verify mask shapes and values match

2. **Check label handling:**
   - Verify -100 labels are properly masked
   - Check if CRF sees padding tokens

3. **Check decoding:**
   - Verify CRF.decode() returns valid label IDs
   - Check if predictions align with input sequences

## Summary

This modification allows you to **isolate the problem** by testing BERT and CRF independently:

- **`USE_CRF = False`** → Tests BERT implementation (should match base)
- **`USE_CRF = True`** → Tests full BERT-CRF (currently broken)

By comparing results, you can definitively determine whether the issue is in the BERT part or the CRF layer.

