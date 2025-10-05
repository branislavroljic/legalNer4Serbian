# CRF Implementation Fix - Critical Bug Resolution

## The Problem

The original CRF implementation had **catastrophic failures** on B-tags (beginning of entities):

```
B-CASE_NUMBER:  Precision 0.0141 (1.4%!)
B-JUDGE:        Precision 0.0000 (completely failed)
B-PROSECUTOR:   Precision 0.0000 (completely failed)
B-REGISTRAR:    Precision 0.0000 (completely failed)
B-DECISION_DATE: Recall 0.2544 (missing 75% of entities)
```

Meanwhile, I-tags (inside entities) performed well:
```
I-PROVISION_MATERIAL:    F1 0.9170
I-CRIMINAL_ACT:          F1 0.9255
I-PROVISION_PROCEDURAL:  F1 0.8849
```

**This pattern indicates the CRF was learning to avoid predicting entity boundaries.**

## Root Cause

### The Bug

**Original (BROKEN) Code:**
```python
def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
    outputs = self.bertic(input_ids, attention_mask, return_dict=True)
    logits = outputs.logits
    
    loss = None
    if labels is not None:
        # BUG: Using attention_mask instead of label-based mask
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones_like(labels, dtype=torch.bool)
        
        # BUG: Replacing -100 with 0, but mask includes special tokens
        labels_for_crf = labels.clone()
        labels_for_crf[labels == -100] = 0
        
        # CRF learns from WRONG positions (includes CLS, SEP, PAD)
        log_likelihood = self.crf(logits, labels_for_crf, mask=mask)
        loss = -log_likelihood
```

### Why This Failed

1. **Wrong Mask Source**: 
   - `attention_mask` includes ALL tokens: `[CLS] token1 token2 ... [SEP] [PAD] [PAD]`
   - But `labels` has `-100` for special tokens: `[-100, label1, label2, ..., -100, -100, -100]`
   - The CRF was learning from positions where labels are `-100` (special tokens)!

2. **Confusion During Training**:
   - Special tokens (CLS, SEP) are replaced with label `0` (O tag)
   - CRF sees: "At position 0 (CLS), the label is O"
   - CRF learns: "Start of sequence → O tag" (WRONG!)
   - This biases the model against predicting B-tags at actual entity starts

3. **Invalid Transition Learning**:
   - CRF learned transitions like: `[CLS] → O` instead of `O → B-ENTITY`
   - This corrupted the transition matrix
   - Result: Model avoids predicting B-tags, prefers I-tags or O

## The Fix

### Fixed Code

```python
def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
    outputs = self.bertic(input_ids, attention_mask, return_dict=True)
    logits = outputs.logits
    
    loss = None
    if labels is not None:
        # FIX: Create mask based on valid labels, NOT attention_mask
        # -100 indicates special tokens (CLS, SEP, PAD) that should be ignored
        mask = (labels != -100)
        
        # Replace -100 with 0 (O tag) for CRF compatibility
        # These positions will be masked out, so the value doesn't affect training
        labels_for_crf = labels.clone()
        labels_for_crf[~mask] = 0
        
        # Compute CRF loss with proper masking
        # The mask ensures CRF only learns from actual tokens, not special tokens
        log_likelihood = self.crf(logits, labels_for_crf, mask=mask, reduction='mean')
        loss = -log_likelihood
```

### What Changed

| Aspect | Before (BROKEN) | After (FIXED) |
|--------|----------------|---------------|
| **Mask Source** | `attention_mask.bool()` | `(labels != -100)` |
| **Masked Positions** | Only PAD tokens | CLS, SEP, PAD (all -100 labels) |
| **CRF Training** | Learns from special tokens | Learns only from actual tokens |
| **Transition Matrix** | Corrupted by special tokens | Clean, valid BIO transitions |

## Visual Example

### Token Sequence
```
Tokens:     [CLS]  Основни  суд   у    Београду  [SEP]  [PAD]  [PAD]
Labels:     -100   B-COURT  I-COURT I-COURT I-COURT  -100   -100   -100
Attention:   1      1        1       1       1        1      0      0
```

### Before Fix (BROKEN)
```python
mask = attention_mask.bool()
# mask = [True, True, True, True, True, True, False, False]

labels_for_crf = [-100, B-COURT, I-COURT, I-COURT, I-COURT, -100, -100, -100]
labels_for_crf[labels == -100] = 0
# labels_for_crf = [O, B-COURT, I-COURT, I-COURT, I-COURT, O, O, O]

# CRF learns from positions 0-5:
# Position 0 (CLS): O tag ← WRONG! This is a special token
# Position 5 (SEP): O tag ← WRONG! This is a special token
# CRF learns: "Sequence boundaries → O tag"
# Result: Avoids predicting B-tags at real entity starts
```

### After Fix (CORRECT)
```python
mask = (labels != -100)
# mask = [False, True, True, True, True, False, False, False]

labels_for_crf = labels.clone()
labels_for_crf[~mask] = 0
# labels_for_crf = [O, B-COURT, I-COURT, I-COURT, I-COURT, O, O, O]

# CRF learns ONLY from positions 1-4:
# Position 1: B-COURT ← CORRECT! Real token
# Position 2-4: I-COURT ← CORRECT! Real tokens
# CRF learns: "O → B-COURT → I-COURT → I-COURT → I-COURT"
# Result: Proper BIO sequence learning
```

## Expected Improvements

With this fix, the CRF should now:

1. **Properly Learn B-tags**: 
   - No longer biased against entity boundaries
   - Should achieve >80% precision on B-tags

2. **Valid BIO Transitions**:
   - `O → B-ENTITY` (start entity)
   - `B-ENTITY → I-ENTITY` (continue entity)
   - `I-ENTITY → I-ENTITY` (continue entity)
   - `I-ENTITY → O` (end entity)

3. **Better Overall Performance**:
   - Expected F1: 0.84-0.86 (vs base BERTić 0.83)
   - Improvement especially on multi-token entities
   - More consistent entity boundary detection

## Verification Steps

After running the fixed notebook, check:

1. **B-tag Performance**:
   ```
   B-COURT:     Should be >0.85 precision
   B-JUDGE:     Should be >0.80 precision
   B-PROSECUTOR: Should be >0.70 precision
   ```

2. **Transition Matrix** (if you can inspect it):
   - High probability: `O → B-*`, `B-* → I-*`
   - Low probability: `O → I-*`, `B-COURT → I-JUDGE`

3. **Overall Metrics**:
   - F1 should be ≥ base BERTić (0.83)
   - Precision and recall should be balanced
   - No catastrophic failures on any entity type

## Technical Details

### Why `labels != -100` Works

In HuggingFace transformers:
- `-100` is the ignore index for loss computation
- It's used for:
  - Special tokens (CLS, SEP)
  - Padding tokens (PAD)
  - Subword tokens (after the first subword)

By masking where `labels != -100`, we ensure:
- CRF only sees actual labeled tokens
- No learning from special tokens
- Proper BIO sequence modeling

### CRF Loss Computation

```python
log_likelihood = self.crf(logits, labels_for_crf, mask=mask, reduction='mean')
loss = -log_likelihood
```

- `logits`: Emission scores from BERTić (shape: [batch, seq_len, num_labels])
- `labels_for_crf`: Target labels (shape: [batch, seq_len])
- `mask`: Valid positions (shape: [batch, seq_len])
- `reduction='mean'`: Average loss across batch and sequence

The CRF computes:
1. **Emission scores**: From BERTić logits
2. **Transition scores**: Learned by CRF layer
3. **Total score**: Emission + Transition for the gold sequence
4. **Partition function**: Sum over all possible sequences
5. **Log-likelihood**: log(gold_score / partition)

With proper masking, the CRF only considers valid positions when computing the partition function.

## Comparison: Before vs After

| Metric | Before Fix | After Fix (Expected) |
|--------|-----------|---------------------|
| B-CASE_NUMBER Precision | 0.0141 | >0.70 |
| B-JUDGE Precision | 0.0000 | >0.80 |
| B-PROSECUTOR Precision | 0.0000 | >0.70 |
| B-REGISTRAR Precision | 0.0000 | >0.80 |
| Overall F1 | ~0.51 | >0.83 |
| Training Stability | Unstable | Stable |

## Files Modified

1. **`serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb`**:
   - Fixed `BerticCRF.forward()` method
   - Updated mask creation: `mask = (labels != -100)`
   - Added explanatory comments

## Next Steps

1. **Run the Fixed Notebook**:
   ```bash
   # In Google Colab
   # Upload: serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb
   # Run all cells
   ```

2. **Compare Results**:
   - Base BERTić: F1 ~0.83
   - Fixed CRF: Should be F1 ~0.84-0.86

3. **If Still Underperforming**:
   - Check CRF transition matrix
   - Verify label distribution in training data
   - Consider adjusting learning rate for CRF layer

## References

- pytorch-crf documentation: https://pytorch-crf.readthedocs.io/
- HuggingFace ignore_index: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.label_smoothing_factor
- BIO tagging scheme: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

