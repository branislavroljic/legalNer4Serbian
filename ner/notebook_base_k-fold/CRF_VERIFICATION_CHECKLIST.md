# CRF Fix Verification Checklist

Use this checklist to verify the CRF fix is working correctly.

## Before Running

- [ ] Notebook file: `serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb`
- [ ] Check line 535: Should say `mask = (labels != -100)`
- [ ] NOT: `mask = attention_mask.bool()`

## During Training

Watch for these signs of correct training:

### ✅ Good Signs
- [ ] Training loss decreases steadily
- [ ] Validation F1 > 0.70 by epoch 3
- [ ] No warnings about invalid CRF transitions
- [ ] GPU memory usage similar to base BERTić

### ❌ Bad Signs (indicates problem)
- [ ] Training loss stuck or increasing
- [ ] Validation F1 < 0.50
- [ ] Warnings about NaN or Inf values
- [ ] Much slower than base BERTić (>2x slower)

## After Training - Fold 1 Results

Check the first fold results immediately:

### B-tag Performance (Critical!)
- [ ] B-COURT precision > 0.80
- [ ] B-JUDGE precision > 0.70
- [ ] B-PROSECUTOR precision > 0.60
- [ ] B-DEFENDANT precision > 0.30
- [ ] B-CRIMINAL_ACT precision > 0.70

**If ANY B-tag has 0.00 precision, STOP - the fix didn't work!**

### I-tag Performance (Should be good)
- [ ] I-PROVISION_MATERIAL F1 > 0.90
- [ ] I-CRIMINAL_ACT F1 > 0.90
- [ ] I-COURT F1 > 0.80

### Overall Metrics
- [ ] Overall F1 > 0.75
- [ ] Precision > 0.70
- [ ] Recall > 0.70
- [ ] Accuracy > 0.95

## After All Folds

### Aggregate Results
- [ ] Mean F1 across folds > 0.80
- [ ] Standard deviation < 0.05
- [ ] All folds have F1 > 0.75

### Comparison with Base BERTić
- [ ] CRF F1 ≥ Base F1 (should be equal or better)
- [ ] CRF has better precision on multi-token entities
- [ ] CRF has fewer invalid BIO sequences

## Expected Results

### Minimum Acceptable Performance
```
Overall F1: > 0.80
B-tag avg precision: > 0.60
I-tag avg precision: > 0.80
```

### Target Performance
```
Overall F1: 0.84-0.86
B-tag avg precision: > 0.70
I-tag avg precision: > 0.85
```

### Comparison Table

| Metric | Base BERTić | Fixed CRF (Expected) |
|--------|-------------|---------------------|
| Overall F1 | 0.83 | 0.84-0.86 |
| B-COURT | 0.90 | 0.90-0.92 |
| B-JUDGE | 0.85 | 0.86-0.88 |
| B-PROSECUTOR | 0.75 | 0.77-0.80 |
| B-DEFENDANT | 0.45 | 0.48-0.52 |

## Troubleshooting

### If B-tags still have 0.00 precision:

1. **Check the mask line**:
   ```python
   # Should be:
   mask = (labels != -100)
   
   # NOT:
   mask = attention_mask.bool()
   ```

2. **Check labels_for_crf**:
   ```python
   # Should be:
   labels_for_crf[~mask] = 0
   
   # NOT:
   labels_for_crf[labels == -100] = 0
   ```

3. **Verify CRF is being used**:
   - Check model creation: `model = BerticCRF(bertic_model, num_labels)`
   - Check forward pass is using CRF loss
   - Check decode is using CRF Viterbi

### If F1 is lower than base BERTić:

1. **Check training epochs**: CRF may need more epochs (try 10-12)
2. **Check learning rate**: Try 2e-5 instead of 3e-5
3. **Check batch size**: Try reducing to 4 if memory allows
4. **Check data**: Ensure same data splits as base

### If training is very slow:

- CRF adds ~20-30% overhead (acceptable)
- If >2x slower, check:
  - Batch size (reduce if needed)
  - GPU memory (may be swapping)
  - Sequence length (reduce max_length if needed)

## Success Criteria

The fix is successful if:

1. ✅ **No B-tags have 0.00 precision**
2. ✅ **Overall F1 ≥ 0.80**
3. ✅ **F1 ≥ Base BERTić F1** (or within 0.02)
4. ✅ **Training completes without errors**
5. ✅ **Results are reproducible across folds**

## Failure Criteria

The fix failed if:

1. ❌ **Any B-tag has 0.00 precision**
2. ❌ **Overall F1 < 0.70**
3. ❌ **F1 much worse than base BERTić** (>0.05 difference)
4. ❌ **Training crashes or produces NaN**
5. ❌ **Results vary wildly across folds** (std > 0.10)

## Next Steps After Verification

### If Successful ✅
- [ ] Document results in thesis
- [ ] Compare with base BERTić in detail
- [ ] Analyze which entity types improved most
- [ ] Consider ensemble with base BERTić

### If Failed ❌
- [ ] Review CRF_FIX_EXPLANATION.md
- [ ] Check code against working base notebook
- [ ] Verify pytorch-crf version (should be latest)
- [ ] Consider alternative CRF implementations

## Quick Reference

### Key Files
- **Notebook**: `serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb`
- **Explanation**: `CRF_FIX_EXPLANATION.md`
- **Summary**: `CRF_FIX_SUMMARY.md`
- **Base (working)**: `serbian_legal_ner_pipeline_bеrtic_base_5fold_cv.ipynb`

### Key Code Lines
- **Line 535**: `mask = (labels != -100)` ← THE FIX
- **Line 540**: `labels_for_crf[~mask] = 0`
- **Line 544**: `log_likelihood = self.crf(logits, labels_for_crf, mask=mask, reduction='mean')`

### Expected Runtime
- Per fold: ~30-45 minutes (GPU)
- Total (5 folds): ~2.5-4 hours
- Slightly slower than base BERTić (acceptable)

## Contact/Notes

If results don't match expectations:
1. Check this checklist
2. Review CRF_FIX_EXPLANATION.md
3. Compare code with base notebook
4. Verify data and splits are identical

