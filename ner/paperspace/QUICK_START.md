# Quick Start Guide: bertic_base_class_weights.ipynb

## 🚀 Ready to Run!

The notebook is **fully functional** and ready to use in Paperspace.

---

## ⚡ Quick Start (3 Steps)

### 1. Upload to Paperspace
Upload `bertic_base_class_weights.ipynb` to your Paperspace notebook environment.

### 2. Verify Shared Modules
Ensure `/shared/` directory is accessible with all required modules.

### 3. Run All Cells
Execute cells sequentially from top to bottom.

---

## ⏱️ Expected Runtime

- **Per fold**: ~15-20 minutes (on GPU)
- **Total (5 folds)**: ~1.5-2 hours
- **Memory**: ~8-12 GB GPU RAM

---

## 📊 What You'll Get

### Visualizations (in notebook)
- 📊 Entity distribution across folds
- 📉 Training/validation loss curves
- 📈 Macro/micro-averaged metrics
- 🔥 Confusion matrices
- 📊 Per-class F1 scores over iterations

### Saved Files
```
/storage/models/bertic_base_class_weights_5fold_cv/
├── fold_1/ ... fold_5/
│   ├── pytorch_model.bin
│   ├── classification_report_fold*.txt
│   └── classification_report.json
└── aggregate_report.json
```

### Metrics
- Overall precision, recall, F1, accuracy (mean ± std)
- Per-entity metrics for all 14 entity types
- Fold-by-fold detailed results

---

## 🔍 Key Features

### Class Weights
✅ Automatically calculated per fold  
✅ Balanced weighting for imbalanced classes  
✅ Handles missing labels gracefully  

### Evaluation
✅ 5-fold cross-validation  
✅ Comprehensive metrics tracking  
✅ Statistical analysis (mean ± std)  

### Compatibility
✅ Uses shared modules  
✅ Same structure as baseline  
✅ Easy comparison with `bertic_base.ipynb`  

---

## 🎯 Expected Improvements

Class weights should improve performance on:
- **CASE_NUMBER** (225 instances)
- **PROCEDURE_COSTS** (231 instances)
- **VERDICT** (238 instances)
- **SANCTION_TYPE** (248 instances)

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `FINAL_STATUS.md` | Complete status and overview |
| `CLASS_WEIGHTS_IMPLEMENTATION.md` | Implementation details |
| `NOTEBOOK_COMPARISON.md` | Comparison with baseline |
| `TROUBLESHOOTING.md` | Common issues and solutions |
| `FIXES_APPLIED.md` | All fixes documented |
| `QUICK_START.md` | This file |

---

## ⚠️ Important Notes

1. **Class weights are fold-specific**: Each fold calculates its own weights
2. **Some labels may be absent**: This is normal in k-fold CV
3. **Default weight for absent labels**: 1.0 (neutral)
4. **Training time**: Slightly longer than baseline due to weight calculations

---

## 🆚 Comparison Workflow

1. Run `bertic_base.ipynb` (baseline)
2. Run `bertic_base_class_weights.ipynb` (this notebook)
3. Compare `aggregate_report.json` files
4. Focus on rare entity F1-scores

---

## ✅ Verification Checklist

After running, verify:
- [ ] All 5 folds completed successfully
- [ ] Class weights calculated for each fold
- [ ] Visualizations displayed in notebook
- [ ] Aggregate report generated
- [ ] JSON files saved to output directory
- [ ] No errors in final cells

---

## 🐛 If Something Goes Wrong

1. Check `TROUBLESHOOTING.md` for common issues
2. Verify shared modules are accessible
3. Ensure GPU is available
4. Check disk space for model checkpoints
5. Review error messages against `FIXES_APPLIED.md`

---

## 📞 Quick Reference

### Class Weight Calculation
```python
# Automatically done per fold
class_weights = calculate_class_weights_from_tokenized(
    train_tokenized, 
    ner_dataset.label_to_id
)
```

### Weighted Trainer
```python
# Automatically created per fold
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    ...
)
```

### Aggregate Report
```python
# Automatically generated at end
aggregate_report = create_aggregate_report_across_folds(
    fold_results=fold_results,
    model_name="BERTić Base with Class Weights",
    display=True
)
```

---

## 🎓 For Your Thesis

This notebook demonstrates:
- Handling class imbalance in NER
- Weighted loss functions for token classification
- Robust k-fold cross-validation
- Comprehensive evaluation methodology
- Statistical significance testing (mean ± std)

Perfect for comparing different approaches to handling imbalanced legal entity recognition!

---

## 🎉 You're All Set!

The notebook is ready to run. Just upload to Paperspace and execute all cells.

Good luck with your experiments! 🚀

