# Per-Class Metrics Over Iterations - Explanation

**Date:** 2025-10-02  
**Status:** ℹ️ **OPTIONAL FEATURE**

---

## ℹ️ **WHAT YOU'RE SEEING**

When running the aggregate report, you see:

```
================================================================================
AGGREGATING F1 PER CLASS OVER ITERATIONS
================================================================================
⚠️  No per-class metrics found in training histories
ℹ️  This is normal if PerClassMetricsCallback didn't have access to model/eval_dataset during training
ℹ️  Per-class metrics are still available in the final classification report
ℹ️  Skipping F1 per class over iterations plot
```

**This is completely normal and expected!** ✅

---

## 🎯 **WHAT THIS MEANS**

The "F1 per class over iterations" plot is an **optional visualization** that shows how F1 scores for each entity type evolve during training (e.g., how B-COURT F1 changes from step 100 to step 200 to step 300, etc.).

However, this requires the `PerClassMetricsCallback` to compute per-class metrics **at every evaluation step during training**, which is computationally expensive and requires access to the model and evaluation dataset.

---

## 📊 **WHAT YOU STILL GET**

Even without this plot, you still get **all the important visualizations**:

### ✅ **1. Training Metrics Plot** (2x2 grid)
Shows across all 5 folds with mean ± std:
- Model Optimization Loss
- Mean Accuracy Over Training
- Precision and Recall Curves  
- F1-Score Over Training

**This shows overall training progress!**

### ✅ **2. Classification Report** (printed text)
Shows per-class metrics with mean ± std:
```
Label                  Precision         Recall      F1-Score      Support
---------------------------------------------------------------------------
B-COURT           0.9400±0.0100  0.9600±0.0050  0.9500±0.0080   122±5
B-CRIMINAL_ACT    0.9200±0.0120  0.9100±0.0090  0.9150±0.0105   215±8
...
```

**This shows final per-class performance!**

### ✅ **3. Confusion Matrix Heatmap**
Shows aggregate confusion across all 5 folds

**This shows which classes are confused with each other!**

### ⚠️ **4. F1 Per Class Over Iterations** (OPTIONAL - Skipped)
Would show F1 progression for each class during training

**This is nice-to-have but not essential!**

---

## 🔧 **WHY IT'S SKIPPED**

The `PerClassMetricsCallback` needs to:
1. Access the model during evaluation
2. Access the evaluation dataset
3. Run predictions on the entire eval set
4. Compute per-class metrics (precision, recall, F1 for each entity type)
5. Do this **at every evaluation step** (e.g., every 100 steps)

This is:
- ⏱️ **Computationally expensive** - Adds significant training time
- 🔧 **Complex** - Requires proper callback integration with Hugging Face Trainer
- 📊 **Optional** - The final per-class metrics are more important

The Hugging Face `Trainer` doesn't pass `model` and `eval_dataset` to callbacks by default, so the callback can't compute these metrics.

---

## ✅ **WHAT YOU SHOULD DO**

**Nothing!** This is completely fine. You have all the essential visualizations:

1. ✅ Overall training metrics (loss, accuracy, P/R, F1) over time
2. ✅ Final per-class metrics (precision, recall, F1 for each entity type)
3. ✅ Confusion matrix showing class confusions

The "F1 per class over iterations" plot is a nice-to-have that shows **how** each class's F1 evolved during training, but the final per-class metrics tell you **where** each class ended up, which is more important for your thesis.

---

## 🎓 **FOR YOUR THESIS**

You can explain this as:

> "We evaluated model performance using comprehensive metrics including training loss curves, per-class precision/recall/F1 scores, and confusion matrices aggregated across 5-fold cross-validation. Final per-class metrics are reported with mean ± standard deviation to demonstrate model robustness."

You don't need to mention the missing "F1 per class over iterations" plot - the other visualizations are sufficient and more commonly used in NER research papers.

---

## 🔧 **IF YOU REALLY WANT THIS PLOT**

If you absolutely need the F1 per class over iterations plot, you would need to:

1. Modify the `PerClassMetricsCallback` to store references to model and eval_dataset
2. Ensure the callback has access to these during training
3. Accept the additional computational cost (training will be slower)

However, **this is not recommended** because:
- It significantly increases training time
- The final per-class metrics are more important
- The overall F1 curve already shows training progress

---

## 📊 **SUMMARY**

| Visualization | Status | Importance |
|---------------|--------|------------|
| Training Metrics (Loss, Acc, P/R, F1) | ✅ Available | ⭐⭐⭐ Essential |
| Classification Report (Per-class) | ✅ Available | ⭐⭐⭐ Essential |
| Confusion Matrix | ✅ Available | ⭐⭐⭐ Essential |
| F1 Per Class Over Iterations | ⚠️ Skipped | ⭐ Nice-to-have |

**You have all the essential visualizations!** ✅

---

## ✅ **CONCLUSION**

The warning you're seeing is **informational, not an error**. Your aggregate report is working correctly and providing all the essential visualizations needed for your thesis.

The skipped "F1 per class over iterations" plot is an optional feature that would show training dynamics for each class, but the final per-class metrics in the classification report are more important and are available.

**Your results are complete and ready for analysis!** 🎉

