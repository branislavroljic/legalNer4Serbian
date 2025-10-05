# Class Weights Implementation for BERTić Base NER

## Overview

The `bertic_base_class_weights.ipynb` notebook is a modified version of `bertic_base.ipynb` that implements **class weights** to address the class imbalance problem observed in the Serbian legal NER dataset.

## Motivation

The original `bertic_base.ipynb` notebook showed significant class imbalance in the entity distribution:

- **High-frequency entities**: DEFENDANT (1240), PROVISION_MATERIAL (1177), CRIMINAL_ACT (792)
- **Low-frequency entities**: CASE_NUMBER (225), PROCEDURE_COSTS (231), VERDICT (238)

This imbalance can cause the model to be biased towards predicting high-frequency entities while performing poorly on rare entities. Class weights help mitigate this by assigning higher importance to underrepresented classes during training.

## Key Differences from Original Notebook

### 1. **Additional Imports**
```python
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from collections import Counter
```

### 2. **Class Weights Calculation Function**
A new function `calculate_class_weights_from_tokenized()` that:
- Collects all label IDs from tokenized training examples
- Filters out padding tokens (label ID = -100)
- Uses sklearn's `compute_class_weight` with `'balanced'` strategy
- Returns a PyTorch tensor of class weights

The balanced strategy computes weights as:
```
weight[i] = n_samples / (n_classes * n_samples_for_class[i])
```

### 3. **Custom WeightedTrainer Class**
A custom trainer that extends HuggingFace's `Trainer` class:
- Accepts `class_weights` parameter in constructor
- Overrides `compute_loss()` method to use weighted CrossEntropyLoss
- Automatically moves class weights to the correct device (CPU/GPU)
- Properly handles the ignore index (-100) for padding tokens

### 4. **Modified Data Preparation Function**
`prepare_fold_data_with_class_weights()` now:
- Tokenizes training and validation data
- **Calculates class weights from training data only** (to avoid data leakage)
- Returns class weights along with datasets and data collator

### 5. **Modified Trainer Creation Function**
`create_model_and_weighted_trainer()` now:
- Accepts `class_weights` parameter
- Creates a `WeightedTrainer` instead of standard `Trainer`
- Passes class weights to the custom trainer

### 6. **Updated Output Directory**
```python
OUTPUT_DIR = f"{paths['models_dir']}/bertic_base_class_weights_5fold_cv"
```

## Implementation Details

### Class Weights Calculation (Per Fold)

For each fold, class weights are calculated from the **training data only**:

1. Tokenize training examples with sliding window
2. Extract all label IDs (excluding -100 padding tokens)
3. Count frequency of each label
4. Apply balanced weighting formula
5. Convert to PyTorch tensor

### Weighted Loss Function

The custom trainer uses `nn.CrossEntropyLoss` with weights:

```python
loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
```

This ensures:
- Rare classes contribute more to the loss
- Common classes contribute less to the loss
- Padding tokens are ignored (via ignore_index=-100)

## Expected Benefits

1. **Improved performance on rare entities**: CASE_NUMBER, PROCEDURE_COSTS, VERDICT, SANCTION_TYPE
2. **More balanced predictions**: Model should not be biased towards high-frequency entities
3. **Better generalization**: Especially for underrepresented entity types

## Usage

The notebook follows the same structure as the original:

1. Run all cells sequentially
2. The notebook will perform 5-fold cross-validation
3. For each fold:
   - Calculate class weights from training data
   - Train model with weighted loss
   - Evaluate on validation set
4. Aggregate results across all folds

## Output

Results are saved to:
```
/storage/models/bertic_base_class_weights_5fold_cv/
├── fold_1/
│   ├── pytorch_model.bin
│   ├── classification_report.json
│   └── ...
├── fold_2/
│   └── ...
├── ...
└── aggregate_classification_report.json
```

## Comparison with Original

To compare the effectiveness of class weights:

1. Run `bertic_base.ipynb` (baseline without class weights)
2. Run `bertic_base_class_weights.ipynb` (with class weights)
3. Compare the aggregate F1-scores, especially for rare entity types

Expected improvements should be visible in:
- Per-entity F1-scores for rare classes
- Overall recall (model should detect more rare entities)
- Balanced performance across all entity types

## Technical Notes

- **Class weights are fold-specific**: Each fold calculates its own weights based on the training split
- **No data leakage**: Weights are computed only from training data, never from validation data
- **Device handling**: Class weights are automatically moved to GPU/CPU as needed
- **Backward compatibility**: Uses the same shared modules and evaluation functions as the original notebook

