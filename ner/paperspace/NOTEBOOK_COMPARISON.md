# Notebook Comparison: bertic_base.ipynb vs bertic_base_class_weights.ipynb

## Quick Reference

| Aspect | bertic_base.ipynb | bertic_base_class_weights.ipynb |
|--------|-------------------|----------------------------------|
| **Purpose** | Baseline BERTić fine-tuning | BERTić with class weights for imbalance |
| **Loss Function** | Standard CrossEntropyLoss | Weighted CrossEntropyLoss |
| **Trainer** | Standard HuggingFace Trainer | Custom WeightedTrainer |
| **Class Imbalance Handling** | None | Balanced class weights per fold |
| **Output Directory** | `bertic_base_5fold_cv` | `bertic_base_class_weights_5fold_cv` |

## Detailed Comparison

### Section-by-Section Changes

#### Section 1-5: Identical
Both notebooks share the same:
- Environment setup and dependencies
- Configuration
- Data loading and analysis
- Data preprocessing and BIO conversion
- Dataset preparation

#### Section 6: **NEW in Class Weights Notebook**
**Class Weights Implementation**
- `calculate_class_weights_from_tokenized()` function
- `WeightedTrainer` class with custom `compute_loss()` method

#### Section 7: Identical
K-Fold Cross-Validation Setup

#### Section 8: Modified
**K-Fold Helper Functions**

| Function | Original | Class Weights Version |
|----------|----------|----------------------|
| Data preparation | `prepare_fold_data()` | `prepare_fold_data_with_class_weights()` |
| Returns | `(train_dataset, val_dataset, data_collator)` | `(train_dataset, val_dataset, data_collator, class_weights)` |
| Trainer creation | `create_model_and_trainer()` | `create_model_and_weighted_trainer()` |
| Trainer type | `Trainer` (from shared.model_utils) | `WeightedTrainer` (custom) |

#### Section 9: Modified
**Training Loop**
- Uses `prepare_fold_data_with_class_weights()` instead of `prepare_fold_data()`
- Uses `create_model_and_weighted_trainer()` instead of `create_model_and_trainer()`
- Passes `class_weights` to trainer creation
- Prints "with class weights" in status messages

#### Section 10: Identical
Aggregate Results Across Folds

## Code Differences

### 1. Additional Imports

**Class Weights Notebook Only:**
```python
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from collections import Counter
```

### 2. New Class Weights Function

**Class Weights Notebook Only:**
```python
def calculate_class_weights_from_tokenized(tokenized_examples, label_to_id):
    """Calculate class weights based on label frequency in tokenized training data."""
    all_label_ids = []
    for example in tokenized_examples:
        valid_labels = [label for label in example['labels'] if label != -100]
        all_label_ids.extend(valid_labels)
    
    label_counts = Counter(all_label_ids)
    classes = np.array(list(range(len(label_to_id))))
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.array(all_label_ids)
    )
    
    return torch.FloatTensor(class_weights)
```

### 3. Custom Trainer Class

**Class Weights Notebook Only:**
```python
class WeightedTrainer(Trainer):
    """Custom Trainer that uses weighted CrossEntropyLoss for handling class imbalance."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted loss for token classification."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            if self.class_weights is not None:
                class_weights = self.class_weights.to(logits.device)
            else:
                class_weights = None
            
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = None
        
        return (loss, outputs) if return_outputs else loss
```

### 4. Modified Data Preparation

**Original:**
```python
def prepare_fold_data(train_examples, val_examples, tokenizer, ner_dataset):
    train_tokenized = tokenize_and_align_labels_with_sliding_window(...)
    val_tokenized = tokenize_and_align_labels_with_sliding_window(...)
    train_dataset, val_dataset, _ = create_huggingface_datasets(...)
    data_collator = DataCollatorForTokenClassification(...)
    return train_dataset, val_dataset, data_collator
```

**Class Weights:**
```python
def prepare_fold_data_with_class_weights(train_examples, val_examples, tokenizer, ner_dataset):
    train_tokenized = tokenize_and_align_labels_with_sliding_window(...)
    val_tokenized = tokenize_and_align_labels_with_sliding_window(...)
    
    # NEW: Calculate class weights from training data
    class_weights = calculate_class_weights_from_tokenized(train_tokenized, ner_dataset.label_to_id)
    
    train_dataset, val_dataset, _ = create_huggingface_datasets(...)
    data_collator = DataCollatorForTokenClassification(...)
    return train_dataset, val_dataset, data_collator, class_weights
```

### 5. Modified Trainer Creation

**Original:**
```python
def create_model_and_trainer(...):
    # ... model creation ...
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        id_to_label=ner_dataset.id_to_label,
        early_stopping_patience=3,
        additional_callbacks=[metrics_callback]
    )
    return model, trainer, metrics_callback, fold_output_dir
```

**Class Weights:**
```python
def create_model_and_weighted_trainer(..., class_weights, ...):
    # ... model creation ...
    
    # NEW: Use WeightedTrainer instead of create_trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,  # NEW parameter
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[metrics_callback]
    )
    return model, trainer, metrics_callback, fold_output_dir
```

## When to Use Each Notebook

### Use `bertic_base.ipynb` when:
- You want a baseline model without any class imbalance handling
- You want to compare against a standard fine-tuning approach
- Your dataset is relatively balanced
- You want faster training (no overhead from class weight computation)

### Use `bertic_base_class_weights.ipynb` when:
- Your dataset has significant class imbalance (as shown in the entity distribution)
- You want to improve performance on rare entity types
- You want more balanced predictions across all entity types
- You're willing to accept slightly longer training time for better balance

## Expected Performance Differences

### Metrics likely to improve with class weights:
- **Recall for rare entities**: CASE_NUMBER, PROCEDURE_COSTS, VERDICT, SANCTION_TYPE
- **F1-score for rare entities**: Better balance between precision and recall
- **Overall recall**: Model should detect more instances of rare entities

### Metrics that might decrease slightly:
- **Precision for common entities**: May decrease slightly as model becomes less biased
- **Overall precision**: Possible slight decrease due to more aggressive predictions

### Overall expected outcome:
- **More balanced performance** across all entity types
- **Better generalization** to rare entity types
- **Potentially higher overall F1-score** due to improved recall on rare classes

## Running Both Notebooks for Comparison

Recommended workflow:
1. Run `bertic_base.ipynb` first (baseline)
2. Run `bertic_base_class_weights.ipynb` second (with class weights)
3. Compare the aggregate reports:
   - `bertic_base_5fold_cv/aggregate_classification_report.json`
   - `bertic_base_class_weights_5fold_cv/aggregate_classification_report.json`
4. Focus comparison on:
   - Per-entity F1-scores (especially rare entities)
   - Overall precision/recall/F1 trade-offs
   - Standard deviations across folds (stability)

