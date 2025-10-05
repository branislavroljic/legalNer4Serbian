# BERT-CRF Critical Fix: Random vs Pretrained Weights

## Problem Identified

The BERT-CRF notebook was performing **catastrophically worse** than the base notebook, even with CRF disabled (`USE_CRF = False`):

### Performance Comparison:

| Entity | Base F1 | BERT-CRF (no CRF) F1 | Difference |
|--------|---------|---------------------|------------|
| B-CASE_NUMBER | 0.75 | **0.00** | ❌ -0.75 |
| B-PROSECUTOR | 0.54 | **0.00** | ❌ -0.54 |
| I-DECISION_DATE | 0.92 | **0.00** | ❌ -0.92 |
| I-PROSECUTOR | 0.59 | **0.00** | ❌ -0.59 |
| B-REGISTRAR | 0.92 | **0.40** | ❌ -0.52 |
| I-REGISTRAR | 0.92 | **0.05** | ❌ -0.87 |
| **Macro Avg** | **0.88** | **0.66** | ❌ **-0.22** |

## Root Cause

The custom `BertCrfForTokenClassification` class was using:

### ❌ WRONG Implementation (Before):

```python
def __init__(self, config, num_labels, use_crf=True):
    super().__init__()
    
    # WRONG: Uses base BERT without classification head
    self.bert = AutoModel.from_pretrained(MODEL_NAME, config=config)
    
    # WRONG: Random initialization of classifier
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    # CRF layer
    if use_crf:
        self.crf = CRF(num_labels, batch_first=True)
```

**Problems:**
1. **`AutoModel`** returns only BERT embeddings, no classification head
2. **`nn.Linear()`** creates a classifier with **random weights**
3. Model starts training from **scratch** instead of using pretrained weights
4. This is why performance was so poor - the classifier had no prior knowledge

### ✅ CORRECT Implementation (After):

```python
def __init__(self, model_name, num_labels, id2label, label2id, use_crf=True):
    super().__init__()
    
    # CORRECT: Load pretrained model with classification head
    self.bert_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Extract config
    self.config = self.bert_model.config
    
    # CRF layer (optional)
    if use_crf:
        self.crf = CRF(num_labels, batch_first=True)
```

**Benefits:**
1. **`AutoModelForTokenClassification`** includes pretrained classification head
2. Classifier weights are **initialized from pretraining**, not random
3. Model starts with **knowledge transfer** from pretraining
4. Should match base notebook performance when `use_crf=False`

## Changes Made

### 1. Model Initialization

**Before:**
```python
self.bert = AutoModel.from_pretrained(MODEL_NAME, config=config)
self.dropout = nn.Dropout(config.hidden_dropout_prob)
self.classifier = nn.Linear(config.hidden_size, num_labels)
```

**After:**
```python
self.bert_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
```

### 2. Forward Pass

**Before:**
```python
outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
sequence_output = outputs.last_hidden_state
sequence_output = self.dropout(sequence_output)
logits = self.classifier(sequence_output)
```

**After:**
```python
outputs = self.bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=None,  # Don't compute loss yet
    return_dict=True
)
logits = outputs.logits  # Get logits from pretrained head
```

### 3. Prediction Method

**Before:**
```python
outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
sequence_output = self.dropout(outputs.last_hidden_state)
logits = self.classifier(sequence_output)
```

**After:**
```python
outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
logits = outputs.logits  # Get logits from pretrained head
```

### 4. Model Creation Call

**Before:**
```python
config = AutoConfig.from_pretrained(MODEL_NAME)
model = BertCrfForTokenClassification(config, ner_dataset.get_num_labels(), use_crf=USE_CRF)
```

**After:**
```python
model = BertCrfForTokenClassification(
    MODEL_NAME,
    ner_dataset.get_num_labels(),
    ner_dataset.id_to_label,
    ner_dataset.label_to_id,
    use_crf=USE_CRF
)
```

## How Base Notebook Does It

The base 5-fold notebook uses the shared `load_model_and_tokenizer()` function:

```python
# From ner/shared/model_utils.py
def load_model_and_tokenizer(model_name, num_labels, id2label, label2id):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model, tokenizer
```

This ensures the model starts with **pretrained weights**, not random initialization.

## Expected Results After Fix

### With `USE_CRF = False` (BERT only):
- Should **match base notebook** performance (F1 ~0.88)
- Confirms BERT implementation is correct
- Proves the issue was in initialization, not architecture

### With `USE_CRF = True` (BERT + CRF):
- Should **improve or match** BERT-only performance
- If still worse, indicates CRF implementation issues
- If better, CRF is working correctly!

## Testing Instructions

1. **Set `USE_CRF = False`** in the configuration cell
2. **Run the notebook** (can use `N_FOLDS = 2` for quick test)
3. **Compare with base notebook results**
4. **Expected:** Macro F1 should be ~0.88 (similar to base)

If results now match the base notebook, the fix is successful!

## Why This Matters

### Random Initialization (Before):
- Classifier starts with **no knowledge**
- Must learn everything from scratch
- Requires **much more data** to converge
- Poor performance on small datasets
- Unstable training

### Pretrained Initialization (After):
- Classifier starts with **transfer learning**
- Leverages knowledge from pretraining
- Requires **less data** to fine-tune
- Better performance on small datasets
- Stable training

## Analogy

**Before (Random Init):**
- Like asking someone who's never seen legal documents to classify entities
- They have to learn everything from 225 examples
- Very difficult, poor results

**After (Pretrained):**
- Like asking a lawyer to classify entities in legal documents
- They already understand language and legal concepts
- Just need to fine-tune on specific entity types
- Much easier, better results

## Key Takeaway

**Always use `AutoModelForTokenClassification.from_pretrained()` for token classification tasks**, not `AutoModel`. The pretrained classification head provides crucial initialization that dramatically improves performance, especially on small datasets.

## Files Modified

- `ner/notebook_base_k-fold/serbian_legal_ner_pipeline_bert_crf_5fold_cv_2.ipynb`
  - Updated `BertCrfForTokenClassification.__init__()`
  - Updated `BertCrfForTokenClassification.forward()`
  - Updated `BertCrfForTokenClassification.predict()`
  - Updated `create_bert_crf_model_and_trainer()`

## Next Steps

1. ✅ Run with `USE_CRF = False` to verify BERT matches base
2. ⏳ If BERT matches, run with `USE_CRF = True` to test CRF
3. ⏳ Compare CRF results with BERT-only to evaluate CRF benefit

