# BERTić + CRF 5-Fold Cross-Validation Notebook

## Overview

This notebook (`serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb`) implements a BERTić model with a Conditional Random Field (CRF) layer for Serbian Legal NER, evaluated using 5-fold cross-validation.

## Architecture

The notebook builds directly on top of the base BERTić 5-fold CV pipeline by adding a CRF layer:

```
Input Text
    ↓
BERTić Tokenization
    ↓
BERTić Encoder (fine-tuned)
    ↓
Token Classification Head (logits)
    ↓
CRF Layer (sequence constraints)
    ↓
Final Predictions
```

## Key Differences from Base Notebook

### 1. Model Architecture

**Base Notebook:**
- Uses `AutoModelForTokenClassification` directly
- Predictions via argmax on logits
- No sequence-level constraints

**CRF Notebook:**
- Wraps fine-tuned BERTić in `BerticCRF` class
- Adds CRF layer on top of classification head
- Enforces valid BIO sequence transitions
- Uses Viterbi decoding for predictions

### 2. Model Class Definition

```python
class BerticCRF(nn.Module):
    def __init__(self, bertic_model, num_labels):
        super().__init__()
        self.bertic = bertic_model  # Fine-tuned BERTić
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get BERTić logits
        outputs = self.bertic(input_ids, attention_mask, return_dict=True)
        logits = outputs.logits
        
        # Compute CRF loss if labels provided
        if labels is not None:
            mask = attention_mask.bool()
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            log_likelihood = self.crf(logits, labels_for_crf, mask=mask)
            loss = -log_likelihood
        
        return {'loss': loss, 'logits': logits}
    
    def decode(self, input_ids, attention_mask=None):
        # CRF Viterbi decoding
        outputs = self.bertic(input_ids, attention_mask, return_dict=True)
        predictions = self.crf.decode(outputs.logits, mask=attention_mask.bool())
        return predictions
```

### 3. Training Process

**Both notebooks:**
- Use same data preprocessing
- Same sliding window tokenization
- Same 5-fold cross-validation splits
- Same training hyperparameters

**CRF-specific:**
- CRF loss instead of cross-entropy
- CRF decoding during evaluation
- Custom evaluation loop to handle CRF predictions

### 4. Evaluation

**Base Notebook:**
```python
# Uses detailed_evaluation from shared utilities
eval_results = detailed_evaluation(
    trainer, val_dataset, f"Fold {fold_num} Validation", ner_dataset.id_to_label
)
```

**CRF Notebook:**
```python
# Custom evaluation with CRF decoding
model.eval()
for batch in trainer.get_eval_dataloader():
    # Get CRF predictions using Viterbi algorithm
    predictions = model.decode(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    # Process and collect predictions
    ...
```

## Why CRF Layer?

### Advantages

1. **Sequence Constraints**: Enforces valid BIO tag transitions (e.g., I-COURT must follow B-COURT)
2. **Global Optimization**: Considers entire sequence when making predictions
3. **Better Boundaries**: More accurate entity span detection
4. **Theoretical Soundness**: Probabilistically sound sequence labeling

### Expected Improvements

- Better handling of entity boundaries
- Fewer invalid tag sequences
- Improved F1 scores, especially for multi-token entities
- More consistent predictions across similar contexts

## Usage

### Running the Notebook

1. **Install Dependencies:**
   ```bash
   pip install transformers torch datasets tokenizers scikit-learn seqeval pandas numpy matplotlib seaborn tqdm pytorch-crf
   ```

2. **Run in Google Colab:**
   - Upload notebook to Colab
   - Mount Google Drive with datasets
   - Run all cells sequentially

3. **Expected Output:**
   - 5 trained models (one per fold)
   - Aggregate metrics across folds
   - Detailed classification reports
   - Confusion matrices

### File Structure

```
/storage/models/bertic_crf_5fold_cv/
├── fold_1/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── aggregate_results.json
```

## Comparison with Base Notebook

To compare results:

1. **Run Base Notebook:**
   - `serbian_legal_ner_pipeline_bеrtic_base_5fold_cv.ipynb`
   - Note F1 scores per fold

2. **Run CRF Notebook:**
   - `serbian_legal_ner_pipeline_bertic_crf_5fold_cv.ipynb`
   - Note F1 scores per fold

3. **Compare:**
   - Overall F1 score (mean ± std)
   - Per-entity F1 scores
   - Precision/Recall trade-offs
   - Training time differences

## Implementation Details

### Dependencies

- `pytorch-crf`: CRF layer implementation
- `transformers`: BERTić model
- `torch`: PyTorch framework
- `seqeval`: NER-specific metrics

### Key Functions

1. **`BerticCRF`**: Model class wrapping BERTić with CRF
2. **`create_model_and_trainer`**: Creates BERTić-CRF model for each fold
3. **`train_and_evaluate_fold`**: Trains and evaluates with CRF decoding
4. **`prepare_fold_data`**: Same as base (unchanged)

### Training Configuration

```python
{
    'num_epochs': 8,
    'batch_size': 8,
    'learning_rate': 3e-5,
    'max_length': 512,
    'stride': 128,
    'warmup_steps': 500,
    'weight_decay': 0.01
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size to 4
   - Reduce max_length to 256

2. **CRF Decoding Errors:**
   - Ensure attention_mask is properly set
   - Check that labels don't contain -100 during CRF forward pass

3. **Slow Training:**
   - CRF adds computational overhead
   - Expected to be ~20-30% slower than base

## Expected Results

Based on the base notebook results (F1: ~0.83), the CRF version should achieve:

- **Overall F1**: 0.84-0.86 (improvement of 1-3%)
- **Precision**: Slight increase
- **Recall**: Slight increase
- **Entity Boundaries**: Significant improvement

## References

- Base notebook: `serbian_legal_ner_pipeline_bеrtic_base_5fold_cv.ipynb`
- BERTić model: https://huggingface.co/classla/bcms-bertic
- pytorch-crf: https://pytorch-crf.readthedocs.io/

## Notes

- This notebook maintains identical pipeline structure to the base notebook
- Only the model architecture and evaluation differ
- All data preprocessing, tokenization, and metrics are the same
- Results are directly comparable to base notebook

