# 🔄 Unrefactoring Guide - Moving Notebook-Specific Code Back

This guide explains what has been "unrefactored" and how to update your notebooks to include notebook-specific code directly.

## ✅ What Remains in Shared Modules

### **Truly Shared Functionality:**
- **`config.py`** - Core configuration (entity types, paths, basic settings)
- **`data_processing.py`** - Data loading and BIO conversion
- **`dataset.py`** - Dataset creation and tokenization
- **`model_utils.py`** - Basic model loading and training utilities
- **`evaluation.py`** - Evaluation and analysis functions

## ❌ What Was Removed from Shared

### **Notebook-Specific Modules Removed:**
1. **`inference.py`** - Removed (not needed as requested)
2. **`bert_crf.py`** - Moved to BERT-CRF notebook only
3. **`mlm_pretraining.py`** - Moved to DAPT notebook only

### **Notebook-Specific Configs Removed:**
- `BERT_CRF_CONFIG`
- `MLM_CONFIG`
- `DEFAULT_CLASS_WEIGHTS`
- `PROBLEMATIC_ENTITIES`

## 🔧 How to Update Each Notebook

### **1. Base Notebook** ✅
**Status:** Already clean - no changes needed
- Uses only shared modules
- No notebook-specific code

### **2. BERT-CRF Notebook** 🔄
**What to add:** Copy code from `bert_crf_code_for_notebook.py`

```python
# Add after imports in BERT-CRF notebook:

# BERT-CRF Configuration
BERT_CRF_CONFIG = {
    "dropout_rate": 0.1,
    "bert_lr": 3e-5,
    "classifier_lr": 1e-4,
    "crf_lr": 1e-3
}

# BERT-CRF Classes
class BertCrfForTokenClassification(nn.Module):
    # ... (copy full class from bert_crf_code_for_notebook.py)

class BertCrfTrainer(Trainer):
    # ... (copy full class from bert_crf_code_for_notebook.py)

def compute_crf_metrics(eval_pred, id_to_label: Dict[int, str]):
    # ... (copy function from bert_crf_code_for_notebook.py)
```

### **3. Class Weights Notebook** 🔄
**What to add:** Copy code from `class_weights_code_for_notebook.py`

```python
# Add after imports in Class Weights notebook:

# Class Weights Configuration
DEFAULT_CLASS_WEIGHTS = {
    "CASE_NUMBER": 43.24,
    "JUDGE": 22.90,
    # ... (copy full dict from class_weights_code_for_notebook.py)
}

# Custom Weighted Trainer
class WeightedTrainer(Trainer):
    # ... (copy full class from class_weights_code_for_notebook.py)

def calculate_class_weights(dataset, label_to_id, method='balanced'):
    # ... (copy function from class_weights_code_for_notebook.py)
```

### **4. DAPT MLM Notebook** 🔄
**What to add:** Copy code from `mlm_pretraining_code_for_notebook.py`

```python
# Add after imports in DAPT MLM notebook:

# MLM Configuration
MLM_CONFIG = {
    "num_epochs": 3,
    "batch_size": 8,
    # ... (copy full dict from mlm_pretraining_code_for_notebook.py)
}

# MLM Functions
def load_mlm_documents(judgments_dir: str) -> List[str]:
    # ... (copy function from mlm_pretraining_code_for_notebook.py)

def preprocess_mlm_data(documents: List[str], tokenizer, max_length: int = 512, stride: int = 256):
    # ... (copy function from mlm_pretraining_code_for_notebook.py)

def perform_mlm_pretraining(model, tokenizer, train_dataset, output_dir: str, mlm_config: dict):
    # ... (copy function from mlm_pretraining_code_for_notebook.py)

def validate_mlm_model(model_path: str, tokenizer_path: str, sample_texts: List[str]):
    # ... (copy function from mlm_pretraining_code_for_notebook.py)
```

### **5. XLM-R BERTić Notebook** ✅
**Status:** Already clean - no changes needed
- Uses only shared modules
- No notebook-specific code

## 🔄 Updated Import Pattern

### **For All Notebooks:**
```python
# Import shared modules
import sys
sys.path.append('/content/drive/MyDrive/NER_Master/ner/shared')

from shared import (
    # Configuration
    ENTITY_TYPES, BIO_LABELS, DEFAULT_TRAINING_ARGS,
    get_default_model_config, get_paths, setup_environment, get_default_training_args,
    
    # Data processing
    LabelStudioToBIOConverter, load_labelstudio_data,
    analyze_labelstudio_data, validate_bio_examples,
    
    # Dataset
    NERDataset, split_dataset, tokenize_and_align_labels_with_sliding_window,
    print_sequence_analysis, create_huggingface_datasets,
    
    # Model utilities
    load_model_and_tokenizer, create_training_arguments, create_trainer,
    detailed_evaluation, save_model_info, setup_device_and_seed,
    
    # Evaluation
    generate_evaluation_report, plot_training_history, plot_entity_distribution
)

# Then add notebook-specific code as needed
```

## 📊 Benefits of This Approach

### **✅ Advantages:**
- **Cleaner shared modules** - Only truly shared code
- **Self-contained notebooks** - Each notebook has its specific code
- **Easier maintenance** - Notebook-specific changes stay in notebooks
- **Better organization** - Clear separation of concerns
- **No unnecessary dependencies** - Notebooks only import what they need

### **📈 Code Organization:**
- **Shared modules:** ~1,500 lines (down from 2,400)
- **Notebook-specific code:** ~900 lines moved back to notebooks
- **Total reduction:** Still ~80% less code duplication than original

## 🚀 Next Steps

1. **Copy the code** from the helper files into your notebooks
2. **Update imports** to use the new shared module functions
3. **Test each notebook** to ensure everything works
4. **Remove the helper files** once you've copied the code

The shared modules now contain only truly shared functionality, while notebook-specific code lives where it belongs - in the notebooks themselves!
