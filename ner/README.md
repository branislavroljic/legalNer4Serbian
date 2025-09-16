# Serbian Legal Named Entity Recognition (NER) Pipeline

This project implements a complete pipeline for fine-tuning the BCSm-BERTić model for Named Entity Recognition on Serbian legal documents using the BIO (Beginning-Inside-Outside) tagging scheme.

## Entity Types

The model recognizes the following entity types in Serbian legal documents:

- **COURT**: Court institutions
- **DECISION_DATE**: Dates of legal decisions
- **CASE_NUMBER**: Case identifiers
- **CRIMINAL_ACT**: Criminal acts/charges
- **PROSECUTOR**: Prosecutor entities
- **DEFENDANT**: Defendant entities
- **JUDGE**: Judge names
- **REGISTRAR**: Court registrar
- **SANCTION**: Sanctions/penalties
- **SANCTION_TYPE**: Type of sanction
- **SANCTION_VALUE**: Value/duration of sanction
- **PROVISION**: Legal provisions
- **PROCEDURE_COSTS**: Legal procedure costs

## Files

- `serbian_legal_ner_pipeline.ipynb` - Complete training and evaluation notebook
- `test_ner_pipeline.py` - Simple test script for inference
- `export_186137_project-186137-at-2025-09-08-17-25-9356b6a3.json` - LabelStudio annotations
- `models/serbian-legal-ner/` - Trained model directory (created after training)

## Requirements

Install the required packages:

```bash
pip install transformers torch datasets tokenizers scikit-learn seqeval pandas numpy matplotlib seaborn tqdm
```

## Usage

### 1. Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook serbian_legal_ner_pipeline.ipynb
```

The notebook includes:
- Data loading and analysis
- LabelStudio annotation conversion to BIO format
- Dataset preparation and splitting
- Model fine-tuning with BCSm-BERTić
- Evaluation and metrics
- Inference pipeline creation

### 2. Testing the Trained Model

After training, you can test the model using the test script:

```bash
# Test with default text
python test_ner_pipeline.py

# Test with a specific file
python test_ner_pipeline.py ../judgments/judgment_K_4_2015.txt
```

### 3. Using the Pipeline in Your Code

```python
from test_ner_pipeline import SerbianLegalNERPipeline

# Load the trained model
pipeline = SerbianLegalNERPipeline("./models/serbian-legal-ner")

# Predict entities in text
text = "OSNOVNI SUD U HERCEG NOVOM, po sudiji Leković Branislavu..."
entities = pipeline.predict(text)

# Display results
for entity in entities:
    print(f"{entity['label']}: '{entity['text']}'")
```

## Model Architecture

- **Base Model**: BCSm-BERTić (`classla/bcms-bertic`)
- **Task**: Token Classification (NER)
- **Tagging Scheme**: BIO (Beginning-Inside-Outside)
- **Fine-tuning**: Supervised learning on annotated Serbian legal documents

## Training Configuration

- **Epochs**: 5
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Warmup Steps**: 500
- **Weight Decay**: 0.01
- **Early Stopping**: Patience of 3 epochs
- **Evaluation Metric**: F1 Score

## Data Format

The pipeline converts LabelStudio annotations to BIO format:
- `B-ENTITY`: Beginning of an entity
- `I-ENTITY`: Inside/continuation of an entity
- `O`: Outside any entity

Example:
```
OSNOVNI    B-COURT
SUD        I-COURT
U          I-COURT
HERCEG     I-COURT
NOVOM      I-COURT
,          O
po         O
sudiji     O
Leković    B-JUDGE
Branislavu I-JUDGE
```

## Evaluation Metrics

The model is evaluated using:
- **Precision**: Percentage of predicted entities that are correct
- **Recall**: Percentage of actual entities that are found
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Token-level accuracy

## Key Improvements Made

### LabelStudio File Handling Fix

**Problem solved:** The original implementation had annotation offset issues because it used local judgment files that differed from the files LabelStudio actually used for annotation.

**Solution implemented:**
1. **Direct file_upload usage**: Uses the `file_upload` property directly instead of extracting from `data.text`
2. **LabelStudio files priority**: Looks for files in `labelstudio_files/` directory first
3. **Automatic download**: Provides scripts to download the exact files LabelStudio used
4. **Fallback mechanism**: Falls back to local judgment files if LabelStudio files aren't available

**Technical details:**
- `file_upload` contains exact filenames like `"5534cab7-judgment_K_959_2012.txt"`
- Download URLs are constructed as `upload/186137/{filename}`
- No complex path extraction needed - direct filename usage
- Perfect annotation alignment when using actual LabelStudio files

**Benefits:**
- ✅ Eliminates annotation offset issues completely
- ✅ Uses exact same text that LabelStudio used for annotation
- ✅ Simpler and more reliable code
- ✅ Better error handling and debugging

### Fast Tokenizer Implementation

**Improvement:** Updated tokenizer initialization to use `use_fast=True` for optimal performance.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
```

**Benefits:**
- **Performance**: Rust-based implementation is significantly faster
- **Memory efficiency**: Better handling of long legal documents
- **Parallel processing**: Can tokenize multiple texts simultaneously
- **Robustness**: More stable handling of edge cases in Serbian text

## Detailed Jupyter Notebook Steps

### Steps 1-3: Data Loading and BIO Conversion (Prerequisites)

**Step 1: Environment Setup**
- Installs required packages (transformers, torch, datasets, etc.)
- Sets up random seeds for reproducibility
- Configures CUDA if available

**Step 2: Data Loading and Analysis**
- Loads LabelStudio annotations from JSON export
- Analyzes annotation structure and entity distribution
- Validates data quality and completeness

**Step 3: Data Preprocessing and BIO Conversion**
- Downloads LabelStudio files (recommended approach)
- Converts LabelStudio annotations to BIO format using `LabelStudioToBIOConverter`
- Handles file loading with proper encoding (UTF-8 for LabelStudio files, UTF-8-SIG for fallback)
- Creates character-to-token mapping for accurate label alignment
- Applies BIO tagging scheme to tokenized text

**Key BIO conversion process:**
1. **File loading**: Uses `file_upload` property to get exact filename
2. **Tokenization**: Simple whitespace tokenization for initial processing
3. **Character mapping**: Maps character positions to token indices
4. **Label alignment**: Assigns BIO tags to appropriate tokens
5. **Entity processing**: Handles overlapping and nested annotations

### Step 4: Label Encoding and Dataset Preparation

**What it does:**
- Creates a `NERDataset` class to handle BIO label encoding
- Maps string labels (like "B-COURT", "I-JUDGE", "O") to numerical IDs for model training
- Prepares examples for training by converting labels to integers

**Key components:**
- `_create_label_mapping()`: Creates bidirectional mapping between labels and IDs
- `encode_labels()` / `decode_labels()`: Convert between string labels and numerical IDs
- `prepare_for_training()`: Converts all examples to numerical format

**Why it's important:** Neural networks work with numbers, not strings. This step converts our BIO tags into a format the model can understand while maintaining the ability to convert back to human-readable labels.

### Step 5: Data Splitting and Tokenization

**What it does:**
- Splits the dataset into training (70%), validation (15%), and test (15%) sets
- Loads the BCSm-BERTić tokenizer with `use_fast=True` for optimal performance
- Ensures reproducible splits using `random_state=42`

**Key features:**
- **Fast tokenizer**: Uses Rust-based implementation for better performance on long legal texts
- **Stratified splitting**: Maintains entity distribution across splits
- **Vocabulary analysis**: Reports tokenizer vocabulary size and capabilities

**Why it's important:** Proper data splitting prevents overfitting and gives reliable performance estimates. The fast tokenizer significantly speeds up processing of lengthy legal documents.

### Step 6: Advanced Tokenization and Label Alignment

**What it does:**
- Implements `tokenize_and_align_labels()` function to handle subword tokenization
- Aligns BIO labels with BERT's WordPiece tokens
- Handles the challenge that one word might become multiple subword tokens

**Key challenges solved:**
- **Subword alignment**: When "Branislavu" becomes ["Bran", "##islavu"], only the first token gets the label
- **Special tokens**: Adds [CLS] and [SEP] tokens with -100 labels (ignored in loss calculation)
- **Padding**: Ensures all sequences have the same length for batch processing
- **Truncation**: Handles sequences longer than 512 tokens

**Technical details:**
- First subword token gets the original label (e.g., "B-JUDGE")
- Subsequent subword tokens get -100 (ignored during training)
- Padding tokens also get -100 labels

### Step 7: Model Setup and Training Configuration

**What it does:**
- Loads the pre-trained BCSm-BERTić model for token classification
- Configures the model with the correct number of output labels
- Sets up label mappings for inference
- Moves model to GPU if available

**Model configuration:**
- **Base model**: `classla/bcms-bertic` (Serbian BERT variant)
- **Task head**: Token classification layer with num_labels outputs
- **Label mappings**: Bidirectional mapping between IDs and label names
- **Device placement**: Automatic GPU detection and usage

**Why BCSm-BERTić:** This model is specifically trained on Bosnian, Croatian, Montenegrin, and Serbian texts, making it ideal for Serbian legal documents.

### Step 8: Dataset Conversion and Data Collation

**What it does:**
- Converts tokenized data to HuggingFace Dataset format
- Sets up `DataCollatorForTokenClassification` for efficient batching
- Handles dynamic padding and tensor conversion

**Key features:**
- **Dynamic padding**: Only pads to the longest sequence in each batch (not max_length)
- **Tensor conversion**: Automatically converts to PyTorch tensors
- **Label handling**: Properly handles -100 labels for ignored tokens

**Performance benefits:** Dynamic padding reduces computational overhead by avoiding unnecessary padding tokens.

### Step 9: Evaluation Metrics Setup

**What it does:**
- Implements `compute_metrics()` function using seqeval library
- Calculates entity-level metrics (not just token-level)
- Handles the conversion from model predictions back to label strings

**Metrics calculated:**
- **Precision**: Percentage of predicted entities that are correct
- **Recall**: Percentage of actual entities that were found
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Entity-level accuracy (stricter than token-level)

**Why seqeval:** Unlike standard classification metrics, seqeval evaluates complete entities. For example, if the model predicts "B-JUDGE I-JUDGE" but the true label is "B-JUDGE I-JUDGE O", it counts as a complete entity match.

### Step 10: Training Configuration and Fine-tuning

**What it does:**
- Sets up comprehensive training arguments for optimal performance
- Configures learning rate scheduling, evaluation strategy, and model saving
- Implements early stopping to prevent overfitting

**Key training parameters:**
- **Learning rate**: 2e-5 (typical for BERT fine-tuning)
- **Batch size**: 8 (balanced for memory and convergence)
- **Warmup steps**: 500 (gradual learning rate increase)
- **Weight decay**: 0.01 (L2 regularization)
- **Gradient accumulation**: 2 steps (effective batch size of 16)
- **Mixed precision**: FP16 if CUDA available (faster training)

**Evaluation strategy:**
- Evaluates every 200 steps
- Saves best model based on F1 score
- Early stopping with patience of 3 evaluations

### Step 11: Model Training Execution

**What it does:**
- Initializes the Trainer with all components
- Executes the training loop with automatic evaluation
- Saves the final model and tokenizer

**Training process:**
1. **Forward pass**: Computes predictions and loss
2. **Backward pass**: Calculates gradients
3. **Optimization**: Updates model weights
4. **Evaluation**: Periodically evaluates on validation set
5. **Checkpointing**: Saves best performing model

**Monitoring:** The trainer automatically logs training loss, validation metrics, and learning rate changes.

### Step 12: Model Evaluation and Analysis

**What it does:**
- Evaluates the trained model on the held-out test set
- Generates detailed classification reports for each entity type
- Provides per-entity precision, recall, and F1 scores

**Evaluation outputs:**
- **Overall metrics**: Test set performance summary
- **Per-entity analysis**: Individual performance for each entity type
- **Classification report**: Detailed breakdown with support counts
- **Confusion analysis**: Shows which entities are commonly confused

**Why separate test evaluation:** The test set was never seen during training or validation, providing an unbiased estimate of real-world performance.

### Step 13: Inference Pipeline Creation

**What it does:**
- Creates a `SerbianLegalNERPipeline` class for easy model deployment
- Implements end-to-end prediction from raw text to extracted entities
- Handles tokenization, prediction, and entity reconstruction

**Pipeline features:**
- **Text preprocessing**: Automatic tokenization and encoding
- **Batch prediction**: Efficient processing of input text
- **Entity reconstruction**: Converts BIO predictions back to entity spans
- **Post-processing**: Combines B- and I- tags into complete entities

**Usage example:**
```python
pipeline = SerbianLegalNERPipeline("./models/serbian-legal-ner")
entities = pipeline.predict("OSNOVNI SUD U NIKŠIĆU, sudija Babović Dragan...")
```

### Step 14: Testing and Validation

**What it does:**
- Tests the pipeline with sample Serbian legal text
- Validates entity extraction on real judgment files
- Demonstrates practical usage of the trained model

**Test scenarios:**
- **Sample text**: Predefined legal text with known entities
- **Real files**: Actual judgment documents from the dataset
- **Entity grouping**: Organizes predictions by entity type for analysis

**Output format:** Each entity includes text span, label, and position information for easy integration into downstream applications.

### Step 15: Results Visualization and Analysis

**What it does:**
- Plots training history showing loss and F1 score progression
- Visualizes entity distribution in the training data
- Creates publication-ready charts for analysis

**Visualizations:**
- **Training curves**: Loss and F1 score over training steps
- **Entity distribution**: Bar chart showing frequency of each entity type
- **Performance metrics**: Visual representation of model performance

**Analysis insights:** These visualizations help identify potential issues like overfitting, class imbalance, or convergence problems.

### Step 16: Model Export and Deployment Preparation

**What it does:**
- Saves complete model configuration and metadata
- Creates deployment-ready model artifacts
- Documents model performance and training details

**Saved artifacts:**
- **Model weights**: Fine-tuned BCSm-BERTić parameters
- **Tokenizer**: Vocabulary and tokenization rules
- **Configuration**: Label mappings and model settings
- **Metadata**: Training statistics and performance metrics

**Deployment ready:** The saved model can be loaded directly for inference without requiring the training code.

## Results

After training, the notebook will display:
- Per-entity type performance metrics
- Confusion matrix
- Training history plots
- Sample predictions

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training arguments
2. **Model not found**: Ensure you've run the training notebook first
3. **Encoding errors**: Ensure all text files are UTF-8 encoded
4. **Missing dependencies**: Install all required packages

### Performance Tips

- Use GPU for faster training (CUDA recommended)
- Adjust batch size based on available memory
- Use mixed precision training (fp16) for faster training
- Consider data augmentation for better performance

## License

This project is for research and educational purposes. Please ensure compliance with the licensing terms of the BCSm-BERTić model and any datasets used.

## Citation

If you use this pipeline in your research, please cite the relevant papers and models:
- BCSm-BERTić model
- Original research methodology
- LabelStudio for annotation

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.
