# GLiNER Zero-Shot Evaluation for Serbian Legal Documents

This directory contains a comprehensive evaluation framework for testing GLiNER (Generalist and Lightweight Named Entity Recognition) models on Serbian legal documents in zero-shot mode.

## 🎯 Overview

GLiNER is a state-of-the-art NER model that can identify any entity type without training on domain-specific data. This evaluation compares GLiNER's zero-shot performance against your fine-tuned BCSm-BERTić model on 225 Serbian legal documents.

## 📁 Files

- **`gliner_zero_shot_evaluation.ipynb`** - Complete Jupyter notebook with detailed analysis
- **`run_gliner_evaluation.py`** - Command-line script for quick evaluation
- **`GLiNER_EVALUATION_README.md`** - This documentation file

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
# Install dependencies
pip install gliner seqeval scikit-learn matplotlib seaborn pandas tqdm

# Open the notebook
jupyter notebook gliner_zero_shot_evaluation.ipynb
```

### Option 2: Command Line Script

```bash
# Basic evaluation with default settings
python run_gliner_evaluation.py

# Use different GLiNER model
python run_gliner_evaluation.py --model large --threshold 0.4

# Evaluate subset of documents
python run_gliner_evaluation.py --model multitask --max-examples 50
```

## 🎛️ Available GLiNER Models

| Model Key | Full Model Name | Description |
|-----------|----------------|-------------|
| `medium` | `urchade/gliner_mediumv2.1` | **Default** - Good balance of speed/accuracy |
| `large` | `urchade/gliner_large` | Larger model, better accuracy |
| `multitask` | `knowledgator/gliner-multitask-large-v0.5` | Specialized multitask model |
| `small` | `urchade/gliner_small-v2.1` | Faster, smaller model |
| `base` | `urchade/gliner_base` | Base model |

## 🏷️ Entity Types Evaluated

The evaluation covers 13 Serbian legal entity types:

### Core Legal Entities
- **COURT** - Court institutions
- **JUDGE** - Judge names  
- **DEFENDANT** - Defendant entities
- **PROSECUTOR** - Prosecutor entities
- **REGISTRAR** - Court registrar

### Case Information
- **CASE_NUMBER** - Case identifiers
- **CRIMINAL_ACT** - Criminal acts/charges
- **PROVISION** - Legal provisions
- **DECISION_DATE** - Dates of legal decisions

### Sanctions & Costs
- **SANCTION** - Sanctions/penalties
- **SANCTION_TYPE** - Type of sanction
- **SANCTION_VALUE** - Value/duration of sanction
- **PROCEDURE_COSTS** - Legal procedure costs

## 📊 Evaluation Metrics

### Entity-Level Metrics
- **Exact Match**: Perfect position and label alignment
- **Overlap Match**: ≥50% span overlap with correct label
- **Precision/Recall/F1**: Standard NER evaluation metrics

### Analysis Features
- Per-entity type performance breakdown
- Confidence score distribution analysis
- Coverage ratio assessment (predicted/true entities)
- Processing speed benchmarks

## 🔧 Configuration Options

### Command Line Arguments

```bash
python run_gliner_evaluation.py \
    --model medium \              # GLiNER model to use
    --threshold 0.3 \             # Confidence threshold
    --max-examples 225 \          # Number of documents to evaluate
    --output results.json         # Output file name
```

### Confidence Threshold Guidelines

| Threshold | Effect |
|-----------|--------|
| `0.1-0.2` | High recall, lower precision |
| `0.3-0.4` | **Balanced** (recommended) |
| `0.5-0.7` | Higher precision, lower recall |
| `0.8+` | Very conservative predictions |

## 📈 Expected Results

Based on GLiNER's capabilities, you can expect:

### Strengths
- **Person Names**: Judges, defendants, prosecutors (high accuracy)
- **Organizations**: Courts, prosecutor offices (good performance)
- **Dates**: Decision dates (moderate accuracy)

### Challenges
- **Case Numbers**: Specific legal formatting (lower accuracy)
- **Legal Provisions**: Domain-specific references (challenging)
- **Sanctions**: Complex legal terminology (variable performance)

## 🔍 Comparison with Fine-Tuned Models

### GLiNER Advantages
- ✅ No training required
- ✅ Immediate deployment
- ✅ Cross-lingual capabilities
- ✅ Flexible entity types

### Fine-Tuned Model Advantages
- ✅ Domain-specific optimization
- ✅ Better handling of legal terminology
- ✅ Consistent performance on all entity types
- ✅ Understanding of Serbian legal context

## 📋 Output Files

### Results JSON Structure
```json
{
  "model_info": {
    "model_name": "urchade/gliner_mediumv2.1",
    "confidence_threshold": 0.3,
    "method": "zero_shot"
  },
  "performance_metrics": {
    "exact_match": {"precision": 0.45, "recall": 0.38, "f1": 0.41},
    "overlap_match": {"precision": 0.62, "recall": 0.55, "f1": 0.58},
    "coverage_ratio": 0.73
  },
  "entity_distribution": {
    "ground_truth": {"COURT": 245, "JUDGE": 189, ...},
    "predictions": {"court": 178, "judge": 156, ...}
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **GLiNER Installation**
   ```bash
   pip install gliner
   # If GPU available:
   pip install gliner[gpu]
   ```

2. **Memory Issues**
   - Use smaller model: `--model small`
   - Reduce batch size in notebook
   - Evaluate fewer documents: `--max-examples 50`

3. **File Not Found**
   - Ensure `annotations.json` is in current directory
   - Check `labelstudio_files/` directory exists
   - Verify file paths in LabelStudio annotations

### Performance Optimization

```bash
# For faster evaluation
python run_gliner_evaluation.py --model small --threshold 0.4

# For better accuracy
python run_gliner_evaluation.py --model large --threshold 0.2

# For balanced approach
python run_gliner_evaluation.py --model medium --threshold 0.3
```

## 📚 Next Steps

1. **Run Evaluation**: Start with the Jupyter notebook for detailed analysis
2. **Compare Results**: Use saved JSON to compare with fine-tuned model
3. **Optimize Threshold**: Experiment with different confidence values
4. **Try Different Models**: Test various GLiNER variants
5. **Ensemble Approach**: Consider combining GLiNER + fine-tuned model

## 🔗 References

- [GLiNER Paper](https://arxiv.org/abs/2311.08526)
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [Hugging Face Models](https://huggingface.co/models?search=gliner)

## 💡 Tips for Best Results

1. **Entity Type Naming**: Use clear, descriptive names for entity types
2. **Confidence Tuning**: Start with 0.3, adjust based on precision/recall needs
3. **Text Preprocessing**: Clean text can improve GLiNER performance
4. **Multiple Models**: Try different GLiNER variants for comparison
5. **Error Analysis**: Use the notebook's sample analysis to understand failures

Happy evaluating! 🚀
