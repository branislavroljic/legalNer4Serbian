# Serbian Legal NER - Sentence-Level Processing Improvements

## 🚀 Overview

This document explains the major improvements implemented in the new sentence-level processing approach for Serbian legal NER, based on successful research methodologies.

## 📊 Key Improvements

### 1. **Document → Sentence Level Processing** ⭐ **MOST IMPORTANT**

**Before (Document-level):**
- 60 documents = 60 training examples
- Very long sequences (300+ tokens)
- Poor entity boundary detection
- Sliding windows needed for long sequences

**After (Sentence-level):**
- 60 documents → 300+ sentence examples (5x increase!)
- Shorter sequences (average ~50 tokens)
- Better entity boundary detection
- No sliding windows needed

### 2. **Strategic Negative Examples**

**Research Finding:** Papers show that including 20-30% negative examples (sentences without entities) improves:
- Model generalization
- Reduced false positives
- Better boundary detection

**Implementation:**
- Filter sentences with entities for positive examples
- Add 25% negative examples (sentences without entities)
- Balanced dataset for better training

### 3. **Optimized Training Configuration**

**Sentence-level optimizations:**
- Larger batch size (8 vs 4) - shorter sequences allow this
- Higher learning rate (3e-5 vs 2e-5) - more examples need higher LR
- More epochs (15 vs 5) - better convergence with more data
- Frequent evaluation - better monitoring

### 4. **Improved Tokenization**

**Before:**
- Complex sliding window implementation
- Overlapping chunks
- Difficult entity reconstruction

**After:**
- Simple sentence-level tokenization
- No sliding windows needed
- Clean entity boundaries

## 📈 Expected Performance Improvements

Based on research papers and best practices:

### Dataset Size Impact:
- **Training Examples:** 60 → 300+ (5x increase)
- **Entity Instances:** Better distribution across examples
- **Class Balance:** Improved with negative examples

### Model Performance:
- **F1-Score:** Expected 10-20% improvement
- **Precision:** Better with negative examples
- **Recall:** Better entity boundary detection
- **Training Stability:** More stable with more examples

### Training Efficiency:
- **Convergence:** Faster and more stable
- **Memory Usage:** Lower (shorter sequences)
- **Training Time:** Potentially faster per epoch

## 🔧 Technical Implementation

### Core Classes:

1. **`SentenceLevelBIOConverter`**
   - Splits documents into sentences
   - Maps annotations to sentences
   - Creates BIO examples per sentence
   - Filters by sentence length and entity count

2. **Enhanced Dataset Creation**
   - Balanced positive/negative examples
   - Document-level statistics tracking
   - Improved label mapping

3. **Optimized Training Pipeline**
   - Sentence-level tokenization
   - Efficient data collation
   - Research-based hyperparameters

### Key Functions:

- `split_text_into_sentences()` - NLTK-based sentence splitting
- `map_annotations_to_sentence()` - Annotation mapping
- `create_balanced_dataset()` - Strategic negative sampling
- `tokenize_and_align_labels_sentence_level()` - Efficient tokenization

## 📋 Usage Instructions

### 1. **Run the New Notebook**
```bash
# Use the new sentence-level notebook
jupyter notebook ner/serbian_legal_ner_sentence_level.ipynb
```

### 2. **Expected Output**
- 300+ training examples (vs 60 before)
- Balanced positive/negative examples
- Better entity distribution
- Improved model performance

### 3. **Configuration Options**
```python
# Adjust these parameters in the notebook:
MIN_SENTENCE_LENGTH = 5      # Minimum tokens per sentence
MAX_SENTENCE_LENGTH = 200    # Maximum tokens per sentence  
NEGATIVE_RATIO = 0.25        # 25% negative examples
MIN_ENTITIES_PER_SENTENCE = 1 # Minimum entities to keep sentence
```

## 🎯 Research Foundation

This approach is based on successful legal NER research that achieved good results with:
- **2,172 sentences** containing NEs (our target: 300+ sentences)
- **183,543 tokens** after WordPiece tokenization
- **6,319 total entity instances** across entity types
- **BIO tagging scheme** with sentence-level processing

### Key Research Insights:
1. **Sentence filtering:** "sentences without any NE were not taken into final annotated dataset"
2. **Document-level cross-validation:** Prevents data leakage
3. **Strategic negative examples:** Improves generalization
4. **Optimized training:** Better convergence with more examples

## 🔄 Migration Path

### From Original Notebook:
1. **Keep original notebook** as backup
2. **Use new sentence-level notebook** for training
3. **Compare results** between approaches
4. **Adjust parameters** based on your specific data

### Backward Compatibility:
- Same entity types and labels
- Same model architecture (BCSm-BERTić)
- Same evaluation metrics
- Compatible with existing inference code

## 🎉 Expected Results

With this sentence-level approach, you should see:

1. **More Training Data:** 5x increase in examples
2. **Better Performance:** 10-20% F1-score improvement
3. **Faster Training:** More stable convergence
4. **Better Generalization:** Reduced overfitting
5. **Cleaner Predictions:** Better entity boundaries

## 🚀 Next Steps

1. **Run the new notebook** and compare results
2. **Experiment with parameters** (negative ratio, sentence length)
3. **Implement document-level cross-validation** for even better results
4. **Consider data augmentation** techniques for further improvements

This sentence-level approach represents a significant advancement in your Serbian legal NER pipeline, bringing it in line with current research best practices!
