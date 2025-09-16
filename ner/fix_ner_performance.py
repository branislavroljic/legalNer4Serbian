"""
Script to Fix Serbian Legal NER Performance Issues

This script addresses the major issues causing poor NER performance:
1. Class imbalance
2. Suboptimal training configuration
3. Insufficient training data handling
4. Poor evaluation metrics

Run this after your data preprocessing to get better results.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report as seqeval_report
from collections import Counter

# Import our improved configuration
from improved_training_config import (
    create_improved_training_args,
    compute_class_weights,
    create_weighted_trainer_class,
    create_improved_callbacks,
    print_training_recommendations
)


def analyze_data_quality(train_examples, val_examples, test_examples):
    """
    Analyze the quality and distribution of the training data.
    """
    print("\n" + "="*60)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Count examples
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples") 
    print(f"  Test: {len(test_examples)} examples")
    
    # Analyze label distribution in training data
    all_labels = []
    total_tokens = 0
    entity_tokens = 0
    
    for example in train_examples:
        labels = example['labels']
        total_tokens += len(labels)
        
        for label in labels:
            if isinstance(label, int):
                # Convert back to string if needed
                label_str = label  # Will handle this in calling code
            else:
                label_str = label
            all_labels.append(label_str)
            
            if label_str != 'O':
                entity_tokens += 1
    
    # Calculate class imbalance
    label_counts = Counter(all_labels)
    o_count = label_counts.get('O', 0)
    entity_count = total_tokens - o_count
    
    print(f"\nClass imbalance analysis:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  'O' (Outside) tokens: {o_count:,} ({o_count/total_tokens*100:.1f}%)")
    print(f"  Entity tokens: {entity_count:,} ({entity_count/total_tokens*100:.1f}%)")
    print(f"  Imbalance ratio: {o_count/max(entity_count, 1):.1f}:1")
    
    if o_count / total_tokens > 0.85:
        print("  ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("     This is likely the main cause of poor performance.")
    
    # Analyze entity distribution
    entity_labels = [label for label in all_labels if label != 'O']
    if entity_labels:
        entity_counts = Counter(entity_labels)
        print(f"\nEntity type distribution:")
        for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(entity_labels)) * 100
            print(f"  {entity}: {count} ({percentage:.1f}%)")
    
    return {
        'total_tokens': total_tokens,
        'entity_tokens': entity_tokens,
        'o_tokens': o_count,
        'imbalance_ratio': o_count / max(entity_count, 1),
        'label_counts': label_counts
    }


def create_improved_compute_metrics(label_to_id):
    """
    Create an improved compute_metrics function with better evaluation.
    """
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Compute seqeval metrics (entity-level)
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
        
        # Add per-entity metrics
        try:
            # Get unique entity types (excluding O)
            entity_types = set()
            for seq in true_labels:
                for label in seq:
                    if label != 'O' and label.startswith(('B-', 'I-')):
                        entity_type = label[2:]  # Remove B- or I- prefix
                        entity_types.add(entity_type)
            
            # Compute per-entity F1 scores
            for entity_type in entity_types:
                entity_true = []
                entity_pred = []
                
                for true_seq, pred_seq in zip(true_labels, true_predictions):
                    entity_true_seq = []
                    entity_pred_seq = []
                    
                    for true_label, pred_label in zip(true_seq, pred_seq):
                        # Convert to binary: is this token part of the target entity?
                        true_is_entity = true_label.endswith(entity_type) if true_label != 'O' else False
                        pred_is_entity = pred_label.endswith(entity_type) if pred_label != 'O' else False
                        
                        entity_true_seq.append('B-' + entity_type if true_is_entity else 'O')
                        entity_pred_seq.append('B-' + entity_type if pred_is_entity else 'O')
                    
                    entity_true.append(entity_true_seq)
                    entity_pred.append(entity_pred_seq)
                
                if entity_true and entity_pred:
                    entity_f1 = f1_score(entity_true, entity_pred)
                    results[f"f1_{entity_type}"] = entity_f1
        
        except Exception as e:
            print(f"Warning: Could not compute per-entity metrics: {e}")
        
        return results
    
    return compute_metrics


def train_improved_model(
    train_examples, val_examples, test_examples,
    model_name, output_dir, label_to_id
):
    """
    Train the model with improved configuration.
    """
    print("\n" + "="*60)
    print("🚀 TRAINING IMPROVED MODEL")
    print("="*60)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label={v: k for k, v in label_to_id.items()},
        label2id=label_to_id
    )
    
    # Compute class weights
    train_labels = [example['labels'] for example in train_examples]
    class_weights = compute_class_weights(train_labels, label_to_id)
    
    # Create improved training arguments
    training_args = create_improved_training_args(output_dir, len(train_examples))
    
    # Create datasets (assuming tokenization is already done)
    train_dataset = HFDataset.from_list(train_examples)
    val_dataset = HFDataset.from_list(val_examples)
    test_dataset = HFDataset.from_list(test_examples)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Improved compute metrics
    compute_metrics = create_improved_compute_metrics(label_to_id)
    
    # Create weighted trainer
    WeightedTrainer = create_weighted_trainer_class(class_weights)
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=create_improved_callbacks()
    )
    
    print(f"\nTraining configuration:")
    print(f"  Model: {model_name}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")
    print(f"  Test examples: {len(test_examples)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Class weighting: {'Enabled' if class_weights else 'Disabled'}")
    
    # Train the model
    print("\n🏃 Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Model saved to {output_dir}")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\nTest Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"  {metric_name}: {value:.4f}")
    
    # Detailed evaluation
    print("\n📋 Generating detailed classification report...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=2)
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_pred, predictions.label_ids)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_pred, predictions.label_ids)
    ]
    
    # Print seqeval classification report
    print("\nEntity-level Classification Report (seqeval):")
    print(seqeval_report(true_labels, true_predictions))
    
    # Save results
    results = {
        'test_results': test_results,
        'model_name': model_name,
        'training_examples': len(train_examples),
        'class_weights': class_weights,
        'training_args': training_args.to_dict()
    }
    
    with open(f"{output_dir}/improved_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return trainer, test_results


def main():
    """
    Main function to demonstrate the improvements.
    """
    print_training_recommendations()
    
    print("\n" + "="*80)
    print("🔧 TO USE THESE IMPROVEMENTS:")
    print("="*80)
    print("\n1. Import this module in your notebook:")
    print("   from fix_ner_performance import train_improved_model, analyze_data_quality")
    print("\n2. Analyze your data quality:")
    print("   analyze_data_quality(train_examples, val_examples, test_examples)")
    print("\n3. Train with improved configuration:")
    print("   trainer, results = train_improved_model(")
    print("       train_examples, val_examples, test_examples,")
    print("       MODEL_NAME, OUTPUT_DIR, ner_dataset.label_to_id")
    print("   )")
    print("\n4. The improved model should show significantly better results!")


if __name__ == "__main__":
    main()
