"""
Shared evaluation and analysis utilities for Serbian Legal NER project.

This module contains functions for confusion matrix analysis, plotting,
and detailed evaluation of NER models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report


def create_confusion_matrices(true_labels: List[List[str]], predictions: List[List[str]], 
                            dataset_name: str = "Test") -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Create and visualize confusion matrices for NER analysis"""

    # Flatten the sequences
    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predictions for label in seq]

    # Get all unique labels
    all_labels = sorted(list(set(flat_true + flat_pred)))

    # Create confusion matrix
    cm = confusion_matrix(flat_true, flat_pred, labels=all_labels)

    print(f"\n📊 {dataset_name} Confusion Matrix Analysis")
    print("=" * 50)

    # Show top confusions
    print("\n🔍 Most Confused Label Pairs:")
    confusion_pairs = []
    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((true_label, pred_label, cm[i][j]))

    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in confusion_pairs[:10]:
        print(f"  {true_label} → {pred_label}: {count}")

    return cm, all_labels, flat_true, flat_pred


def analyze_misclassifications(true_labels: List[List[str]], predictions: List[List[str]], 
                             focus_entities: Optional[List[str]] = None):
    """Detailed analysis of misclassifications for specific entities"""

    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predictions for label in seq]

    if focus_entities is None:
        # Extract entity types from labels
        entity_types = set()
        for label in flat_true + flat_pred:
            if label != 'O' and '-' in label:
                entity_type = label.split('-')[1]
                entity_types.add(entity_type)
        focus_entities = sorted(list(entity_types))

    print(f"\n🔍 DETAILED MISCLASSIFICATION ANALYSIS")
    print("=" * 50)

    for entity in focus_entities:
        print(f"\n🔍 ANALYZING: {entity}")
        print("=" * 50)

        # Find all variants of this entity (B- and I-)
        entity_variants = [f"B-{entity}", f"I-{entity}"]
        total_entity_instances = 0

        for variant in entity_variants:
            if variant in flat_true:
                variant_true_indices = [i for i, label in enumerate(flat_true) if label == variant]
                total_variant_instances = len(variant_true_indices)
                total_entity_instances += total_variant_instances

                if total_variant_instances > 0:
                    print(f"\n  📋 {variant}:")

                    # Count correct predictions
                    correct_predictions = sum(1 for i in variant_true_indices if flat_pred[i] == variant)
                    accuracy = correct_predictions / total_variant_instances

                    # Count misclassifications
                    misclassifications = Counter()
                    for i in variant_true_indices:
                        if flat_pred[i] != variant:
                            misclassifications[flat_pred[i]] += 1

                    print(f"    ✅ Correct predictions: {correct_predictions}/{total_variant_instances} ({accuracy:.2%})")

                    if misclassifications:
                        print(f"    ❌ Misclassified as:")
                        for wrong_label, count in misclassifications.most_common():
                            percentage = count / total_variant_instances
                            print(f"       → {wrong_label}: {count} times ({percentage:.2%})")
                    else:
                        print(f"    🎯 Perfect classification for {variant}!")

        print(f"📊 Total true instances: {total_entity_instances}")


def analyze_entity_confusion_patterns(true_labels: List[List[str]], predictions: List[List[str]]):
    """Analyze common confusion patterns between similar entities"""

    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predictions for label in seq]

    print("\n=== ENTITY CONFUSION PATTERNS ===")
    print("\nMost common misclassification pairs:")

    # Find all misclassifications
    misclassifications = Counter()

    for true_label, pred_label in zip(flat_true, flat_pred):
        if true_label != pred_label:
            misclassifications[(true_label, pred_label)] += 1

    # Show top misclassification patterns
    print("\n🔄 Top 20 Misclassification Patterns:")
    print("   (True Label → Predicted Label: Count)")
    print("-" * 60)

    for (true_label, pred_label), count in misclassifications.most_common(20):
        print(f"   {true_label:20} → {pred_label:20}: {count:4d}")

    # Analyze O vs Entity confusions
    print("\n🎯 O (Outside) vs Entity Confusions:")
    print("-" * 40)

    # Entities missed (predicted as O)
    missed_entities = Counter()
    false_entities = Counter()

    for (true_label, pred_label), count in misclassifications.items():
        if true_label != 'O' and pred_label == 'O':
            missed_entities[true_label] += count
        elif true_label == 'O' and pred_label != 'O':
            false_entities[pred_label] += count

    print("\n  📉 Entities missed (predicted as O):")
    for entity, count in missed_entities.most_common(10):
        print(f"     {entity:20}: {count:4d} times")

    print("\n  📈 False entity predictions (O predicted as entity):")
    for entity, count in false_entities.most_common(10):
        print(f"     {entity:20}: {count:4d} times")


def suggest_improvements_based_on_analysis(true_labels: List[List[str]], predictions: List[List[str]]) -> Tuple[List[str], List[str]]:
    """Provide specific recommendations based on confusion matrix analysis"""

    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predictions for label in seq]

    # Count entity frequencies
    entity_counts = Counter()
    for label in flat_true:
        if label != 'O':
            entity_type = label.split('-')[1] if '-' in label else label
            entity_counts[entity_type] += 1

    # Identify low-frequency entities (potential class imbalance issues)
    total_entities = sum(entity_counts.values())
    low_frequency_entities = []
    high_frequency_entities = []

    for entity, count in entity_counts.items():
        frequency = count / total_entities
        if frequency < 0.05:  # Less than 5% of total entities
            low_frequency_entities.append(entity)
        elif frequency > 0.15:  # More than 15% of total entities
            high_frequency_entities.append(entity)

    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 50)

    if low_frequency_entities:
        print(f"\n🎯 Low-frequency entities (potential class imbalance):")
        for entity in low_frequency_entities:
            count = entity_counts[entity]
            percentage = count / total_entities * 100
            print(f"   • {entity}: {count} instances ({percentage:.1f}%)")

        print(f"\n💡 Recommendations for low-frequency entities:")
        print(f"   • Implement class weights in loss function")
        print(f"   • Use focal loss for hard examples")
        print(f"   • Collect more training data for these entities")
        print(f"   • Consider data augmentation techniques")

    if high_frequency_entities:
        print(f"\n📊 High-frequency entities:")
        for entity in high_frequency_entities:
            count = entity_counts[entity]
            percentage = count / total_entities * 100
            print(f"   • {entity}: {count} instances ({percentage:.1f}%)")

    # Calculate class weights for implementation
    print(f"\n🔢 SUGGESTED CLASS WEIGHTS:")
    print(f"   (Use these in your loss function)")

    # Calculate inverse frequency weights
    max_count = max(entity_counts.values()) if entity_counts else 1

    print(f"\n   Entity Type          Count    Weight")
    print(f"   ----------------------------------------")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1]):
        weight = max_count / count
        print(f"   {entity:20} {count:8d}     {weight:5.2f}")

    print(f"\n🚀 GENERAL RECOMMENDATIONS:")
    print(f"   • Use the confusion matrices above to identify specific error patterns")
    print(f"   • Consider post-processing rules for common misclassifications")
    print(f"   • Experiment with different learning rates for different layers")
    print(f"   • Try ensemble methods combining multiple models")
    print(f"   • Implement CRF layer for better sequence modeling")
    print(f"   • Use domain-adaptive pretraining on legal texts")
    print(f"   • Experiment with different optimizers (AdamW, RMSprop)")

    return low_frequency_entities, high_frequency_entities


def plot_training_history(trainer):
    """Plot training metrics"""
    if hasattr(trainer.state, 'log_history'):
        logs = trainer.state.log_history

        # Extract metrics
        train_loss = []
        eval_loss = []
        eval_f1 = []
        steps = []

        for log in logs:
            if 'loss' in log:
                train_loss.append(log['loss'])
                steps.append(log['step'])
            if 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
            if 'eval_f1' in log:
                eval_f1.append(log['eval_f1'])

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        if train_loss:
            ax1.plot(steps, train_loss, label='Training Loss', color='blue')
        if eval_loss:
            eval_steps = [log['step'] for log in logs if 'eval_loss' in log]
            ax1.plot(eval_steps, eval_loss, label='Validation Loss', color='red')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot F1 score
        if eval_f1:
            eval_steps = [log['step'] for log in logs if 'eval_f1' in log]
            ax2.plot(eval_steps, eval_f1, label='Validation F1', color='green')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Validation F1 Score')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

        print("📈 Training history plotted successfully!")
    else:
        print("No training history available")


def plot_entity_distribution(entity_counts: Dict[str, int]):
    """Plot distribution of entity types"""
    if not entity_counts:
        print("No entity counts available")
        return

    # Sort entities by count
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    entities, counts = zip(*sorted_entities)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(entities, counts, color='skyblue', edgecolor='navy', alpha=0.7)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Entity Types')
    plt.ylabel('Count')
    plt.title('Distribution of Entity Types in Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("📊 Entity distribution plotted successfully!")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    print("🔥 Confusion matrix plotted successfully!")


def generate_evaluation_report(true_labels: List[List[str]], predictions: List[List[str]], 
                             dataset_name: str = "Test", focus_entities: Optional[List[str]] = None) -> Dict:
    """Generate comprehensive evaluation report"""
    
    print(f"\n🔍 COMPREHENSIVE EVALUATION REPORT - {dataset_name.upper()}")
    print("=" * 60)
    
    # 1. Create confusion matrices
    cm, all_labels, flat_true, flat_pred = create_confusion_matrices(true_labels, predictions, dataset_name)
    
    # 2. Analyze misclassifications
    analyze_misclassifications(true_labels, predictions, focus_entities)
    
    # 3. Analyze confusion patterns
    analyze_entity_confusion_patterns(true_labels, predictions)
    
    # 4. Generate improvement suggestions
    low_freq, high_freq = suggest_improvements_based_on_analysis(true_labels, predictions)
    
    # 5. Calculate overall metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, support = precision_recall_fscore_support(flat_true, flat_pred, average='weighted')
    accuracy = accuracy_score(flat_true, flat_pred)
    
    report = {
        'dataset_name': dataset_name,
        'confusion_matrix': cm,
        'labels': all_labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'low_frequency_entities': low_freq,
        'high_frequency_entities': high_freq,
        'total_predictions': len(flat_pred),
        'total_sequences': len(true_labels)
    }
    
    print(f"\n✅ Evaluation report generated for {dataset_name} dataset")
    
    return report
