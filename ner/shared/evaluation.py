"""
Shared evaluation and analysis utilities for Serbian Legal NER project.

This module contains functions for confusion matrix analysis, plotting,
and detailed evaluation of NER models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


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


# ============================================================================
# NEW COMPREHENSIVE TRACKING FUNCTIONS FOR K-FOLD CV
# ============================================================================

def analyze_entity_distribution_per_fold(examples: List[Dict], fold_name: str = "Fold") -> Dict[str, int]:
    """
    Analyze and display entity type distribution in a dataset fold.

    Args:
        examples: List of examples with 'labels' field
        fold_name: Name of the fold for display

    Returns:
        Dictionary with entity type counts
    """
    entity_counts = Counter()

    for example in examples:
        labels = example.get('labels', [])
        for label in labels:
            if label != 'O':
                # Extract entity type from BIO label
                if '-' in label:
                    entity_type = label.split('-', 1)[1]
                    entity_counts[entity_type] += 1

    print(f"\n📊 Entity Distribution - {fold_name}")
    print("=" * 60)
    print(f"{'Entity Type':<30} {'Count':>10} {'Percentage':>15}")
    print("-" * 60)

    total = sum(entity_counts.values())
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{entity:<30} {count:>10} {percentage:>14.2f}%")

    print("-" * 60)
    print(f"{'TOTAL':<30} {total:>10} {100.0:>14.2f}%")

    return dict(entity_counts)


# REMOVED: plot_per_class_metrics_over_time - replaced by aggregate_f1_per_class_over_iterations


# REMOVED: plot_training_metrics_comprehensive - replaced by aggregate_training_metrics_across_folds


# REMOVED: plot_confusion_matrix_comprehensive - replaced by aggregate_confusion_matrix_across_folds


def generate_detailed_classification_report(true_labels: List[List[str]],
                                            predictions: List[List[str]],
                                            output_dir: str,
                                            fold_num: int = None,
                                            dataset_name: str = "Validation"):
    """
    Generate and save detailed classification report with per-class metrics.

    Args:
        true_labels: True labels (nested lists)
        predictions: Predicted labels (nested lists)
        output_dir: Directory to save report
        fold_num: Fold number for filename
        dataset_name: Name of dataset for title

    Returns:
        Dictionary with per-class metrics
    """
    # Flatten labels
    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predictions for label in seq]

    # Generate classification report
    report_text = classification_report(flat_true, flat_pred, digits=4, zero_division=0)

    # Print report
    fold_suffix = f" - Fold {fold_num}" if fold_num is not None else ""
    print(f"\n{'='*80}")
    print(f"{dataset_name}{fold_suffix} Set - Detailed Classification Report:")
    print(f"{'='*80}")
    print(report_text)

    # Save to file
    fold_suffix_file = f"_fold{fold_num}" if fold_num is not None else ""
    filename = f"{output_dir}/classification_report{fold_suffix_file}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{dataset_name}{fold_suffix} Set - Detailed Classification Report:\n")
        f.write("="*80 + "\n")
        f.write(report_text)

    print(f"✅ Classification report saved to: {filename}")

    # Parse report into dictionary
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true, flat_pred, average=None, zero_division=0, labels=sorted(list(set(flat_true)))
    )

    labels = sorted(list(set(flat_true)))
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    return per_class_metrics


def plot_f1_per_class_over_training(training_history: List[Dict],
                                    output_dir: str,
                                    fold_num: int = None):
    """
    Plot F1 measure for each output class vs training iterations.

    Args:
        training_history: List of evaluation results at different steps
        output_dir: Directory to save plot
        fold_num: Fold number for filename
    """
    if not training_history:
        print("⚠️  No training history available for F1 per class plotting")
        return

    # Extract F1 scores per class over time
    steps = []
    per_class_f1 = defaultdict(list)

    for entry in training_history:
        if 'step' in entry and 'per_class_metrics' in entry:
            steps.append(entry['step'])
            for label, metrics in entry['per_class_metrics'].items():
                per_class_f1[label].append(metrics.get('f1', 0))

    if not steps:
        print("⚠️  No per-class F1 metrics found in training history")
        return

    # Get entity labels (exclude 'O')
    entity_labels = [label for label in per_class_f1.keys() if label != 'O']

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot F1 for each class
    colors = plt.cm.tab20(np.linspace(0, 1, len(entity_labels)))
    for idx, label in enumerate(sorted(entity_labels)):
        if per_class_f1[label]:
            ax.plot(steps, per_class_f1[label], label=label,
                   linewidth=2, marker='o', markersize=3, color=colors[idx])

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Measure per Output Class vs Training Iterations',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    fold_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    filename = f"{output_dir}/f1_per_class_vs_iterations{fold_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ F1 per class plot saved to: {filename}")


# REMOVED: create_comprehensive_fold_report - replaced by individual calls + create_aggregate_report_across_folds


# ============================================================================
# AGGREGATE METRICS ACROSS ALL FOLDS
# ============================================================================

def aggregate_training_metrics_across_folds(fold_results: List[Dict],
                                            model_name: str = "Model",
                                            display: bool = True):
    """
    Aggregate and plot training metrics across all folds with mean and std.

    Args:
        fold_results: List of fold results, each containing training_history
        model_name: Name of model for titles
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATING TRAINING METRICS ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # Collect all training histories
    all_histories = [fold['training_history'] for fold in fold_results
                     if 'training_history' in fold]

    if not all_histories:
        print("⚠️  No training histories found in fold results")
        return

    # Extract evaluation metrics from each fold (align by evaluation index, not step number)
    eval_metrics_per_fold = []
    for history in all_histories:
        eval_entries = [entry for entry in history if 'eval_loss' in entry]
        if eval_entries:
            eval_metrics_per_fold.append(eval_entries)

    if not eval_metrics_per_fold:
        print("⚠️  No evaluation metrics found in training histories")
        return

    # Find the minimum number of evaluations across all folds
    min_evals = min(len(evals) for evals in eval_metrics_per_fold)

    if min_evals == 0:
        print("⚠️  No evaluation steps found")
        return

    print(f"📊 Aligning {min_evals} evaluation points across {len(eval_metrics_per_fold)} folds")

    # Aggregate metrics by evaluation index (not by step number)
    metrics_by_eval_idx = defaultdict(lambda: defaultdict(list))

    for fold_evals in eval_metrics_per_fold:
        for eval_idx in range(min_evals):
            entry = fold_evals[eval_idx]
            metrics_by_eval_idx[eval_idx]['step'].append(entry.get('step', 0))
            metrics_by_eval_idx[eval_idx]['loss'].append(entry.get('eval_loss', 0))
            metrics_by_eval_idx[eval_idx]['accuracy'].append(entry.get('eval_accuracy', 0))
            metrics_by_eval_idx[eval_idx]['precision'].append(entry.get('eval_precision', 0))
            metrics_by_eval_idx[eval_idx]['recall'].append(entry.get('eval_recall', 0))
            metrics_by_eval_idx[eval_idx]['f1'].append(entry.get('eval_f1', 0))

    # Calculate mean step numbers and metrics
    eval_indices = sorted(metrics_by_eval_idx.keys())
    steps = [np.mean(metrics_by_eval_idx[idx]['step']) for idx in eval_indices]

    loss_mean = [np.mean(metrics_by_eval_idx[idx]['loss']) for idx in eval_indices]
    loss_std = [np.std(metrics_by_eval_idx[idx]['loss']) for idx in eval_indices]

    acc_mean = [np.mean(metrics_by_eval_idx[idx]['accuracy']) for idx in eval_indices]
    acc_std = [np.std(metrics_by_eval_idx[idx]['accuracy']) for idx in eval_indices]

    prec_mean = [np.mean(metrics_by_eval_idx[idx]['precision']) for idx in eval_indices]
    prec_std = [np.std(metrics_by_eval_idx[idx]['precision']) for idx in eval_indices]

    rec_mean = [np.mean(metrics_by_eval_idx[idx]['recall']) for idx in eval_indices]
    rec_std = [np.std(metrics_by_eval_idx[idx]['recall']) for idx in eval_indices]

    f1_mean = [np.mean(metrics_by_eval_idx[idx]['f1']) for idx in eval_indices]
    f1_std = [np.std(metrics_by_eval_idx[idx]['f1']) for idx in eval_indices]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Training Metrics Across {len(fold_results)} Folds (Mean ± Std)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Validation Loss
    ax1.plot(steps, loss_mean, 'r-', linewidth=2, label='Mean Loss')
    ax1.fill_between(steps,
                     np.array(loss_mean) - np.array(loss_std),
                     np.array(loss_mean) + np.array(loss_std),
                     alpha=0.3, color='red')
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Model Optimization Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean Accuracy
    ax2.plot(steps, acc_mean, 'g-', linewidth=2, marker='o', markersize=4, label='Mean Accuracy')
    ax2.fill_between(steps,
                     np.array(acc_mean) - np.array(acc_std),
                     np.array(acc_mean) + np.array(acc_std),
                     alpha=0.3, color='green')
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Accuracy Over Training Iterations', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Plot 3: Precision and Recall
    ax3.plot(steps, prec_mean, 'b-', linewidth=2, marker='s', markersize=4, label='Precision')
    ax3.fill_between(steps,
                     np.array(prec_mean) - np.array(prec_std),
                     np.array(prec_mean) + np.array(prec_std),
                     alpha=0.3, color='blue')
    ax3.plot(steps, rec_mean, 'r-', linewidth=2, marker='^', markersize=4, label='Recall')
    ax3.fill_between(steps,
                     np.array(rec_mean) - np.array(rec_std),
                     np.array(rec_mean) + np.array(rec_std),
                     alpha=0.3, color='red')
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Precision and Recall Curves', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # Plot 4: F1 Score
    ax4.plot(steps, f1_mean, 'purple', linewidth=2, marker='D', markersize=4, label='F1-Score')
    ax4.fill_between(steps,
                     np.array(f1_mean) - np.array(f1_std),
                     np.array(f1_mean) + np.array(f1_std),
                     alpha=0.3, color='purple')
    ax4.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('F1-Score Over Training Iterations', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Aggregate training metrics visualization displayed")

    return {
        'steps': steps,
        'loss': {'mean': loss_mean, 'std': loss_std},
        'accuracy': {'mean': acc_mean, 'std': acc_std},
        'precision': {'mean': prec_mean, 'std': prec_std},
        'recall': {'mean': rec_mean, 'std': rec_std},
        'f1': {'mean': f1_mean, 'std': f1_std},
        'figure': fig if not display else None
    }


def aggregate_per_class_metrics_across_folds(fold_results: List[Dict],
                                             model_name: str = "Model",
                                             display: bool = True):
    """
    Aggregate per-class metrics across folds and create comprehensive report.

    Args:
        fold_results: List of fold results with per_class_metrics
        model_name: Name of model for titles
        display: If True, display report in notebook; if False, return as string
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATING PER-CLASS METRICS ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # Collect per-class metrics from all folds
    all_per_class = [fold['per_class_metrics'] for fold in fold_results
                     if 'per_class_metrics' in fold]

    if not all_per_class:
        print("⚠️  No per-class metrics found in fold results")
        return

    # Get all unique labels
    all_labels = set()
    for metrics in all_per_class:
        all_labels.update(metrics.keys())
    all_labels = sorted(list(all_labels))

    # Aggregate metrics for each label
    aggregated_metrics = {}
    for label in all_labels:
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for metrics in all_per_class:
            if label in metrics:
                precisions.append(metrics[label]['precision'])
                recalls.append(metrics[label]['recall'])
                f1s.append(metrics[label]['f1'])
                supports.append(metrics[label]['support'])

        aggregated_metrics[label] = {
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'support_mean': np.mean(supports),
            'support_std': np.std(supports)
        }

    # Generate detailed classification report
    print(f"\n{'='*80}")
    print(f"{model_name} - Aggregated Classification Report (Mean ± Std)")
    print(f"{'='*80}")
    print(f"\n{'Label':<25} {'Precision':>15} {'Recall':>15} {'F1-Score':>15} {'Support':>12}")
    print("-" * 85)

    # Sort labels: B- tags first, then I- tags, then O
    b_labels = [l for l in all_labels if l.startswith('B-')]
    i_labels = [l for l in all_labels if l.startswith('I-')]
    o_labels = [l for l in all_labels if l == 'O']
    sorted_labels = sorted(b_labels) + sorted(i_labels) + o_labels

    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in sorted_labels:
        metrics = aggregated_metrics[label]
        prec_str = f"{metrics['precision_mean']:.4f}±{metrics['precision_std']:.4f}"
        rec_str = f"{metrics['recall_mean']:.4f}±{metrics['recall_std']:.4f}"
        f1_str = f"{metrics['f1_mean']:.4f}±{metrics['f1_std']:.4f}"
        supp_str = f"{metrics['support_mean']:.0f}±{metrics['support_std']:.0f}"

        print(f"{label:<25} {prec_str:>15} {rec_str:>15} {f1_str:>15} {supp_str:>12}")

        # Calculate weighted averages
        total_support += metrics['support_mean']
        weighted_precision += metrics['precision_mean'] * metrics['support_mean']
        weighted_recall += metrics['recall_mean'] * metrics['support_mean']
        weighted_f1 += metrics['f1_mean'] * metrics['support_mean']

    # Print averages
    print("-" * 85)

    # Macro average
    macro_prec = np.mean([m['precision_mean'] for m in aggregated_metrics.values()])
    macro_rec = np.mean([m['recall_mean'] for m in aggregated_metrics.values()])
    macro_f1 = np.mean([m['f1_mean'] for m in aggregated_metrics.values()])

    print(f"{'macro avg':<25} {macro_prec:>15.4f} {macro_rec:>15.4f} {macro_f1:>15.4f} {total_support:>12.0f}")

    # Weighted average
    if total_support > 0:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support

    print(f"{'weighted avg':<25} {weighted_precision:>15.4f} {weighted_recall:>15.4f} {weighted_f1:>15.4f} {total_support:>12.0f}")

    print(f"\n✅ Aggregate classification report displayed")

    return aggregated_metrics


def aggregate_confusion_matrix_across_folds(fold_results: List[Dict],
                                            model_name: str = "Model",
                                            display: bool = True):
    """
    Aggregate confusion matrices across folds and visualize.

    Args:
        fold_results: List of fold results with confusion matrices
        model_name: Name of model for title
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATING CONFUSION MATRICES ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # Collect all confusion matrices and labels from each fold
    fold_cms = []
    fold_labels = []

    for fold in fold_results:
        if 'confusion_matrix' in fold and 'labels' in fold:
            fold_cms.append(fold['confusion_matrix'])
            fold_labels.append(fold['labels'])

    if not fold_cms:
        print("⚠️  No confusion matrices found in fold results")
        return

    # Get all unique labels across all folds
    all_labels_set = set()
    for labels in fold_labels:
        all_labels_set.update(labels)
    all_labels = sorted(list(all_labels_set))

    # Create a unified confusion matrix with all labels
    n_labels = len(all_labels)
    aggregate_cm = np.zeros((n_labels, n_labels), dtype=int)

    # Map each fold's confusion matrix to the unified label space
    for cm, labels in zip(fold_cms, fold_labels):
        # Create mapping from fold labels to unified labels
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        fold_label_to_idx = {label: idx for idx, label in enumerate(labels)}

        # Add this fold's confusion matrix to the aggregate
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                unified_i = label_to_idx[true_label]
                unified_j = label_to_idx[pred_label]
                aggregate_cm[unified_i, unified_j] += cm[i, j]

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(aggregate_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels,
                cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_title(f'Aggregate Confusion Matrix - {model_name} ({len(fold_results)} Folds)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Aggregate confusion matrix visualization displayed")

    return aggregate_cm, all_labels, fig if not display else None


def aggregate_f1_per_class_over_iterations(fold_results: List[Dict],
                                           model_name: str = "Model",
                                           display: bool = True):
    """
    Aggregate F1 scores per class over training iterations across folds.

    Args:
        fold_results: List of fold results with training histories
        model_name: Name of model for title
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATING F1 PER CLASS OVER ITERATIONS")
    print(f"{'='*80}")

    # Collect all training histories
    all_histories = [fold['training_history'] for fold in fold_results
                     if 'training_history' in fold]

    if not all_histories:
        print("⚠️  No training histories found")
        print("ℹ️  Skipping F1 per class over iterations plot")
        return

    # Extract per-class metrics from each fold (align by evaluation index)
    per_class_metrics_per_fold = []
    for history in all_histories:
        eval_entries = [entry for entry in history if 'per_class_metrics' in entry and entry['per_class_metrics']]
        if eval_entries:
            per_class_metrics_per_fold.append(eval_entries)

    if not per_class_metrics_per_fold:
        print("⚠️  No per-class metrics found in training histories")
        print("ℹ️  This is normal if PerClassMetricsCallback didn't have access to model/eval_dataset during training")
        print("ℹ️  Per-class metrics are still available in the final classification report")
        print("ℹ️  Skipping F1 per class over iterations plot")
        return

    # Find minimum number of evaluations
    min_evals = min(len(evals) for evals in per_class_metrics_per_fold)

    if min_evals == 0:
        print("⚠️  No evaluation steps found")
        print("ℹ️  Skipping F1 per class over iterations plot")
        return

    print(f"📊 Aligning {min_evals} evaluation points across {len(per_class_metrics_per_fold)} folds")

    # Collect F1 scores per class at each evaluation index
    f1_by_class_and_eval = defaultdict(lambda: defaultdict(list))
    steps_by_eval = defaultdict(list)

    for fold_evals in per_class_metrics_per_fold:
        for eval_idx in range(min_evals):
            entry = fold_evals[eval_idx]
            steps_by_eval[eval_idx].append(entry.get('step', 0))

            for label, metrics in entry.get('per_class_metrics', {}).items():
                if label != 'O':  # Exclude 'O' label
                    f1_by_class_and_eval[label][eval_idx].append(metrics.get('f1', 0))

    # Get all entity labels
    entity_labels = sorted([l for l in f1_by_class_and_eval.keys() if l != 'O'])

    if not entity_labels:
        print("⚠️  No entity labels found")
        return

    # Calculate mean steps for x-axis
    eval_indices = sorted(steps_by_eval.keys())
    mean_steps = [np.mean(steps_by_eval[idx]) for idx in eval_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(entity_labels)))

    for idx, label in enumerate(entity_labels):
        f1_means = [np.mean(f1_by_class_and_eval[label][eval_idx])
                   for eval_idx in eval_indices]
        f1_stds = [np.std(f1_by_class_and_eval[label][eval_idx])
                  for eval_idx in eval_indices]

        ax.plot(mean_steps, f1_means, label=label,
               linewidth=2, marker='o', markersize=3, color=colors[idx])
        ax.fill_between(mean_steps,
                       np.array(f1_means) - np.array(f1_stds),
                       np.array(f1_means) + np.array(f1_stds),
                       alpha=0.2, color=colors[idx])

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title(f'F1 Measure per Output Class vs Training Iterations - {model_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Aggregate F1 per class visualization displayed")


def plot_entity_distribution_across_folds(fold_results: List[Dict],
                                          model_name: str = "Model",
                                          display: bool = True):
    """
    Plot entity type distribution across all folds with grouped bar chart.

    Args:
        fold_results: List of fold results with 'distributions' field
        model_name: Name of model for title
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"PLOTTING ENTITY DISTRIBUTION ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # Extract entity distributions from each fold
    fold_distributions = []
    for fold in fold_results:
        if 'distributions' in fold and 'val' in fold['distributions']:
            fold_distributions.append(fold['distributions']['val'])

    if not fold_distributions:
        print("⚠️  No entity distributions found in fold results")
        return

    # Get all unique entity types
    all_entities = set()
    for dist in fold_distributions:
        all_entities.update(dist.keys())
    all_entities = sorted(list(all_entities))

    # Prepare data for grouped bar chart
    n_folds = len(fold_distributions)
    n_entities = len(all_entities)

    # Create matrix: rows = entities, columns = folds
    data_matrix = np.zeros((n_entities, n_folds))
    for fold_idx, dist in enumerate(fold_distributions):
        for entity_idx, entity in enumerate(all_entities):
            data_matrix[entity_idx, fold_idx] = dist.get(entity, 0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(n_entities)
    width = 0.15  # Width of each bar

    colors = plt.cm.Set3(np.linspace(0, 1, n_folds))

    for fold_idx in range(n_folds):
        offset = (fold_idx - n_folds/2) * width + width/2
        bars = ax.bar(x + offset, data_matrix[:, fold_idx], width,
                     label=f'Fold {fold_idx + 1}', color=colors[fold_idx],
                     edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Entity Types', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Appearances', fontsize=13, fontweight='bold')
    ax.set_title(f'Entity Type Distribution Across {n_folds} Cross-Validation Folds - {model_name}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_entities, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Entity distribution across folds visualization displayed")

    return fig if not display else None


def plot_training_validation_loss_across_folds(fold_results: List[Dict],
                                                model_name: str = "Model",
                                                display: bool = True):
    """
    Plot training and validation loss curves across all folds.

    Args:
        fold_results: List of fold results with training_history
        model_name: Name of model for title
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"PLOTTING TRAINING AND VALIDATION LOSS ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # Collect all training histories
    all_histories = [fold['training_history'] for fold in fold_results
                     if 'training_history' in fold]

    if not all_histories:
        print("⚠️  No training histories found in fold results")
        return

    # Extract training and validation loss from each fold
    train_losses_per_fold = []
    val_losses_per_fold = []
    train_steps_per_fold = []
    val_steps_per_fold = []

    for history in all_histories:
        # Training loss - entries with 'loss' but NOT 'eval_loss'
        train_entries = [(entry['step'], entry['loss'])
                        for entry in history
                        if 'loss' in entry and 'eval_loss' not in entry]
        if train_entries:
            steps, losses = zip(*train_entries)
            train_steps_per_fold.append(list(steps))
            train_losses_per_fold.append(list(losses))

        # Validation loss - entries with 'eval_loss'
        val_entries = [(entry['step'], entry['eval_loss'])
                      for entry in history if 'eval_loss' in entry]
        if val_entries:
            steps, losses = zip(*val_entries)
            val_steps_per_fold.append(list(steps))
            val_losses_per_fold.append(list(losses))

    if not train_losses_per_fold and not val_losses_per_fold:
        print("⚠️  No loss data found in training histories")
        return

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Training and Validation Loss Across Folds - {model_name}',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))

    # Plot 1: Training Loss
    if train_losses_per_fold:
        for fold_idx, (steps, losses) in enumerate(zip(train_steps_per_fold, train_losses_per_fold)):
            ax1.plot(steps, losses, label=f'Fold {fold_idx + 1}',
                    color=colors[fold_idx], linewidth=2, alpha=0.7)
        ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Over Iterations', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No training loss data available',
                ha='center', va='center', fontsize=12)
        ax1.set_title('Training Loss Over Iterations', fontsize=13, fontweight='bold')

    # Plot 2: Validation Loss
    if val_losses_per_fold:
        for fold_idx, (steps, losses) in enumerate(zip(val_steps_per_fold, val_losses_per_fold)):
            ax2.plot(steps, losses, label=f'Fold {fold_idx + 1}',
                    color=colors[fold_idx], linewidth=2, alpha=0.7, marker='o', markersize=4)
        ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Loss Over Iterations', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No validation loss data available',
                ha='center', va='center', fontsize=12)
        ax2.set_title('Validation Loss Over Iterations', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Training and validation loss visualization displayed")

    return fig if not display else None


def plot_macro_micro_metrics_over_iterations(fold_results: List[Dict],
                                             model_name: str = "Model",
                                             display: bool = True):
    """
    Plot macro and micro-averaged precision, recall, and F1 over training iterations.

    Args:
        fold_results: List of fold results with training_history
        model_name: Name of model for title
        display: If True, display plot in notebook; if False, return figure object
    """
    print(f"\n{'='*80}")
    print(f"PLOTTING MACRO/MICRO-AVERAGED METRICS OVER ITERATIONS")
    print(f"{'='*80}")

    # Collect all training histories
    all_histories = [fold['training_history'] for fold in fold_results
                     if 'training_history' in fold]

    if not all_histories:
        print("⚠️  No training histories found in fold results")
        return

    # Extract evaluation metrics from each fold
    eval_metrics_per_fold = []
    for history in all_histories:
        # Filter for entries that have evaluation metrics
        eval_entries = [entry for entry in history
                       if 'eval_loss' in entry and 'eval_precision' in entry]
        if eval_entries:
            eval_metrics_per_fold.append(eval_entries)

    if not eval_metrics_per_fold:
        print("⚠️  No evaluation metrics found in training histories")
        print("    Make sure your TrainingArguments has eval_steps set and evaluation is running")
        print("    ")
        print("    Debug info:")
        if all_histories:
            sample_history = all_histories[0]
            if sample_history:
                print(f"      - Training history has {len(sample_history)} entries")
                print(f"      - Sample entry keys: {list(sample_history[0].keys()) if sample_history else 'N/A'}")
                eval_entries_count = sum(1 for e in sample_history if 'eval_loss' in e)
                print(f"      - Entries with 'eval_loss': {eval_entries_count}")
                eval_prec_count = sum(1 for e in sample_history if 'eval_precision' in e)
                print(f"      - Entries with 'eval_precision': {eval_prec_count}")
        return

    # Find minimum number of evaluations
    min_evals = min(len(evals) for evals in eval_metrics_per_fold)

    if min_evals == 0:
        print("⚠️  No evaluation steps found")
        return

    print(f"📊 Aligning {min_evals} evaluation points across {len(eval_metrics_per_fold)} folds")

    # Aggregate metrics by evaluation index
    metrics_by_eval_idx = defaultdict(lambda: defaultdict(list))

    for fold_evals in eval_metrics_per_fold:
        for eval_idx in range(min_evals):
            entry = fold_evals[eval_idx]
            metrics_by_eval_idx[eval_idx]['step'].append(entry.get('step', 0))
            metrics_by_eval_idx[eval_idx]['precision'].append(entry.get('eval_precision', 0))
            metrics_by_eval_idx[eval_idx]['recall'].append(entry.get('eval_recall', 0))
            metrics_by_eval_idx[eval_idx]['f1'].append(entry.get('eval_f1', 0))
            metrics_by_eval_idx[eval_idx]['accuracy'].append(entry.get('eval_accuracy', 0))

    # Calculate mean and std
    eval_indices = sorted(metrics_by_eval_idx.keys())
    steps = [np.mean(metrics_by_eval_idx[idx]['step']) for idx in eval_indices]

    prec_mean = [np.mean(metrics_by_eval_idx[idx]['precision']) for idx in eval_indices]
    prec_std = [np.std(metrics_by_eval_idx[idx]['precision']) for idx in eval_indices]

    rec_mean = [np.mean(metrics_by_eval_idx[idx]['recall']) for idx in eval_indices]
    rec_std = [np.std(metrics_by_eval_idx[idx]['recall']) for idx in eval_indices]

    f1_mean = [np.mean(metrics_by_eval_idx[idx]['f1']) for idx in eval_indices]
    f1_std = [np.std(metrics_by_eval_idx[idx]['f1']) for idx in eval_indices]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Macro-Averaged Performance Metrics Over Iterations - {model_name}\n(Mean ± Std across {len(fold_results)} folds)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Precision
    ax1.plot(steps, prec_mean, 'b-', linewidth=2.5, marker='o', markersize=5, label='Precision')
    ax1.fill_between(steps,
                     np.array(prec_mean) - np.array(prec_std),
                     np.array(prec_mean) + np.array(prec_std),
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Macro-Averaged Precision', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Recall
    ax2.plot(steps, rec_mean, 'r-', linewidth=2.5, marker='s', markersize=5, label='Recall')
    ax2.fill_between(steps,
                     np.array(rec_mean) - np.array(rec_std),
                     np.array(rec_mean) + np.array(rec_std),
                     alpha=0.3, color='red')
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Macro-Averaged Recall', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Plot 3: F1-Score
    ax3.plot(steps, f1_mean, 'g-', linewidth=2.5, marker='^', markersize=5, label='F1-Score')
    ax3.fill_between(steps,
                     np.array(f1_mean) - np.array(f1_std),
                     np.array(f1_mean) + np.array(f1_std),
                     alpha=0.3, color='green')
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('Macro-Averaged F1-Score', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # Plot 4: All metrics together
    ax4.plot(steps, prec_mean, 'b-', linewidth=2, marker='o', markersize=4, label='Precision')
    ax4.plot(steps, rec_mean, 'r-', linewidth=2, marker='s', markersize=4, label='Recall')
    ax4.plot(steps, f1_mean, 'g-', linewidth=2, marker='^', markersize=4, label='F1-Score')
    ax4.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('All Metrics Combined', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close()

    print(f"✅ Macro/micro-averaged metrics visualization displayed")

    return fig if not display else None


def create_aggregate_report_across_folds(fold_results: List[Dict],
                                         model_name: str = "Model",
                                         display: bool = True):
    """
    Create comprehensive aggregate report across all folds with visualizations displayed in notebook.

    Args:
        fold_results: List of fold results
        model_name: Name of model
        display: If True, display visualizations in notebook; if False, return figure objects
    """
    print(f"\n{'='*80}")
    print(f"CREATING AGGREGATE REPORT ACROSS {len(fold_results)} FOLDS")
    print(f"{'='*80}")

    # 1. Plot entity distribution across folds
    print(f"\n📊 Plotting entity distribution across folds...")
    plot_entity_distribution_across_folds(fold_results, model_name, display)

    # 2. Plot training and validation loss
    print(f"\n📉 Plotting training and validation loss...")
    plot_training_validation_loss_across_folds(fold_results, model_name, display)

    # 3. Plot macro/micro-averaged metrics over iterations
    print(f"\n📈 Plotting macro/micro-averaged metrics over iterations...")
    plot_macro_micro_metrics_over_iterations(fold_results, model_name, display)

    # 4. Aggregate training metrics
    print(f"\n📈 Aggregating training metrics...")
    training_metrics = aggregate_training_metrics_across_folds(
        fold_results, model_name, display
    )

    # 5. Aggregate per-class metrics
    print(f"\n📊 Aggregating per-class metrics...")
    per_class_metrics = aggregate_per_class_metrics_across_folds(
        fold_results, model_name, display
    )

    # 6. Aggregate confusion matrix
    print(f"\n🔥 Aggregating confusion matrices...")
    cm_result = aggregate_confusion_matrix_across_folds(
        fold_results, model_name, display
    )
    cm, labels = cm_result[0], cm_result[1]

    # 7. Aggregate F1 per class over iterations
    print(f"\n📈 Aggregating F1 per class over iterations...")
    aggregate_f1_per_class_over_iterations(
        fold_results, model_name, display
    )

    print(f"\n✅ Aggregate report completed!")
    print(f"\nAll visualizations displayed in notebook above.")

    return {
        'training_metrics': training_metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'labels': labels
    }

