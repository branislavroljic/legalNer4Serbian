"""
Improved Training Configuration for Serbian Legal NER

This file contains optimized training parameters and techniques to address
the poor performance issues in the original pipeline.

Key improvements:
1. Better learning rate and training schedule
2. Class weighting to handle imbalance
3. Improved evaluation metrics
4. Data augmentation suggestions
"""

import torch
import numpy as np
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


def create_improved_training_args(output_dir, num_train_examples):
    """
    Create improved training arguments optimized for NER with class imbalance.
    
    Args:
        output_dir: Directory to save model outputs
        num_train_examples: Number of training examples for scheduling
    
    Returns:
        TrainingArguments with optimized settings
    """
    
    # Calculate steps per epoch for better scheduling
    batch_size = 4  # Smaller batch size for better gradients
    gradient_accumulation_steps = 4  # Effective batch size = 16
    steps_per_epoch = max(1, num_train_examples // (batch_size * gradient_accumulation_steps))
    
    # More frequent evaluation for small datasets
    eval_steps = max(10, steps_per_epoch // 4)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # ✅ IMPROVED: More epochs for better convergence
        num_train_epochs=20,  # Increased from 5 to 20
        
        # ✅ IMPROVED: Smaller batch size for better gradients with class imbalance
        per_device_train_batch_size=4,  # Reduced from 8 to 4
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # ✅ IMPROVED: Higher learning rate for NER fine-tuning
        learning_rate=5e-5,  # Increased from 2e-5 to 5e-5
        
        # ✅ IMPROVED: Reduced warmup for smaller dataset
        warmup_steps=50,  # Reduced from 500 to 50
        warmup_ratio=0.1,  # Alternative warmup strategy
        
        # ✅ IMPROVED: Better weight decay
        weight_decay=0.01,
        
        # ✅ IMPROVED: More frequent evaluation and saving
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps", 
        save_steps=eval_steps,
        
        # ✅ IMPROVED: Keep more checkpoints and better monitoring
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # ✅ IMPROVED: More frequent logging
        logging_steps=max(5, eval_steps // 2),
        logging_dir=f'{output_dir}/logs',
        
        # ✅ IMPROVED: Better optimization settings
        optim="adamw_torch",  # Explicit optimizer
        lr_scheduler_type="cosine",  # Better learning rate schedule
        
        # ✅ IMPROVED: Early stopping with more patience
        # (Will be added via callback)
        
        # Technical settings
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        seed=42,
        
        # ✅ IMPROVED: Better evaluation
        eval_accumulation_steps=1,
        prediction_loss_only=False,
    )
    
    return training_args


def compute_class_weights(train_labels, label_to_id):
    """
    Compute class weights to handle class imbalance.
    
    Args:
        train_labels: List of label sequences
        label_to_id: Mapping from labels to IDs
    
    Returns:
        Dictionary mapping label IDs to weights
    """
    # Flatten all labels
    flat_labels = []
    for label_seq in train_labels:
        for label in label_seq:
            if isinstance(label, str):
                flat_labels.append(label)
            else:
                # Convert back to string if needed
                id_to_label = {v: k for k, v in label_to_id.items()}
                flat_labels.append(id_to_label.get(label, 'O'))
    
    # Count label frequencies
    label_counts = Counter(flat_labels)
    print(f"\nLabel distribution in training data:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(flat_labels)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    # Compute class weights
    unique_labels = list(label_counts.keys())
    label_frequencies = [label_counts[label] for label in unique_labels]
    
    # Use sklearn's balanced class weight computation
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(range(len(unique_labels))),
        y=np.array([unique_labels.index(label) for label in flat_labels])
    )
    
    # Create mapping from label to weight
    label_weights = {}
    for i, label in enumerate(unique_labels):
        if label in label_to_id:
            label_id = label_to_id[label]
            label_weights[label_id] = class_weights[i]
    
    print(f"\nComputed class weights:")
    for label_id, weight in sorted(label_weights.items()):
        id_to_label = {v: k for k, v in label_to_id.items()}
        label_name = id_to_label.get(label_id, f"ID_{label_id}")
        print(f"  {label_name}: {weight:.3f}")
    
    return label_weights


def create_weighted_trainer_class(class_weights):
    """
    Create a custom Trainer class that uses class weights for loss computation.
    
    Args:
        class_weights: Dictionary mapping label IDs to weights
    
    Returns:
        Custom Trainer class
    """
    from transformers import Trainer
    
    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Convert class weights to tensor
            if class_weights:
                max_label_id = max(class_weights.keys())
                weight_tensor = torch.ones(max_label_id + 1)
                for label_id, weight in class_weights.items():
                    weight_tensor[label_id] = weight
                self.class_weights = weight_tensor.to(self.model.device)
            else:
                self.class_weights = None
        
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Compute loss with class weights to handle imbalance.
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            
            if labels is not None and self.class_weights is not None:
                # Move class weights to the same device as the model
                if self.class_weights.device != outputs.logits.device:
                    self.class_weights = self.class_weights.to(outputs.logits.device)
                
                # Compute weighted cross entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    ignore_index=-100
                )
                
                # Flatten for loss computation
                active_loss = labels.view(-1) != -100
                active_logits = outputs.logits.view(-1, self.model.config.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss
    
    return WeightedTrainer


def create_improved_callbacks():
    """
    Create improved callbacks for training.
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # More patient early stopping for small datasets
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
    
    return callbacks


def print_training_recommendations():
    """
    Print recommendations for improving NER performance.
    """
    print("\n" + "="*80)
    print("🔧 RECOMMENDATIONS FOR IMPROVING NER PERFORMANCE")
    print("="*80)
    
    print("\n1. 📊 DATA IMPROVEMENTS:")
    print("   • Add more annotated documents (aim for 200+ documents)")
    print("   • Ensure balanced representation of all entity types")
    print("   • Review annotation quality and consistency")
    print("   • Consider data augmentation techniques")
    
    print("\n2. 🎯 MODEL IMPROVEMENTS:")
    print("   • Use class weighting (implemented above)")
    print("   • Try different learning rates (3e-5, 5e-5, 1e-4)")
    print("   • Experiment with different models (Serbian BERT variants)")
    print("   • Consider ensemble methods")
    
    print("\n3. 🔄 TRAINING IMPROVEMENTS:")
    print("   • Increase training epochs (15-25)")
    print("   • Use smaller batch sizes (2-4)")
    print("   • Implement curriculum learning")
    print("   • Add regularization techniques")
    
    print("\n4. 📈 EVALUATION IMPROVEMENTS:")
    print("   • Use entity-level F1 score (not token-level)")
    print("   • Implement cross-validation")
    print("   • Analyze per-entity performance")
    print("   • Check for overfitting")
    
    print("\n5. 🛠️ TECHNICAL IMPROVEMENTS:")
    print("   • Optimize sliding window parameters")
    print("   • Handle long sequences better")
    print("   • Improve tokenization alignment")
    print("   • Add post-processing rules")
    
    print("="*80)


if __name__ == "__main__":
    print_training_recommendations()
