"""
Shared model utilities for Serbian Legal NER project.

This module contains functions for model loading, training configuration,
and evaluation metrics.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report


def load_model_and_tokenizer(model_name: str, num_labels: int, 
                           id2label: Dict, label2id: Dict) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """Load pre-trained model and tokenizer for NER"""
    
    print(f"🔄 Loading model and tokenizer...")
    print(f"📥 Model: {model_name}")
    print(f"🏷️  Number of labels: {num_labels}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ Loaded tokenizer (vocab size: {tokenizer.vocab_size})")
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Loaded model (parameters: {model.num_parameters():,})")
    print(f"🖥️  Device: {device}")
    
    return model, tokenizer


def create_training_arguments(output_dir: str, num_epochs: int = 8, 
                            batch_size: int = 4, learning_rate: float = 3e-5,
                            warmup_steps: int = 500, weight_decay: float = 0.01,
                            eval_steps: int = 100, save_steps: int = 500,
                            logging_steps: int = 50, early_stopping_patience: int = 3) -> TrainingArguments:
    """Create training arguments for the Trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None  # Disable wandb/tensorboard
    )
    
    print(f"⚙️  Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    
    return training_args


def compute_metrics(eval_pred, id_to_label: Dict[int, str]):
    """Compute evaluation metrics for NER (supports both standard and CRF models)"""
    predictions, labels = eval_pred

    # Handle both CRF (2D) and standard model (3D) predictions
    if len(predictions.shape) == 3:
        # Standard model: (batch, seq, num_labels) -> apply argmax
        predictions = np.argmax(predictions, axis=2)
    # else: CRF model: (batch, seq) -> already decoded

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate metrics
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

    return results


def create_trainer(model, training_args, train_dataset, val_dataset, 
                  data_collator, tokenizer, id_to_label: Dict[int, str],
                  early_stopping_patience: int = 3) -> Trainer:
    """Create and configure the Trainer"""
    
    # Create compute_metrics function with id_to_label bound
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, id_to_label)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    print(f"🏋️  Created trainer with early stopping (patience: {early_stopping_patience})")
    print(f"📊 Training dataset size: {len(train_dataset)}")
    print(f"📊 Validation dataset size: {len(val_dataset)}")
    
    return trainer


def detailed_evaluation(trainer: Trainer, dataset, dataset_name: str = "Test",
                       id_to_label: Dict[int, str] = None) -> Dict:
    """Perform detailed evaluation with per-entity metrics"""
    predictions = trainer.predict(dataset)

    # Handle both CRF (2D) and standard model (3D) predictions
    if len(predictions.predictions.shape) == 3:
        # Standard model: (batch, seq, num_labels) -> apply argmax
        y_pred = np.argmax(predictions.predictions, axis=2)
    else:
        # CRF model: (batch, seq) -> already decoded
        y_pred = predictions.predictions

    y_true = predictions.label_ids

    # Convert to label strings
    if id_to_label is None:
        id_to_label = trainer.model.config.id2label

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_pred, y_true)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_pred, y_true)
    ]

    # Calculate overall metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    print(f"\n📊 {dataset_name} Set Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Flatten for classification report
    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in true_predictions for label in seq]

    # Print classification report
    print(f"\n{dataset_name} Set - Detailed Classification Report:")
    print(classification_report(flat_true, flat_pred, zero_division=0))

    # Return both metrics and predictions
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_predictions': true_predictions,
        'true_labels': true_labels,
        'flat_true': flat_true,
        'flat_pred': flat_pred
    }


def save_model_info(output_dir: str, model_name: str, num_labels: int,
                   label_to_id: Dict[str, int], id_to_label: Dict[int, str],
                   model_type: str = "standard", entity_types: List[str] = None,
                   base_model_name: str = None, additional_info: Dict = None,
                   training_args = None):
    """Save model configuration and metadata"""
    
    model_info = {
        "model_name": model_name,
        "num_labels": num_labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "model_type": model_type
    }

    if entity_types:
        model_info["entity_types"] = sorted(entity_types)
    
    if base_model_name:
        model_info["base_model_name"] = base_model_name
    
    if additional_info:
        model_info.update(additional_info)
    
    # Save model info
    info_path = Path(output_dir) / "model_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Model information saved to {info_path}")
    return model_info


def load_model_info(model_path: str) -> Optional[Dict]:
    """Load model information from saved model directory"""
    info_path = Path(model_path) / "model_info.json"
    
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


def calculate_class_weights(entity_counts: Dict[str, int], method: str = "inverse_frequency") -> Dict[str, float]:
    """Calculate class weights for handling imbalanced datasets"""
    
    if method == "inverse_frequency":
        max_count = max(entity_counts.values()) if entity_counts else 1
        weights = {}
        
        for entity, count in entity_counts.items():
            weights[entity] = max_count / count
        
        print(f"🔢 Calculated class weights (inverse frequency):")
        for entity, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity}: {weight:.2f}")
        
        return weights
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def setup_device_and_seed(seed: int = 42) -> torch.device:
    """Setup device and random seeds for reproducibility"""
    import random
    import numpy as np

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Setup complete:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")
    print(f"  Random seed: {seed}")

    return device


def load_inference_pipeline(model_path: str, max_length: int = 512, stride: int = 128):
    """Load trained model for inference"""
    from transformers import pipeline

    try:
        # Load the model info to get configuration
        model_info = load_model_info(model_path)

        if model_info:
            print(f"📥 Loading model: {model_info.get('model_name', 'Unknown')}")
            print(f"🏷️  Model type: {model_info.get('model_type', 'standard')}")
            print(f"📊 Number of labels: {model_info.get('num_labels', 'Unknown')}")

        # Create inference pipeline
        ner_pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

        print(f"✅ Inference pipeline loaded successfully")
        print(f"⚙️  Max length: {max_length}, Stride: {stride}")

        return ner_pipeline

    except Exception as e:
        print(f"❌ Error loading inference pipeline: {e}")
        return None
