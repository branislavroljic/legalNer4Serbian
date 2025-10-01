# Class Weights Functions to Add to Class Weights Notebook
# Copy this code into a new cell in the Class Weights notebook after the imports

import torch
import torch.nn as nn
from transformers import Trainer
from collections import Counter
import numpy as np

# Class Weights Configuration (notebook-specific)
DEFAULT_CLASS_WEIGHTS = {
    "CASE_NUMBER": 43.24,
    "JUDGE": 22.90,
    "REGISTRAR": 23.68,
    "SANCTION_TYPE": 25.78,
    "PROCEDURE_COSTS": 23.13,
    "COURT": 1.0,
    "CRIMINAL_ACT": 1.0,
    "DEFENDANT": 1.0,
    "PROSECUTOR": 1.0,
    "PROVISION": 1.0
}

def calculate_class_weights(examples, label_to_id, method="inverse_frequency"):
    """Calculate class weights based on label frequency"""
    
    # Count label frequencies
    label_counts = Counter()
    total_tokens = 0
    
    for example in examples:
        for label in example['labels']:
            if label != 'O':  # Ignore 'O' labels
                entity_type = label.split('-')[-1] if '-' in label else label
                label_counts[entity_type] += 1
                total_tokens += 1
    
    # Calculate weights
    weights = {}
    
    if method == "inverse_frequency":
        # Inverse frequency weighting
        for entity_type in label_counts:
            frequency = label_counts[entity_type] / total_tokens
            weights[entity_type] = 1.0 / frequency
            
        # Normalize weights
        min_weight = min(weights.values())
        for entity_type in weights:
            weights[entity_type] = weights[entity_type] / min_weight
            
    elif method == "balanced":
        # Balanced class weights (sklearn style)
        n_classes = len(label_counts)
        for entity_type in label_counts:
            weights[entity_type] = total_tokens / (n_classes * label_counts[entity_type])
    
    return weights

# Custom Weighted Trainer (notebook-specific)
class WeightedTrainer(Trainer):
    """Custom trainer that applies class weights to the loss function"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            # Convert to tensor and move to device
            self.class_weights_tensor = torch.tensor(
                list(class_weights.values()), 
                dtype=torch.float32
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to apply class weights
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None and self.class_weights is not None:
            # Move class weights to the same device as the model
            if self.class_weights_tensor.device != outputs.logits.device:
                self.class_weights_tensor = self.class_weights_tensor.to(outputs.logits.device)
            
            # Compute weighted cross entropy loss
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor, ignore_index=-100)
            
            # Flatten for loss computation
            active_loss = inputs["attention_mask"].view(-1) == 1
            active_logits = outputs.logits.view(-1, self.model.config.num_labels)
            active_labels = torch.where(
                active_loss, 
                labels.view(-1), 
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            
            loss = loss_fct(active_logits, active_labels)
        else:
            # Use default loss if no class weights
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

print("✅ Class weights functions and WeightedTrainer defined")
