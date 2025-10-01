# Class Weights Configuration (add to notebook)
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

# Custom Weighted Trainer (add to notebook)
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


def calculate_class_weights(dataset, label_to_id, method='balanced'):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        dataset: NER dataset with examples
        label_to_id: Mapping from label names to IDs
        method: 'balanced' or 'log_balanced'
    
    Returns:
        Dictionary of class weights
    """
    from collections import Counter
    import numpy as np
    
    # Count label frequencies
    label_counts = Counter()
    
    for example in dataset:
        for label in example['labels']:
            if label != 'O':  # Ignore 'O' labels
                label_counts[label] += 1
    
    # Calculate weights
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    
    class_weights = {}
    
    for label, count in label_counts.items():
        if method == 'balanced':
            weight = total_samples / (num_classes * count)
        elif method == 'log_balanced':
            weight = np.log(total_samples / count)
        else:
            weight = 1.0
        
        class_weights[label] = weight
    
    # Add weight for 'O' label (usually 1.0)
    class_weights['O'] = 1.0
    
    print(f"📊 Calculated class weights ({method}):")
    for label, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {weight:.2f}")
    
    return class_weights
