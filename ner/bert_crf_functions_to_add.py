# BERT-CRF Functions to Add to BERT-CRF Notebook
# Copy this code into a new cell in the BERT-CRF notebook after the imports

# Install pytorch-crf if not available
try:
    from torchcrf import CRF
except ImportError:
    !pip install pytorch-crf
    from torchcrf import CRF

import torch
import torch.nn as nn
from transformers import Trainer

# BERT-CRF Model Implementation (notebook-specific)
class BertCrfForTokenClassification(nn.Module):
    """BERT model with CRF layer for token classification"""
    
    def __init__(self, bert_model, num_labels, dropout_rate=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            # Create mask for CRF (exclude padding and special tokens)
            mask = (attention_mask == 1) & (labels != -100)
            
            # Replace -100 with 0 for CRF computation
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0
            
            # Compute CRF loss
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return {'loss': loss, 'logits': logits}
        else:
            # Decode best path
            mask = attention_mask == 1
            predictions = self.crf.decode(logits, mask=mask)
            return {'logits': logits, 'predictions': predictions}

def create_bert_crf_model(model_name, num_labels, dropout_rate=0.1):
    """Create BERT-CRF model"""
    from transformers import AutoModel
    
    # Load BERT model (without classification head)
    bert_model = AutoModel.from_pretrained(model_name)
    
    # Create BERT-CRF model
    model = BertCrfForTokenClassification(
        bert_model=bert_model,
        num_labels=num_labels,
        dropout_rate=dropout_rate
    )
    
    return model

# Note: compute_crf_metrics and create_bert_crf_trainer functions have been
# moved to shared/model_utils.py as compute_metrics and create_trainer.
# The shared functions now handle both standard and CRF models automatically.

print("✅ BERT-CRF model class defined")
print("ℹ️  Use shared.model_utils.compute_metrics and shared.model_utils.create_trainer")
