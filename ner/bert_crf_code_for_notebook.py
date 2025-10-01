# BERT-CRF Configuration (add to notebook)
BERT_CRF_CONFIG = {
    "dropout_rate": 0.1,
    "bert_lr": 3e-5,
    "classifier_lr": 1e-4,
    "crf_lr": 1e-3
}

# BERT-CRF Classes (add to notebook)
class BertCrfForTokenClassification(nn.Module):
    """BERT model with CRF layer for token classification."""

    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.config = self.bert.config
        self.config.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones_like(input_ids).bool()
        
        outputs_dict = {
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
        
        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            outputs_dict['loss'] = loss
        
        class ModelOutput:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        return ModelOutput(**outputs_dict)

    def predict(self, input_ids, attention_mask=None):
        """Predict using CRF Viterbi decoding"""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
            
            if attention_mask is not None:
                mask = attention_mask.bool()
            else:
                mask = torch.ones_like(input_ids).bool()
            
            predictions = self.crf.decode(logits, mask=mask)
            return predictions


class BertCrfTrainer(Trainer):
    """Custom trainer for BERT-CRF model that handles CRF predictions"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            if 'labels' in inputs:
                loss = outputs.loss
            else:
                loss = None

            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)
            predictions = model.predict(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_size, seq_len = input_ids.shape
            pred_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
            
            for i, pred_seq in enumerate(predictions):
                pred_len = min(len(pred_seq), seq_len)
                pred_tensor[i, :pred_len] = torch.tensor(pred_seq[:pred_len], device=input_ids.device)

            labels = inputs.get('labels', None)

        return (loss, pred_tensor, labels)


def compute_crf_metrics(eval_pred, id_to_label: Dict[int, str]):
    """Compute evaluation metrics for BERT-CRF model."""
    predictions, labels = eval_pred

    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

    return results
