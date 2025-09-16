"""
Sliding Window Inference Pipeline for Serbian Legal NER

This module provides an inference pipeline that can handle long sequences
using sliding windows, similar to the training approach.
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification


class SerbianLegalNERPipelineWithSlidingWindow:
    """Inference pipeline for Serbian legal NER with sliding window support"""
    
    def __init__(self, model_path: str, max_length: int = 512, stride: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load label mappings
        self.id_to_label = self.model.config.id2label
        self.label_to_id = self.model.config.label2id
        
        # Sliding window parameters
        self.max_length = max_length
        self.stride = stride
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, text: str) -> List[Dict]:
        """
        Predict entities from text using sliding windows if necessary.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of entity dictionaries with 'text', 'label', 'start', 'end'
        """
        # Tokenize the entire text first
        tokens = text.split()  # Use whitespace tokenization first
        
        # Convert to WordPiece tokens
        wordpiece_tokens = []
        token_to_wordpiece_map = []  # Maps original token index to wordpiece token indices
        
        for token_idx, token in enumerate(tokens):
            word_tokens = self.tokenizer.tokenize(token)
            if word_tokens:
                start_wp_idx = len(wordpiece_tokens)
                wordpiece_tokens.extend(word_tokens)
                end_wp_idx = len(wordpiece_tokens)
                token_to_wordpiece_map.append((start_wp_idx, end_wp_idx))
            else:
                token_to_wordpiece_map.append((None, None))
        
        # Convert to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(wordpiece_tokens)
        
        # Handle sliding windows
        effective_max_length = self.max_length - 2  # Reserve space for [CLS] and [SEP]
        
        if len(input_ids) <= effective_max_length:
            # Single chunk
            predictions = self._predict_chunk(input_ids)
        else:
            # Multiple chunks with sliding window
            predictions = self._predict_with_sliding_window(input_ids)
        
        # Convert predictions back to entities
        entities = self._extract_entities(tokens, wordpiece_tokens, predictions, token_to_wordpiece_map)
        
        return entities
    
    def _predict_chunk(self, input_ids: List[int]) -> List[str]:
        """Predict labels for a single chunk."""
        # Add special tokens
        chunk_input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        
        # Convert to tensor
        input_tensor = torch.tensor([chunk_input_ids]).to(self.device)
        attention_mask = torch.ones_like(input_tensor)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert to labels (skip [CLS] and [SEP])
        predicted_labels = [self.id_to_label[pred.item()] for pred in predictions[0][1:-1]]
        
        return predicted_labels
    
    def _predict_with_sliding_window(self, input_ids: List[int]) -> List[str]:
        """Predict labels using sliding windows and merge results."""
        effective_max_length = self.max_length - 2
        all_predictions = [None] * len(input_ids)
        prediction_counts = [0] * len(input_ids)
        
        start = 0
        while start < len(input_ids):
            end = min(start + effective_max_length, len(input_ids))
            
            # Get chunk
            chunk_input_ids = input_ids[start:end]
            chunk_predictions = self._predict_chunk(chunk_input_ids)
            
            # Merge predictions (voting for overlapping regions)
            for i, pred in enumerate(chunk_predictions):
                global_idx = start + i
                if global_idx < len(all_predictions):
                    if all_predictions[global_idx] is None:
                        all_predictions[global_idx] = {}
                    
                    if pred not in all_predictions[global_idx]:
                        all_predictions[global_idx][pred] = 0
                    all_predictions[global_idx][pred] += 1
                    prediction_counts[global_idx] += 1
            
            # Move to next chunk
            if end == len(input_ids):
                break
            start += self.stride
        
        # Resolve conflicts by majority voting
        final_predictions = []
        for pred_dict in all_predictions:
            if pred_dict is None:
                final_predictions.append('O')
            else:
                # Get the label with the most votes
                best_label = max(pred_dict.items(), key=lambda x: x[1])[0]
                final_predictions.append(best_label)
        
        return final_predictions
    
    def _extract_entities(self, tokens: List[str], wordpiece_tokens: List[str], 
                         predictions: List[str], token_to_wordpiece_map: List[tuple]) -> List[Dict]:
        """Extract entities from predictions, mapping back to original tokens."""
        entities = []
        current_entity = None
        
        for token_idx, (token, (wp_start, wp_end)) in enumerate(zip(tokens, token_to_wordpiece_map)):
            if wp_start is None or wp_end is None:
                continue
            
            # Get the label for the first wordpiece token of this word
            if wp_start < len(predictions):
                label = predictions[wp_start]
            else:
                label = 'O'
            
            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': token_idx,
                    'end': token_idx + 1
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['label']:
                # Continue current entity
                current_entity['text'] += ' ' + token
                current_entity['end'] = token_idx + 1
            else:
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_from_file(self, file_path: str) -> List[Dict]:
        """Predict entities from a text file"""
        # Use utf-8-sig to handle BOM correctly
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            text = f.read()
        return self.predict(text)
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities for a batch of texts"""
        return [self.predict(text) for text in texts]


# Legacy pipeline for backward compatibility
class SerbianLegalNERPipeline(SerbianLegalNERPipelineWithSlidingWindow):
    """Legacy pipeline that now uses sliding windows by default"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, max_length=512, stride=128)
    
    def predict(self, text: str) -> List[Dict]:
        """
        Legacy predict method - now uses sliding windows automatically.
        """
        return super().predict(text)
