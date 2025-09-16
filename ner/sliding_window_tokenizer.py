"""
Sliding Window Tokenization for Serbian Legal NER

This module implements a 3-step tokenization process for handling long sequences:
1. Whitespace-based BIO tagging (already done in examples)
2. Convert to WordPiece BIO with proper label alignment
3. Sliding window chunking with overlap to handle long sequences
"""

def tokenize_and_align_labels_with_sliding_window(
    examples, tokenizer, label_to_id, max_length=512, stride=128
):
    """
    Tokenize text and align labels with subword tokens using sliding windows.

    This function implements a 3-step process:
    1. Whitespace-based BIO tagging (already done in examples)
    2. Convert to WordPiece BIO with proper label alignment
    3. Sliding window chunking with overlap to handle long sequences

    Args:
        examples: List of examples with 'tokens' and 'labels'
        tokenizer: HuggingFace tokenizer
        label_to_id: Mapping from labels to IDs
        max_length: Maximum sequence length (default: 512)
        stride: Overlap between windows (default: 128)

    Returns:
        List of tokenized chunks with input_ids, attention_mask, and labels
    """
    tokenized_inputs = []
    total_chunks = 0
    long_sequences = 0

    # Create reverse mapping for converting integer labels back to strings
    id_to_label = {v: k for k, v in label_to_id.items()}

    for example_idx, example in enumerate(examples):
        tokens = example['tokens']
        labels = example['labels']

        # Convert integer labels back to strings if needed
        if labels and isinstance(labels[0], int):
            labels = [id_to_label[label_id] for label_id in labels]

        # Step 2: Convert to WordPiece BIO
        # Tokenize each word and track alignment
        wordpiece_tokens = []
        wordpiece_labels = []

        for token, label in zip(tokens, labels):
            # Tokenize the word using WordPiece
            word_tokens = tokenizer.tokenize(token)

            if not word_tokens:  # Skip empty tokenizations
                continue

            wordpiece_tokens.extend(word_tokens)

            # Align labels: first subword inherits original label
            # Remaining subwords get I-<ENTITY> if inside entity, O if outside
            if label == 'O':
                # Outside entity: all subwords get O
                wordpiece_labels.extend(['O'] * len(word_tokens))
            elif label.startswith('B-'):
                # Beginning of entity: first subword gets B-, rest get I-
                entity_type = label[2:]  # Remove 'B-' prefix
                wordpiece_labels.append(f'B-{entity_type}')
                wordpiece_labels.extend([f'I-{entity_type}'] * (len(word_tokens) - 1))
            elif label.startswith('I-'):
                # Inside entity: all subwords get I-
                entity_type = label[2:]  # Remove 'I-' prefix
                wordpiece_labels.extend([f'I-{entity_type}'] * len(word_tokens))
            else:
                # Unknown label: treat as O
                wordpiece_labels.extend(['O'] * len(word_tokens))

        # Convert labels to IDs
        wordpiece_label_ids = [
            label_to_id.get(label, label_to_id['O']) for label in wordpiece_labels
        ]

        # Convert tokens to input IDs
        input_ids = tokenizer.convert_tokens_to_ids(wordpiece_tokens)

        # Step 3: Sliding window chunking
        # Reserve space for special tokens [CLS] and [SEP]
        effective_max_length = max_length - 2
        if len(input_ids) <= effective_max_length:
            # Sequence fits in one chunk
            chunks = [{
                'input_ids': input_ids,
                'labels': wordpiece_label_ids,
                'start_idx': 0,
                'end_idx': len(input_ids)
            }]
        else:
            # Create overlapping chunks
            long_sequences += 1
            chunks = []
            start = 0
            
            while start < len(input_ids):
                end = min(start + effective_max_length, len(input_ids))
                
                chunk_input_ids = input_ids[start:end]
                chunk_labels = wordpiece_label_ids[start:end]
                
                chunks.append({
                    'input_ids': chunk_input_ids,
                    'labels': chunk_labels,
                    'start_idx': start,
                    'end_idx': end
                })
                
                # Move to next chunk with stride
                if end == len(input_ids):
                    break  # Last chunk
                start += stride
        
        total_chunks += len(chunks)
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_input_ids = chunk['input_ids']
            chunk_labels = chunk['labels']
            
            # Add special tokens
            final_input_ids = [tokenizer.cls_token_id] + chunk_input_ids + [tokenizer.sep_token_id]
            final_labels = [-100] + chunk_labels + [-100]
            
            # Create attention mask
            attention_mask = [1] * len(final_input_ids)
            
            # Pad to max_length
            padding_length = max_length - len(final_input_ids)
            final_input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            final_labels.extend([-100] * padding_length)
            
            tokenized_inputs.append({
                'input_ids': final_input_ids,
                'attention_mask': attention_mask,
                'labels': final_labels,
                'example_idx': example_idx,
                'chunk_idx': chunk_idx,
                'start_idx': chunk['start_idx'],
                'end_idx': chunk['end_idx']
            })
    
    print(f"Created {total_chunks} chunks from {len(examples)} examples")
    print(f"Long sequences requiring chunking: {long_sequences}")
    
    return tokenized_inputs


def analyze_sequence_lengths(examples, tokenizer):
    """
    Analyze the distribution of sequence lengths after WordPiece tokenization.
    
    Args:
        examples: List of examples with 'tokens' and 'labels'
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Dictionary with length statistics
    """
    lengths = []
    
    for example in examples:
        tokens = example['tokens']
        
        # Tokenize each word
        wordpiece_tokens = []
        for token in tokens:
            word_tokens = tokenizer.tokenize(token)
            if word_tokens:
                wordpiece_tokens.extend(word_tokens)
        
        # Add 2 for [CLS] and [SEP] tokens
        total_length = len(wordpiece_tokens) + 2
        lengths.append(total_length)
    
    lengths.sort()
    
    stats = {
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': sum(lengths) / len(lengths),
        'median_length': lengths[len(lengths) // 2],
        'sequences_over_512': sum(1 for l in lengths if l > 512),
        'sequences_over_256': sum(1 for l in lengths if l > 256),
        'total_sequences': len(lengths)
    }
    
    return stats, lengths


def print_sequence_analysis(examples, tokenizer):
    """
    Print detailed analysis of sequence lengths.
    """
    stats, lengths = analyze_sequence_lengths(examples, tokenizer)
    
    print("=== Sequence Length Analysis ===")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Min length: {stats['min_length']}")
    print(f"Max length: {stats['max_length']}")
    print(f"Mean length: {stats['mean_length']:.1f}")
    print(f"Median length: {stats['median_length']}")
    print(f"Sequences > 512 tokens: {stats['sequences_over_512']} ({stats['sequences_over_512']/stats['total_sequences']*100:.1f}%)")
    print(f"Sequences > 256 tokens: {stats['sequences_over_256']} ({stats['sequences_over_256']/stats['total_sequences']*100:.1f}%)")
    
    # Show percentiles
    percentiles = [50, 75, 90, 95, 99]
    print("\nLength percentiles:")
    for p in percentiles:
        idx = int(len(lengths) * p / 100)
        print(f"  {p}th percentile: {lengths[idx]} tokens")
    
    return stats, lengths


# Legacy function for backward compatibility
def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=512):
    """
    Legacy tokenization function - now calls the sliding window version.
    Kept for backward compatibility.
    """
    return tokenize_and_align_labels_with_sliding_window(
        examples, tokenizer, label_to_id, max_length=max_length, stride=128
    )
