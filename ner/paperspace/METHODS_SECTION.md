# Section 5: Methodology

## 5.1 Model Architecture and Configuration

The experimental framework employed the BCSm-BERTić model (classla/bcms-bertic), a domain-specific transformer architecture pre-trained on Bosnian, Croatian, Montenegrin, and Serbian corpora. This model represents an ELECTRA-based architecture optimized for South Slavic languages, providing superior linguistic representations for Serbian legal text compared to general-purpose multilingual models. The model architecture comprises 110,049,053 trainable parameters, utilizing a discriminative pre-training approach that distinguishes replaced tokens from original tokens, thereby achieving more efficient learning compared to traditional masked language modeling approaches.

The model was configured for token classification with 29 distinct BIO-tagged labels, encompassing 14 legal entity types specific to Serbian criminal court judgments. The BIO (Begin-Inside-Outside) tagging scheme was employed to precisely delineate entity boundaries, wherein each entity type receives two labels (B-ENTITY for the beginning token and I-ENTITY for continuation tokens) in addition to the O label for non-entity tokens. This labeling strategy enables the model to accurately identify multi-token entities while maintaining clear boundaries between adjacent entities of the same type.

## 5.2 Sequence Processing and Sliding Window Tokenization

Serbian legal judgments present substantial challenges for sequence processing due to their considerable length, frequently exceeding the maximum input length constraints of transformer models. To address this limitation without information loss through truncation, a sliding window tokenization strategy was implemented. This approach segments lengthy documents into overlapping subsequences while preserving contextual information across segment boundaries.

The tokenization process employed a maximum sequence length of 512 tokens, consistent with the architectural constraints of BERT-family models, while incorporating special tokens ([CLS] and [SEP]) that demarcate sequence boundaries. To maintain contextual continuity across windows, a stride parameter of 128 tokens was configured, resulting in a 128-token overlap between consecutive windows. This overlap ensures that entities appearing near window boundaries are captured with sufficient context in at least one window, mitigating the risk of fragmented entity recognition.

The implementation of sliding window tokenization proceeded through several systematic stages. Initially, each word in the input sequence was tokenized using the model's WordPiece tokenizer, which decomposes words into subword units present in the model's vocabulary. Subsequently, labels were aligned with the resulting subword tokens, wherein the first subword of each word inherited the original BIO label while subsequent subwords received a special ignore index (-100) to prevent their inclusion in loss computation. This alignment strategy ensures that the model learns to predict entity labels at the word level rather than the subword level, maintaining consistency with the annotation schema.

For sequences exceeding the effective maximum length (510 tokens after accounting for special tokens), the algorithm generated multiple overlapping chunks. Each chunk was constructed by extracting a contiguous subsequence of input tokens and their corresponding labels, prepending the [CLS] token and appending the [SEP] token, and padding the sequence to the maximum length with padding tokens and ignore labels. The sliding window advanced by the stride value until the entire sequence was processed, ensuring complete coverage of the input document.

This sliding window approach offers several methodological advantages. It eliminates information loss inherent in truncation-based methods, preserves local context around entities through overlapping regions, enables processing of arbitrarily long documents within fixed memory constraints, and maintains computational efficiency by processing chunks independently during training and inference.

## 5.3 Cross-Validation Strategy

To ensure robust evaluation and mitigate the risk of overfitting to specific data partitions, a stratified 5-fold cross-validation methodology was employed. This approach partitions the dataset into five disjoint subsets of approximately equal size, iteratively training five independent models wherein each fold serves once as the validation set while the remaining four folds constitute the training set. The cross-validation procedure was implemented using scikit-learn's KFold class with shuffle enabled and a fixed random seed (42) to ensure reproducibility across experimental runs.

The dataset comprised 225 annotated legal judgments, resulting in approximately 45 documents per validation fold and 180 documents per training fold. This partition size provides sufficient training data for model convergence while maintaining adequate validation samples for reliable performance estimation. For each fold, the training and validation sets were analyzed to verify balanced entity distribution across folds, ensuring that no fold exhibited systematic bias in entity type representation that could compromise generalization assessment.

The cross-validation process generated fold-specific entity distribution statistics, revealing consistent proportions across folds. Material provisions (PROVISION_MATERIAL) constituted approximately 35-38% of all entities, procedural provisions (PROVISION_PROCEDURAL) represented 15-16%, and criminal acts (CRIMINAL_ACT) accounted for 10-11% of entities across both training and validation partitions. This distributional consistency across folds validates the effectiveness of the random shuffling strategy and ensures that performance metrics reflect genuine model capabilities rather than artifacts of data partitioning.

## 5.4 Training Configuration and Optimization

The training procedure employed the AdamW optimizer, an adaptive learning rate method with decoupled weight decay regularization that has demonstrated superior performance for transformer fine-tuning tasks. The base learning rate was set to 3×10⁻⁵, a value empirically validated for BERT-family models in sequence labeling tasks. To facilitate stable training initialization, a linear warmup schedule was implemented over the first 500 training steps, gradually increasing the learning rate from zero to the target value. This warmup period prevents destabilization during early training when model parameters are far from optimal configurations.

Weight decay regularization was applied with a coefficient of 0.01 to all parameters except bias terms and layer normalization parameters, following established best practices for transformer fine-tuning. This regularization strategy mitigates overfitting by penalizing large parameter values while preserving the model's capacity to learn complex patterns in the training data.

The training process spanned 8 epochs with a per-device batch size of 8 examples, resulting in effective batch sizes that varied based on the number of tokenized chunks generated by the sliding window procedure. Each training fold produced approximately 1,845 training examples and 500 validation examples after sliding window tokenization, reflecting the expansion of the original document count due to sequence segmentation.

## 5.5 Model Evaluation and Early Stopping

Model performance was continuously monitored throughout training using a comprehensive evaluation protocol. Validation metrics were computed every 100 training steps, providing fine-grained tracking of model convergence and enabling early detection of overfitting. The evaluation metrics included precision, recall, F1-score, and accuracy, computed using the seqeval library, which correctly handles BIO-tagged sequences by evaluating entity-level predictions rather than token-level predictions.

An early stopping mechanism was implemented with a patience parameter of 3 evaluation steps, monitoring the validation F1-score as the primary metric. If the F1-score failed to improve for three consecutive evaluation steps, training was terminated to prevent overfitting and reduce computational costs. The model checkpoint achieving the highest validation F1-score was retained for final evaluation, ensuring that reported performance metrics reflect the optimal model state rather than the final training iteration.

The evaluation procedure employed strict entity-level matching, wherein a predicted entity was considered correct only if both its span (start and end positions) and its type label exactly matched the gold standard annotation. This stringent criterion ensures that reported metrics accurately reflect the model's practical utility for downstream applications, as partially correct predictions (e.g., correct type but incorrect boundaries) are not credited.

## 5.6 Data Collation and Batch Processing

Training and evaluation batches were constructed using the DataCollatorForTokenClassification from the Transformers library, which implements dynamic padding to optimize computational efficiency. Rather than padding all sequences to the maximum length during tokenization, this collator pads sequences to the length of the longest sequence in each batch, reducing unnecessary computation on padding tokens. The collator also ensures proper handling of label padding, assigning the ignore index (-100) to padding positions to exclude them from loss computation.

This dynamic padding strategy offers computational advantages particularly relevant for datasets with variable sequence lengths. By minimizing the number of padding tokens processed during forward and backward passes, the approach reduces both memory consumption and training time while maintaining numerical equivalence to static padding approaches.

## 5.7 Computational Infrastructure and Reproducibility

All experiments were conducted on NVIDIA Quadro RTX 5000 GPUs with CUDA acceleration enabled, providing the computational resources necessary for efficient transformer training. The PyTorch deep learning framework (version 2.1.1) served as the underlying computational engine, with the Transformers library (version 4.35.2) providing high-level abstractions for model loading, training, and evaluation.

To ensure reproducibility of experimental results, all random number generators were seeded with a fixed value (42) at the initialization of each training run. This seeding strategy encompasses PyTorch's random number generator, NumPy's random state, and Python's built-in random module, ensuring deterministic behavior across all stochastic operations including data shuffling, dropout, and parameter initialization.

The training process generated comprehensive logging outputs, including per-step training loss, per-evaluation validation metrics, and detailed classification reports for each fold. These logs were preserved for subsequent analysis and aggregation across folds, enabling computation of mean performance metrics and standard deviations that characterize model stability and generalization capability.

## 5.8 Per-Fold Training Procedure

For each cross-validation fold, a fresh model instance was initialized from the pre-trained BCSm-BERTić checkpoint, ensuring that no information leaked between folds through parameter sharing. The fold-specific training procedure comprised several sequential stages. First, the training and validation examples for the current fold were extracted based on the cross-validation indices. Second, entity distribution analysis was performed to verify balanced representation across entity types. Third, sliding window tokenization was applied to both training and validation sets, generating the expanded set of training examples. Fourth, a fold-specific output directory was created to store model checkpoints, training logs, and evaluation results. Fifth, the model was instantiated and moved to the GPU device. Sixth, training arguments and the Trainer object were configured with fold-specific parameters. Seventh, the training loop was executed with continuous validation monitoring and early stopping. Finally, the best model checkpoint was loaded and evaluated on the validation set, generating comprehensive performance metrics.

This systematic procedure was repeated independently for each of the five folds, producing five trained models and five sets of validation metrics. The independence of fold training ensures that performance estimates are unbiased and that the reported metrics accurately reflect expected performance on unseen data.

## 5.9 Metrics Aggregation and Statistical Analysis

Following the completion of all fold training procedures, validation metrics were aggregated across folds to produce summary statistics characterizing overall model performance. For each evaluation metric (precision, recall, F1-score, and accuracy), the mean and standard deviation across the five folds were computed. These aggregate statistics provide a comprehensive characterization of model performance, wherein the mean reflects expected performance on unseen data and the standard deviation quantifies performance variability attributable to data partitioning.

Additionally, per-entity-type metrics were aggregated to identify systematic strengths and weaknesses in the model's entity recognition capabilities. This fine-grained analysis enables identification of entity types that pose particular challenges, informing potential directions for model improvement such as data augmentation for underrepresented entities or architectural modifications to better capture entity-specific patterns.

The aggregation procedure employed the seqeval library's classification report functionality, which generates precision, recall, and F1-score for each entity type based on strict entity-level matching. These per-entity metrics were collected across all folds and summarized using descriptive statistics, providing insights into the consistency of entity recognition performance across different data partitions.

## 5.10 Label Alignment and Loss Computation

The label alignment strategy employed during tokenization plays a critical role in ensuring correct model training. When words are decomposed into multiple subword tokens by the WordPiece tokenizer, only the first subword receives the original BIO label, while subsequent subwords are assigned the ignore index (-100). This approach ensures that the model learns to predict entity labels based on the first subword of each word, which typically carries the most semantic information.

During loss computation, the cross-entropy loss function automatically excludes positions with the ignore index, preventing these positions from contributing to the gradient updates. This exclusion is essential because subword tokens do not have independent semantic meaning in the context of entity recognition, and forcing the model to predict labels for these positions would introduce noise into the training signal.

Special tokens ([CLS], [SEP], and [PAD]) are similarly assigned the ignore index, ensuring that the model focuses exclusively on content tokens during training. This strategy aligns with the evaluation protocol, which considers only content tokens when computing entity-level metrics, maintaining consistency between training objectives and evaluation criteria.

## 5.11 Memory Management and Computational Efficiency

To manage GPU memory constraints during training, several optimization strategies were employed. First, gradient accumulation was implicitly handled by the Trainer's batch processing logic, enabling effective batch sizes larger than the per-device batch size when necessary. Second, mixed-precision training could be enabled through the Trainer's fp16 parameter, reducing memory consumption and accelerating computation on compatible hardware. Third, model checkpoints were saved periodically (every 500 steps) and only the best checkpoint was retained, preventing excessive disk usage.

Between fold training runs, explicit memory cleanup was performed by deleting model and trainer objects and invoking PyTorch's cache clearing function (torch.cuda.empty_cache()). This cleanup ensures that GPU memory is fully released before initializing the next fold's model, preventing out-of-memory errors that could arise from memory fragmentation or incomplete garbage collection.

The sliding window tokenization strategy itself contributes to memory efficiency by enabling batch processing of fixed-size chunks regardless of original document length. This uniformity in chunk size ensures predictable memory consumption during training and prevents memory spikes that could occur with variable-length sequence processing.

## 5.12 Validation and Quality Assurance

Prior to model training, comprehensive data validation procedures were executed to ensure data quality and consistency. The BIO conversion process was validated by checking for illegal label sequences (e.g., I-ENTITY tags without preceding B-ENTITY tags), verifying alignment between text spans and entity annotations, and confirming that all entity types in the annotations correspond to the predefined entity type schema.

Sequence length analysis was performed on both training and validation sets to characterize the distribution of document lengths and validate the appropriateness of the sliding window parameters. This analysis confirmed that the majority of documents exceeded the 512-token limit, justifying the necessity of the sliding window approach and informing the selection of the stride parameter to balance context preservation and computational efficiency.

Entity distribution analysis was conducted at multiple levels: across the entire dataset to characterize overall entity frequency, within each cross-validation fold to verify balanced partitioning, and within training versus validation sets to detect potential distribution shifts. These analyses confirmed that the dataset exhibits natural class imbalance (with provision entities substantially more frequent than other types) but that this imbalance is consistent across folds, ensuring that validation metrics accurately reflect real-world performance expectations.

