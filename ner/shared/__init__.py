"""
Shared utilities for Serbian Legal NER project.

This package contains common modules used across different notebooks:
- data_processing: Data loading and preprocessing utilities
- dataset: NER dataset creation and tokenization
- model_utils: Model loading and training utilities
- evaluation: Evaluation and analysis utilities
- config: Configuration settings and constants

Note: Notebook-specific functionality (BERT-CRF, MLM pretraining, inference)
has been moved to individual notebooks to maintain clean separation.
"""

# Import key classes and functions for easy access
from .config import (
    ENTITY_TYPES, BIO_LABELS, DEFAULT_TRAINING_ARGS,
    get_default_model_config, get_model_config, get_paths, setup_environment, get_default_training_args
)

from .data_processing import (
    LabelStudioToBIOConverter,
    load_labelstudio_data,
    analyze_labelstudio_data,
    validate_bio_examples,
    load_mlm_documents
)

from .dataset import (
    NERDataset,
    split_dataset,
    tokenize_and_align_labels_with_sliding_window,
    analyze_sequence_lengths,
    print_sequence_analysis,
    create_huggingface_datasets
)

from .model_utils import (
    load_model_and_tokenizer,
    create_training_arguments,
    create_trainer,
    detailed_evaluation,
    save_model_info,
    setup_device_and_seed,
    load_inference_pipeline
)

from .evaluation import (
    create_confusion_matrices,
    analyze_misclassifications,
    analyze_entity_confusion_patterns,
    suggest_improvements_based_on_analysis,
    plot_training_history,
    plot_entity_distribution,
    generate_evaluation_report
)

__version__ = "1.0.0"
__author__ = "Serbian Legal NER Team"

__all__ = [
    # Config
    "ENTITY_TYPES", "BIO_LABELS", "DEFAULT_TRAINING_ARGS",
    "get_default_model_config", "get_model_config", "get_paths", "setup_environment", "get_default_training_args",

    # Data processing
    "LabelStudioToBIOConverter", "load_labelstudio_data",
    "analyze_labelstudio_data", "validate_bio_examples", "load_mlm_documents",

    # Dataset
    "NERDataset", "split_dataset", "tokenize_and_align_labels_with_sliding_window",
    "analyze_sequence_lengths", "print_sequence_analysis", "create_huggingface_datasets",

    # Model utilities
    "load_model_and_tokenizer", "create_training_arguments", "create_trainer",
    "detailed_evaluation", "save_model_info", "setup_device_and_seed", "load_inference_pipeline",

    # Evaluation
    "create_confusion_matrices", "analyze_misclassifications",
    "analyze_entity_confusion_patterns", "suggest_improvements_based_on_analysis",
    "plot_training_history", "plot_entity_distribution", "generate_evaluation_report"
]
