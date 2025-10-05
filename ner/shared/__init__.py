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
    load_inference_pipeline,
    PerClassMetricsCallback
)

from .evaluation import (
    # Old evaluation functions (used in base refactored notebooks)
    create_confusion_matrices,
    analyze_misclassifications,
    analyze_entity_confusion_patterns,
    suggest_improvements_based_on_analysis,
    plot_training_history,
    plot_entity_distribution,
    generate_evaluation_report,
    # Per-fold tracking functions (used in 5-fold CV notebooks)
    analyze_entity_distribution_per_fold,
    generate_detailed_classification_report,
    # Aggregate functions across all folds (used in 5-fold CV notebooks)
    aggregate_training_metrics_across_folds,
    aggregate_per_class_metrics_across_folds,
    aggregate_confusion_matrix_across_folds,
    aggregate_f1_per_class_over_iterations,
    create_aggregate_report_across_folds,
    # New visualization functions
    plot_entity_distribution_across_folds,
    plot_training_validation_loss_across_folds,
    plot_macro_micro_metrics_over_iterations
)

from .debug_training_history import inspect_training_history

__version__ = "1.0.0"
__author__ = "Serbian Legal NER Team"

__all__ = [
    # Config
    "ENTITY_TYPES", "BIO_LABELS", "DEFAULT_TRAINING_ARGS",
    "get_default_model_config", "get_model_config", "get_paths", "setup_environment",
    "get_default_training_args",

    # Data processing
    "LabelStudioToBIOConverter", "load_labelstudio_data",
    "analyze_labelstudio_data", "validate_bio_examples", "load_mlm_documents",

    # Dataset
    "NERDataset", "split_dataset", "tokenize_and_align_labels_with_sliding_window",
    "analyze_sequence_lengths", "print_sequence_analysis", "create_huggingface_datasets",

    # Model utilities
    "load_model_and_tokenizer", "create_training_arguments", "create_trainer",
    "detailed_evaluation", "save_model_info", "setup_device_and_seed", "load_inference_pipeline",
    "PerClassMetricsCallback",

    # Evaluation - Old functions (used in base refactored notebooks)
    "create_confusion_matrices", "analyze_misclassifications",
    "analyze_entity_confusion_patterns", "suggest_improvements_based_on_analysis",
    "plot_training_history", "plot_entity_distribution", "generate_evaluation_report",
    # Evaluation - Per-fold tracking (used in 5-fold CV notebooks)
    "analyze_entity_distribution_per_fold", "generate_detailed_classification_report",
    # Evaluation - Aggregate across folds (used in 5-fold CV notebooks)
    "aggregate_training_metrics_across_folds", "aggregate_per_class_metrics_across_folds",
    "aggregate_confusion_matrix_across_folds", "aggregate_f1_per_class_over_iterations",
    "create_aggregate_report_across_folds",
    # Evaluation - New visualization functions
    "plot_entity_distribution_across_folds", "plot_training_validation_loss_across_folds",
    "plot_macro_micro_metrics_over_iterations",
    # Debug utilities
    "inspect_training_history"
]
