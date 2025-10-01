"""
Shared configuration settings for Serbian Legal NER project.

This module contains common constants, paths, and configuration settings
used across different notebooks and modules.
"""

import os
from pathlib import Path
from typing import Dict, List


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Available models for NER training
AVAILABLE_MODELS = {
    "bertic": "classla/bcms-bertic",
    "xlm-r-bertic": "classla/xlm-r-bertic", 
    "xlm-roberta-base": "xlm-roberta-base",
    "bert-base-multilingual": "bert-base-multilingual-cased"
}

# Default model
DEFAULT_MODEL = "classla/bcms-bertic"

# Basic model configurations (notebook-specific configs should be in notebooks)
DEFAULT_MODEL_CONFIG = {
    "max_length": 512,
    "stride": 128,
    "learning_rate": 3e-5,
    "batch_size": 4,
    "num_epochs": 8
}


# ============================================================================
# ENTITY TYPES AND LABELS
# ============================================================================

# Serbian Legal NER entity types
ENTITY_TYPES = [
    "COURT",
    "DECISION_DATE", 
    "CASE_NUMBER",
    "CRIMINAL_ACT",
    "PROSECUTOR",
    "DEFENDANT",
    "JUDGE",
    "REGISTRAR",
    "SANCTION",
    "SANCTION_TYPE",
    "SANCTION_VALUE",
    "PROVISION",
    "PROVISION_MATERIAL",
    "PROVISION_PROCEDURAL",
    "PROCEDURE_COSTS",
    "VERDICT"
]

# BIO label mapping
def create_bio_labels(entity_types: List[str]) -> List[str]:
    """Create BIO labels from entity types"""
    labels = ["O"]  # Outside label
    for entity_type in entity_types:
        labels.extend([f"B-{entity_type}", f"I-{entity_type}"])
    return sorted(labels)

BIO_LABELS = create_bio_labels(ENTITY_TYPES)


# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Default paths (can be overridden)
DEFAULT_PATHS = {
    # Data paths
    "labelstudio_json": "/content/drive/MyDrive/NER_Master/annotations.json",
    "judgments_dir": "/content/drive/MyDrive/NER_Master/judgments",
    "labelstudio_files_dir": "/content/drive/MyDrive/NER_Master/judgments",
    
    # MLM paths
    "mlm_data_dir": "/content/drive/MyDrive/NER_Master/dapt-mlm",
    
    # Output paths
    "models_dir": "/content/drive/MyDrive/NER_Master/models",
    "logs_dir": "/content/drive/MyDrive/NER_Master/logs",
    "results_dir": "/content/drive/MyDrive/NER_Master/results"
}

# Local alternative paths
LOCAL_PATHS = {
    "labelstudio_json": "./ner/annotations.json",
    "judgments_dir": "./mlm",
    "labelstudio_files_dir": "./ner/labelstudio_files",
    "mlm_data_dir": "./mlm",
    "models_dir": "./models",
    "logs_dir": "./logs", 
    "results_dir": "./results"
}


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

# Default training arguments (basic shared settings)
DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 8,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "learning_rate": 3e-5,
    "logging_steps": 50,
    "eval_steps": 100,
    "save_steps": 500,
    "early_stopping_patience": 3,
    "max_length": 512,
    "stride": 128
}


# ============================================================================
# EVALUATION CONFIGURATIONS
# ============================================================================

# Evaluation metrics to track
EVALUATION_METRICS = [
    "precision",
    "recall",
    "f1",
    "accuracy"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_default_model_config() -> Dict:
    """Get default model configuration (notebook-specific configs should be in notebooks)"""
    return DEFAULT_MODEL_CONFIG.copy()


def get_model_config(model_name: str = None) -> Dict:
    """Get model configuration for specific model (alias for get_default_model_config)"""
    config = DEFAULT_MODEL_CONFIG.copy()
    if model_name:
        config["model_name"] = model_name
    return config


def get_paths(use_local: bool = False) -> Dict[str, str]:
    """Get file paths (local or cloud)"""
    return LOCAL_PATHS if use_local else DEFAULT_PATHS


def create_output_dir(base_dir: str, model_name: str, experiment_name: str = None) -> str:
    """Create output directory for model training"""
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
    if experiment_name:
        output_dir = f"{base_dir}/{model_short_name}_{experiment_name}"
    else:
        output_dir = f"{base_dir}/{model_short_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_paths(paths: Dict[str, str]) -> Dict[str, bool]:
    """Validate that required paths exist"""
    validation_results = {}
    
    for path_name, path_value in paths.items():
        path_obj = Path(path_value)
        
        if path_name.endswith("_dir"):
            # Directory path
            validation_results[path_name] = path_obj.exists() and path_obj.is_dir()
        else:
            # File path
            validation_results[path_name] = path_obj.exists() and path_obj.is_file()
    
    return validation_results


def setup_environment(use_local: bool = False, create_dirs: bool = True) -> Dict:
    """Setup environment and validate paths"""
    paths = get_paths(use_local)
    
    if create_dirs:
        # Create output directories if they don't exist
        for path_name, path_value in paths.items():
            if path_name in ["models_dir", "logs_dir", "results_dir"]:
                os.makedirs(path_value, exist_ok=True)
    
    validation = validate_paths(paths)
    
    print(f"🔧 Environment setup ({'local' if use_local else 'cloud'}):")
    for path_name, is_valid in validation.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {path_name}: {paths[path_name]}")
    
    return {
        "paths": paths,
        "validation": validation,
        "all_valid": all(validation.values())
    }


def get_default_training_args() -> Dict:
    """Get default training arguments (experiment-specific configs should be in notebooks)"""
    return DEFAULT_TRAINING_ARGS.copy()


# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42
