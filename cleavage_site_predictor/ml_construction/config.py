#!/usr/bin/env python3
"""
ML Construction Configuration
=============================

Python-based configuration for ML pipeline allowing automatic updates.
"""

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================
FEATURE_CONFIG = {
    # Sequence-based features
    'include_sequence_features': True,
    
    # Structure-based features  
    'include_structure_features': True,
    
    # Protein Language Model embeddings
    'include_plm_embeddings': True,
    'embedding_level': 'residue',
    
    # Node feature pooling configuration
    'pooling_method': 'mean',
    'decay_strength': 1.0,
    
    # Physicochemical features
    'include_weighted_metrix': True,
    
    # Composition features
    'include_cksaap': True,
    'include_cpp': True,
    'cpp_n_filter': 100,  # Number of CPP features to extract (default: 100)
    # FIMO
    'use_fimo': False,
    
    # Coevolution patterns
    'include_coevolution_patterns': True,
    
    # Data format and normalization
    'return_fusion_format': True,
    'normalization': 'zscore'
}

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
DATA_PROCESSING_CONFIG = {
    # Memory and performance settings
    'extract_node_features': True,
    'batch_processing': False,
    'cache_features': True,
    
    # Data validation
    'validate_features': True,
    'remove_invalid_samples': True,
    'handle_missing_values': True,
    
    # Feature scaling and normalization
    'fit_on_train_only': True,
    'scale_per_segment': True,
    'preserve_feature_names': True
}

# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================
CV_CONFIG = {
    # Model types for pipeline (list)
    'model_types': ['SVM'],  # Available: 'SVM', 'RandomForest', 'XGBoost'
    
    # Streamlined: Only soft_vote with F1-optimized thresholds
    'ensemble_strategies': ['soft_vote'],
    
    # Data balancing
    'pos_neg_ratio': 1.0
}

# =============================================================================
# THRESHOLD OPTIMIZATION CONFIGURATION (for soft_vote)
# =============================================================================
THRESHOLD_CONFIG = {
    # Threshold range for F1-score optimization (exclude extreme values)
    'threshold_range': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                       0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    'optimization_metric': 'f1'
}


# =============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# =============================================================================
HYPERPARAMETER_CONFIG = {
    # Scoring metric for hyperparameter tuning: roc_auc, f1, precision, recall
    'scoring_metric': 'roc_auc',
    
    # Number of parallel jobs
    'n_jobs': -1,
    
    # SVM hyperparameter grid
    'svm_grid': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    
    # Random Forest hyperparameter grid 
    'rf_grid': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    # XGBoost hyperparameter grid
    'xgb_grid': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# =============================================================================
# MODEL DEFAULT PARAMETERS
# =============================================================================
SVM_DEFAULTS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced'
}

RF_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced'
}

XGB_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'scale_pos_weight': 1
}

MODEL_DEFAULTS = {
    'svm': SVM_DEFAULTS,
    'random_forest': RF_DEFAULTS,
    'xgboost': XGB_DEFAULTS
}

# =============================================================================
# CONFIGURATION ACCESS FUNCTIONS
# =============================================================================

def get_feature_config():
    """Get feature extraction configuration."""
    return FEATURE_CONFIG.copy()

def get_data_processing_config():
    """Get data processing configuration.""" 
    return DATA_PROCESSING_CONFIG.copy()

def get_cv_config():
    """Get cross-validation configuration."""
    config = CV_CONFIG.copy()
    config['random_state'] = RANDOM_STATE
    return config

def get_threshold_config():
    """Get threshold optimization configuration."""
    return THRESHOLD_CONFIG.copy()


def get_hyperparameter_config():
    """Get hyperparameter tuning configuration."""
    return HYPERPARAMETER_CONFIG.copy()

def get_model_defaults(model_name):
    """Get default parameters for a specific model."""
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_DEFAULTS:
        raise ValueError(f"Model {model_name} not found in defaults")
    return MODEL_DEFAULTS[model_name_lower].copy()
