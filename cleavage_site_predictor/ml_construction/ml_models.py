#!/usr/bin/env python3
"""
ML Models Factory for Leak-Safe ML Pipeline
=================================================

This module provides model creation and hyperparameter tuning functionality
integrated with the centralized configuration system.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb

from .config_manager import get_config_manager


class MLModelFactory:
    """
    Factory class for creating and configuring ML models with hyperparameter tuning.
    
    This class provides a unified interface for model creation, hyperparameter
    tuning, and best model selection using configuration from config.ini.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize model factory.
        
        Args:
            config_manager: ConfigManager instance. If None, uses global instance.
        """
        self.config_manager = config_manager or get_config_manager()
        self.hp_config = self.config_manager.get_hyperparameter_config()
        self.global_config = self.config_manager.get_global_config()
        
        print(" ML Model Factory initialized")
        print(f"   Scoring metric: {self.hp_config['scoring_metric']}")
        print(f"   Parallel jobs: {self.hp_config['n_jobs']}")
    
    def create_model(self, model_type: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a model instance with specified hyperparameters.
        
        Args:
            model_type: Type of model ('SVM', 'RandomForest', 'XGBoost')
            hyperparameters: Dictionary of hyperparameters. If None, uses defaults from config.
            
        Returns:
            Configured model instance
        """
        model_type_lower = model_type.lower()
        
        if hyperparameters is None:
            # Get default hyperparameters from config
            model_config = self.config_manager.get_model_config(model_type_lower)
            hyperparameters = {k: v for k, v in model_config.items() 
                             if k not in ['model_type', 'optimized_score']}
        
        if model_type_lower == 'svm':
            return self._create_svm(hyperparameters)
        elif model_type_lower in ['rf', 'random_forest', 'randomforest']:
            return self._create_random_forest(hyperparameters)
        elif model_type_lower in ['xgb', 'xgboost']:
            return self._create_xgboost(hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_svm(self, hyperparameters: Dict[str, Any]) -> SVC:
        """Create SVM model with specified hyperparameters."""
        # Handle class_weight parameter
        if 'class_weight' in hyperparameters and hyperparameters['class_weight'] == 'balanced':
            hyperparameters['class_weight'] = 'balanced'
        
        return SVC(
            probability=True,  # Enable probability prediction
            random_state=self.global_config['random_state'],
            **hyperparameters
        )
    
    def _create_random_forest(self, hyperparameters: Dict[str, Any]) -> RandomForestClassifier:
        """Create Random Forest model with specified hyperparameters."""
        return RandomForestClassifier(
            random_state=self.global_config['random_state'],
            n_jobs=self.hp_config.get('n_jobs', -1),
            **hyperparameters
        )
    
    def _create_xgboost(self, hyperparameters: Dict[str, Any]) -> xgb.XGBClassifier:
        """Create XGBoost model with specified hyperparameters."""
        params = hyperparameters.copy()
        
        # Handle scale_pos_weight for class imbalance
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = 1
        
        return xgb.XGBClassifier(
            random_state=self.global_config['random_state'],
            n_jobs=self.hp_config.get('n_jobs', -1),
            eval_metric='logloss',
            **params
        )
    
    def tune_hyperparameters(self, 
                           model_type: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           scoring_metric: str = None) -> Tuple[Dict[str, Any], float]:
        """
        Test all hyperparameter combinations on single train/val split.
        
        This method is called for each inner CV fold with pre-extracted features.
        It tests all parameter combinations and returns the best one for this fold.
        
        Args:
            model_type: Type of model to tune
            X_train: Training features (already extracted)
            y_train: Training labels
            X_val: Validation features (already extracted)
            y_val: Validation labels
            scoring_metric: Scoring metric to use (default: from config)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"        Testing hyperparameters for {model_type} on fold")
        
        model_type_lower = model_type.lower()
        
        # Get hyperparameter grid
        param_grid = self._get_param_grid(model_type_lower)
        
        # Setup scoring metric
        if scoring_metric is None:
            scoring_metric = self.hp_config['scoring_metric']
        
        # Grid search over parameter combinations (no additional CV)
        from sklearn.model_selection import ParameterGrid
        param_combinations = list(ParameterGrid(param_grid))
        
        print(f"          Testing {len(param_combinations)} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        
        for param_idx, params in enumerate(param_combinations):
            # Train model with current parameters
            model = self.create_model(model_type, params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            if scoring_metric == 'roc_auc':
                val_probas = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_probas)
            elif scoring_metric == 'f1':
                val_preds = model.predict(X_val)
                score = f1_score(y_val, val_preds)
            else:
                # Use default scoring
                val_preds = model.predict(X_val)
                score = f1_score(y_val, val_preds)  # Default to F1
            
            # Update best parameters if this is better
            if score > best_score:
                best_score = score
                best_params = params
        
        print(f"          Best params for this fold: {best_params} (score: {best_score:.4f})")
        
        return best_params, best_score
    
    def select_best_params_from_folds(self, fold_results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """
        Select best hyperparameters from multiple inner CV folds.
        
        Args:
            fold_results: List of (best_params, best_score) from each fold
            
        Returns:
            Best parameters across all folds
        """
        if not fold_results:
            raise ValueError("No fold results provided")
        
        # Find the fold with the best score
        best_fold_idx = np.argmax([score for params, score in fold_results])
        best_params, best_score = fold_results[best_fold_idx]
        
        print(f"      Selected hyperparameters from fold {best_fold_idx + 1} (score: {best_score:.4f})")
        print(f"      Final params: {best_params}")
        
        return best_params
    
    def train_final_model(self, 
                         model_type: str,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         best_params: Dict[str, Any]) -> Any:
        """
        Train final model with best hyperparameters.
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training labels
            best_params: Best hyperparameters
            
        Returns:
            Trained model
        """
        model = self.create_model(model_type, best_params)
        model.fit(X_train, y_train)
        return model
    
    def _get_param_grid(self, model_type: str) -> Dict[str, List]:
        """
        Get hyperparameter grid for the specified model type.
        
        Args:
            model_type: Type of model ('svm', 'random_forest', 'xgboost')
            
        Returns:
            Dictionary with parameter grid
        """
        if model_type == 'svm':
            return {
                'C': self.hp_config['svm_grid']['c'],  # Note: configparser converts to lowercase
                'kernel': self.hp_config['svm_grid']['kernel'], 
                'gamma': self.hp_config['svm_grid']['gamma']
            }
        elif model_type in ['rf', 'random_forest', 'randomforest']:
            return {
                'n_estimators': [int(x) for x in self.hp_config['rf_grid']['n_estimators']],
                'max_depth': [int(x) if x != 'None' else None for x in self.hp_config['rf_grid']['max_depth']],
                'min_samples_split': [int(x) for x in self.hp_config['rf_grid']['min_samples_split']],
                'min_samples_leaf': [int(x) for x in self.hp_config['rf_grid']['min_samples_leaf']]
            }
        elif model_type == 'xgb' or model_type == 'xgboost':
            return {
                'n_estimators': [int(x) for x in self.hp_config['xgb_grid']['n_estimators']],
                'max_depth': [int(x) for x in self.hp_config['xgb_grid']['max_depth']],
                'learning_rate': [float(x) for x in self.hp_config['xgb_grid']['learning_rate']],
                'subsample': [float(x) for x in self.hp_config['xgb_grid']['subsample']],
                'colsample_bytree': [float(x) for x in self.hp_config['xgb_grid']['colsample_bytree']]
            }
        else:
            raise ValueError(f"Unknown model type for param grid: {model_type}")
    
    def evaluate_model(self, 
                      model: Any,
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score,
            precision_score, recall_score, matthews_corrcoef,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
            # Convert to probabilities using sigmoid
            y_proba = 1 / (1 + np.exp(-y_proba))
        else:
            y_proba = y_pred.astype(float)
        
        # Calculate precision and specificity for PS-score
        precision = precision_score(y_test, y_pred, zero_division=0)
        
        # Specificity from confusion matrix
        if len(np.unique(y_pred)) > 1 and len(np.unique(y_test)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        # Calculate PS1 score
        def calculate_ps_score(precision, specificity, beta=1.0):
            if precision == 0 and specificity == 0:
                return 0.0
            numerator = (1 + beta**2) * precision * specificity
            denominator = (beta**2 * precision) + specificity
            return numerator / denominator if denominator > 0 else 0.0
        
        ps1 = calculate_ps_score(precision, specificity, beta=1.0)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'precision': precision,
            'specificity': specificity,
            'ps1': ps1,
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
        }
        
        # AUC and average precision (only if we have both classes)
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_proba)
        else:
            metrics['auc'] = 0.0
            metrics['average_precision'] = 0.0
        
        return metrics

    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model type strings
        """
        return ['SVM', 'RandomForest', 'XGBoost']

