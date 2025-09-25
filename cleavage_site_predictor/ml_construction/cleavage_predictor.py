#!/usr/bin/env python3
"""
Cleavage Site Predictor
=======================

This module provides production-ready cleavage site prediction using trained ensemble models
with log-odds average probability calculations and F1-optimized soft voting strategies.

Based on the streamlined EnsembleTrainer approach but designed for inference on new data.
Uses only the best-performing combination: logodds_average + soft_vote.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .ml_models import MLModelFactory
from .data_processing import create_balanced_subdatasets
from .config import get_hyperparameter_config


class CleavagePredictor:
    """
    Production cleavage site predictor using streamlined ensemble methods.
    
    Uses log-odds average probability calculations with F1-optimized soft voting thresholds.
    Focused on the best-performing configuration only.
    """
    
    def __init__(self, random_state=42):
        """Initialize the cleavage predictor."""
        self.random_state = random_state
        self.model_factory = MLModelFactory()
        self.hyperparameter_config = get_hyperparameter_config()
        
        # Storage for trained models
        self.trained_models = {}
        self.feature_extractor = None
        self.is_fitted = False
    
    def fit(self, training_data: pd.DataFrame, hyperparameters: Dict[str, Any],
            model_type: str = 'SVM') -> None:
        """
        Fit ensemble models on training data.
        
        Args:
            training_data: Training DataFrame with sequences and labels
            hyperparameters: Dict with hyperparameters for each averaging method
            model_type: Type of models to train ('SVM', 'RandomForest', 'XGBoost')
        """
        print(f" FITTING CLEAVAGE PREDICTOR")
        print(f"   Training samples: {len(training_data)} ({(training_data['known_cleavage_site'] == 1).sum()} positive)")
        print(f"   Model type: {model_type}")
        
        # Extract features (fit on training data)
        print("   Extracting features...")
        self.feature_extractor = DataLoader(verbose=False)
        training_features = self.feature_extractor.fit_transform(training_data)
        
        # Create balanced subdatasets
        print("   Creating balanced subdatasets...")
        balanced_datasets = self._create_balanced_feature_datasets(
            training_features, training_data
        )
        print(f"   Created {len(balanced_datasets)} balanced subdatasets")
        
        # Train models using logodds averaging (best performing method)
        for avg_method in ['logodds_average']:
            if avg_method in hyperparameters:
                print(f"   Training {avg_method} ensemble...")
                
                # Handle different hyperparameter structures
                if 'best_hyperparameters' in hyperparameters[avg_method]:
                    # Structure from extract_final_results()
                    hp_params = hyperparameters[avg_method]['best_hyperparameters']['params']
                elif 'params' in hyperparameters[avg_method]:
                    # Direct params structure
                    hp_params = hyperparameters[avg_method]['params']
                else:
                    # Assume hyperparameters[avg_method] is the params dict itself
                    hp_params = hyperparameters[avg_method]
                
                trained_models = []
                
                for i, dataset in enumerate(balanced_datasets):
                    model = self.model_factory.create_model(model_type, hp_params)
                    model.fit(dataset['X'], dataset['y'])
                    trained_models.append(model)
                
                self.trained_models[avg_method] = trained_models
                print(f"      Trained {len(trained_models)} models with params: {hp_params}")
        
        self.is_fitted = True
        print(" Cleavage predictor fitted successfully!")
    
    def predict(self, query_data: pd.DataFrame,
                custom_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Predict cleavage sites on query data using logodds averaging + soft voting.
        
        Args:
            query_data: Query DataFrame with sequences
            custom_thresholds: Custom probability thresholds (default: [0.5])
            
        Returns:
            Dictionary with soft vote predictions for logodds averaging
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before prediction")
        
        print(f" PREDICTING CLEAVAGE SITES")
        print(f"   Query samples: {len(query_data)}")
        print(f"   Using logodds averaging + soft voting")
        
        # Extract features (transform only)
        query_features = self.feature_extractor.transform(query_data)
        
        # Set default thresholds
        if custom_thresholds is None:
            custom_thresholds = [0.5]
        
        print(f"   Custom thresholds: {custom_thresholds}")
        
        results = {}
        
        # Process logodds averaging method only
        if 'logodds_average' in self.trained_models:
            models = self.trained_models['logodds_average']
            print(f"   Processing logodds_average...")
            
            # Calculate ensemble probabilities
            ensemble_proba = self._calculate_ensemble_probability(
                models, query_features['X'], 'logodds_average'
            )
            
            # Initialize results for logodds averaging
            results['logodds_average'] = {}
            results['logodds_average']['soft_vote'] = {}
            
            # Soft voting with custom thresholds
            for threshold in custom_thresholds:
                pred = (ensemble_proba >= threshold).astype(int)
                
                results['logodds_average']['soft_vote'][f'threshold_{threshold:.2f}'] = {
                    'threshold': threshold,
                    'predictions': pred,
                    'probabilities': ensemble_proba.copy(),
                    'positive_count': np.sum(pred),
                    'positive_rate': np.mean(pred)
                }
            
            print(f"      logodds_average: Generated soft vote predictions")
        else:
            raise ValueError("No logodds_average models found. Please fit the predictor first.")
        
        # Add metadata
        results['metadata'] = {
            'n_samples': len(query_data),
            'strategy': 'soft_vote',
            'averaging_method': 'logodds_average',
            'custom_thresholds': custom_thresholds,
            'query_data': query_data.copy()
        }
        
        print(" Prediction completed!")
        return results
    
    def _create_balanced_feature_datasets(self, features: Dict[str, Any], 
                                         data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create balanced feature datasets for ensemble training."""
        # Use existing balanced dataset creation function
        balanced_segments = create_balanced_subdatasets(
            data, 
            pos_neg_ratio=1.0,  # 1:1 ratio for balanced datasets
            random_state=self.random_state
        )
        
        # Convert to feature format
        feature_datasets = []
        for segment_df in balanced_segments:
            # Get indices of samples in this segment
            segment_indices = segment_df.index.tolist()
            
            # Extract corresponding features
            X_segment = features['X'][segment_indices]
            y_segment = features['y'][segment_indices]
            
            feature_datasets.append({'X': X_segment, 'y': y_segment})
        
        return feature_datasets
    
    def _calculate_ensemble_probability(self, models: List, X: np.ndarray, 
                                      method: str) -> np.ndarray:
        """Calculate ensemble probabilities using specified averaging method."""
        # Get individual model probabilities
        all_probabilities = []
        for model in models:
            proba = model.predict_proba(X)[:, 1]  # Get positive class probability
            all_probabilities.append(proba)
        
        all_probabilities = np.array(all_probabilities)  # Shape: (n_models, n_samples)
        
        if method == 'logodds_average':
            # Log-odds average: logit(p̂) = (1/n)∑logit(pi), p̂ = σ(logit(p̂))
            # Convert probabilities to log-odds (logits)
            epsilon = 1e-15  # Small value to avoid log(0)
            clipped_proba = np.clip(all_probabilities, epsilon, 1 - epsilon)
            logits = np.log(clipped_proba / (1 - clipped_proba))
            
            # Average log-odds
            avg_logits = np.mean(logits, axis=0)
            
            # Convert back to probabilities using sigmoid
            ensemble_proba = 1 / (1 + np.exp(-avg_logits))
            
        else:
            raise ValueError(f"Unknown averaging method: {method}")
        
        return ensemble_proba
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            specificity = 0.0
        
        # Calculate PS1 score
        def calculate_ps_score(precision, specificity, beta=1.0):
            if precision == 0 and specificity == 0:
                return 0.0
            numerator = (1 + beta**2) * precision * specificity
            denominator = (beta**2 * precision) + specificity
            return numerator / denominator if denominator > 0 else 0.0
        
        ps1 = calculate_ps_score(precision, specificity, beta=1.0)
        
        metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision,
            'specificity': specificity,
            'ps1': ps1,
            'recall': recall,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }
        
        # AUC calculation
        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.5
        else:
            metrics['auc'] = 0.5
        
        return metrics
