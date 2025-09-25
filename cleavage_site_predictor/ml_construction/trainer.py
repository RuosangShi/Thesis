#!/usr/bin/env python3
"""
Streamlined ML Training Pipeline - Logodds + Soft Vote + F1 Optimization
========================================================================

This module implements a focused approach to ensemble learning:

1. Single Probability Averaging Method:
   - Log-odds Average: logit(p̂) = (1/n)∑logit(pi), p̂ = σ(logit(p̂))

2. Single Voting Method:
   - Soft Vote: Prediction based on F1-optimized probability thresholds

3. Hyperparameter Tuning:
   - Tune hyperparameters for logodds averaging method
   - Optimize thresholds using F1-score

4. Clean Pipeline:
   - Single method combination: logodds_average + soft_vote
   - F1-score based threshold optimization
   - Focused on best performing configuration only
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .ml_models import MLModelFactory
from .data_processing import create_balanced_subdatasets


class EnsembleTrainer:
    """
    Streamlined ensemble training focused on logodds averaging + soft voting
    with F1-score based threshold optimization.
    """
    
    def __init__(self, random_state=42):
        """Initialize the trainer."""
        from .config import (
            get_cv_config, get_threshold_config, get_voting_config, 
            get_feature_config, get_global_config, get_hyperparameter_config
        )
        
        self.cv_config = get_cv_config()
        self.threshold_config = get_threshold_config()
        self.voting_config = get_voting_config()
        self.feature_config = get_feature_config()
        self.global_config = get_global_config()
        self.hyperparameter_config = get_hyperparameter_config()
        
        self.random_state = random_state
        self.model_factory = MLModelFactory()
        
        # Results storage
        self.fold_results = []
        self.final_metrics = {}

        print("=" * 60)
    
    def run(self, data: pd.DataFrame, use_group_based_split = False,
                            model_type: str = 'SVM',
                            outer_folds: int = 5) -> Dict[str, Any]:
        """
        Run the complete training pipeline with the new systematic approach.
        
        Args:
            data: Input DataFrame with sequence windows
            model_type: Model type to use ('SVM', 'RandomForest', 'XGBoost')
            outer_folds: Number of outer CV folds
            
        Returns:
            Dict containing comprehensive results for all combinations
        """
        print(f"   Dataset: {len(data)} samples ({(data['known_cleavage_site'] == 1).sum()} positive)")

    
        # Outer CV loop
        if use_group_based_split:
            # Use protein entries as groups to ensure no protein appears in both train and test
            outer_cv = GroupKFold(n_splits=outer_folds)
            groups = data['entry']
            split_iterator = outer_cv.split(data, data['known_cleavage_site'], groups=groups)
        else:
            outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=self.random_state)
            split_iterator = outer_cv.split(data, data['known_cleavage_site'])
        
        for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(split_iterator):
            print(f"\n======== OUTER FOLD {fold_idx + 1}/{outer_folds} ========")
            
            train_outer = data.iloc[train_outer_idx].reset_index(drop=True)
            test_outer = data.iloc[test_outer_idx].reset_index(drop=True)
            fold_result = self._process_outer_fold(train_outer, test_outer, fold_idx + 1, model_type, use_group_based_split)
            self.fold_results.append(fold_result)
        
        # Aggregate results
        final_results = self._aggregate_results()
        
        self._print_final_summary(final_results)
        
        return final_results
    
    def _process_outer_fold(self, train_outer: pd.DataFrame, test_outer: pd.DataFrame, 
                          fold_idx: int, model_type: str, use_group_based_split: bool) -> Dict[str, Any]:
        """Process a single outer fold with the new approach."""
        
        print(f"   Train outer: {len(train_outer)} samples ({(train_outer['known_cleavage_site'] == 1).sum()} pos)")
        print(f"   Test outer: {len(test_outer)} samples ({(test_outer['known_cleavage_site'] == 1).sum()} pos)")

        # Step 1: Perform 3 inner splits
        all_inner_results = []
        all_hyperparameters = []
        all_best_thresholds = []
        
        # Create StratifiedKFold with 5 folds (yields ~80%-20% split)
        if use_group_based_split:
            print("   Step 1: Perform 3 inner splits using GroupKFold (5-fold, use first 3)")
            # Use protein entries as groups for inner CV as well
            inner_kfold = GroupKFold(n_splits=5)
            inner_groups = train_outer['entry']
            inner_splits = list(inner_kfold.split(train_outer, train_outer['known_cleavage_site'], groups=inner_groups))
        else:
            print("   Step 1: Perform 3 inner splits using StratifiedKFold (5-fold, use first 3)")
            inner_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state + fold_idx)
            inner_splits = list(inner_kfold.split(train_outer, train_outer['known_cleavage_site']))
        
        # Get the splits and use only the first 3
        
        for inner_split_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_splits[:3]):
            print(f"      Inner split {inner_split_idx + 1}/3:")
            
            # Create train/val datasets using fold indices
            train_inner = train_outer.iloc[train_inner_idx].reset_index(drop=True)
            val_inner = train_outer.iloc[val_inner_idx].reset_index(drop=True)
            
            print(f"         Train inner: {len(train_inner)} samples ({(train_inner['known_cleavage_site'] == 1).sum()} pos)")
            print(f"         Val inner: {len(val_inner)} samples ({(val_inner['known_cleavage_site'] == 1).sum()} pos)")

            # Step 2: Feature extraction (fit on train_inner only)
            print("         Feature extraction (fit on train_inner)")
            feature_extractor = DataLoader(verbose=False)
            train_inner_features = feature_extractor.fit_transform(train_inner)
            val_inner_features = feature_extractor.transform(val_inner)
            
            # Step 3: Create balanced subdatasets for train_inner
            print("         Create balanced subdatasets for train_inner")
            train_inner_data = train_inner.copy()
            inner_balanced_datasets = self._create_balanced_feature_datasets(train_inner_features, train_inner_data)
            print(f"         Created {len(inner_balanced_datasets)} balanced subdatasets")

            # Step 4: Hyperparameter tuning for each averaging method (based on AUC)
            print("         Hyperparameter tuning (optimizing AUC)")
            best_hyperparameters = self._tune_hyperparameters_auc(
                inner_balanced_datasets, val_inner_features, model_type
            )
            
            # Step 5: Threshold selection with best hyperparameters (based on F1)
            print("         Threshold selection (optimizing F1)")
            val_inner_results = self._train_and_evaluate(
                inner_balanced_datasets, val_inner_features, 
                best_hyperparameters, model_type
            )

            # Step 5.5: Select best thresholds from validation results (no counts)
            print("         Select best thresholds from validation")
            best_thresholds = self._select_best_thresholds_only(val_inner_results)
            
            # Store results from this inner split (no counts)
            all_inner_results.append(val_inner_results)
            all_hyperparameters.append(best_hyperparameters)
            print(all_hyperparameters)
            all_best_thresholds.append(best_thresholds)
            
            print(f"         Inner split {inner_split_idx + 1} completed")
        
        # Step 6: Average results from 3 inner splits (no counts)
        print("   Step 2: Average results from 3 inner splits")
        final_hyperparameters = self._average_hyperparameters(all_hyperparameters)
        print(final_hyperparameters)
        final_best_thresholds = self._average_best_thresholds(all_best_thresholds)
        
        print(f"         Final hyperparameters: {final_hyperparameters}")
        print(f"         Final best thresholds: {final_best_thresholds}")

        # Combine all inner results for reporting
        combined_val_inner_results = self._combine_inner_results(all_inner_results)
        
        # Step 3: Feature extraction for train_outer (refit on full train_outer)
        print("   Step 3: Feature extraction for train_outer (refit on full train_outer)")
        train_outer_feature_extractor = DataLoader(verbose=False)
        train_outer_features = train_outer_feature_extractor.fit_transform(train_outer)
        test_outer_features = train_outer_feature_extractor.transform(test_outer)
        
        # Step 4: Generate balanced datasets for train_outer  
        print("   Step 4: Generate balanced datasets for train_outer")
        train_outer_data = train_outer.copy()
        train_outer_balanced_datasets = self._create_balanced_feature_datasets(train_outer_features, train_outer_data)
        print(f"         Created {len(train_outer_balanced_datasets)} balanced subdatasets")
        
        # Step 5: Test on test_outer with averaged best parameters (no counts)
        print("   Step 5: Test on test_outer with averaged best parameters")
        test_outer_results = self._evaluate_on_test(
            train_outer_balanced_datasets, test_outer_features, 
            final_hyperparameters, model_type, final_best_thresholds
        )



        return {
            'fold_idx': fold_idx,
            'best_hyperparameters': final_hyperparameters,
            'best_thresholds': final_best_thresholds,
            'val_inner_results': combined_val_inner_results,
            'test_outer_results': test_outer_results,
            'all_inner_results': all_inner_results,
            'data_info': {
                'n_inner_splits': 3,
                'test_outer': len(test_outer),
                'n_subdatasets': len(train_outer_balanced_datasets)
            }
        }
    
    def _create_balanced_feature_datasets(self, train_inner_features: Dict[str, Any], 
                                         train_inner_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create balanced feature datasets from train_inner features."""
        
        # Create temporary DataFrame with features and labels for balanced sampling
        temp_df = train_inner_data.copy()
        
        # Use existing balanced dataset creation function
        balanced_segments = create_balanced_subdatasets(
            temp_df, 
            pos_neg_ratio=1.0,  # 1:1 ratio for balanced datasets
            random_state=self.random_state
        )

        # NOTE!!! The indices will be changed by the function create_balanced_subdatasets because of the Shuffle negative samples
        # Therefore, we need to get the indices from the original data

        # Convert to feature format
        feature_datasets = []
        for segment_df in balanced_segments:
            # Get indices of samples in this segment
            segment_indices = segment_df.index.tolist()
            
            # Extract corresponding features
            X_segment = train_inner_features['X'][segment_indices]
            y_segment = train_inner_features['y'][segment_indices]
            
            feature_datasets.append({'X': X_segment, 'y': y_segment})
        
        return feature_datasets
    
    def _tune_hyperparameters(self, balanced_datasets: List[Dict[str, Any]], 
                            val_features: Dict[str, Any], model_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters separately for each averaging method.
        
        Returns:
            Dict with best hyperparameters for each averaging method
        """
        print("      Tuning hyperparameters for averaging methods...")
        
        # Get hyperparameter grid
        hp_config = self.hyperparameter_config
        model_grid = hp_config[f'{model_type.lower()}_grid']
        scoring_metric = hp_config['scoring_metric']
        
        averaging_methods = ['logodds_average']
        best_hyperparameters = {}
        
        for avg_method in averaging_methods:
            print(f"        Tuning for {avg_method}...")
            
            # Train ensemble with different hyperparameters and get predictions for this averaging method
            best_score = -np.inf
            best_params = None
            
            # Grid search over hyperparameters
            for params in self._generate_param_combinations(model_grid):
                # Train models with these parameters
                trained_models = []
                for dataset in balanced_datasets:
                    model = self.model_factory.create_model(model_type, params)
                    model.fit(dataset['X'], dataset['y'])
                    trained_models.append(model)
                
                # Get ensemble predictions with this averaging method
                ensemble_proba = self._calculate_ensemble_probability(
                    trained_models, val_features['X'], avg_method
                )
                
                # Evaluate with simple threshold for HP selection
                y_pred = (ensemble_proba >= 0.5).astype(int)
                
                if scoring_metric == 'roc_auc':
                    score = roc_auc_score(val_features['y'], ensemble_proba)
                elif scoring_metric == 'f1':
                    score = f1_score(val_features['y'], y_pred)
                elif scoring_metric == 'balanced_accuracy':
                    score = balanced_accuracy_score(val_features['y'], y_pred)
                else:
                    score = roc_auc_score(val_features['y'], ensemble_proba)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            best_hyperparameters[avg_method] = {
                'params': best_params,
                'score': best_score,
                'metric': scoring_metric
            }
            
            print(f"         Best {avg_method}: {best_params} (score: {best_score:.4f})")
        
        return best_hyperparameters
    
    def _tune_hyperparameters_auc(self, balanced_datasets: List[Dict[str, Any]], 
                                val_features: Dict[str, Any], model_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters based on AUC score only.
        
        Returns:
            Dict with best hyperparameters for each averaging method
        """
        print("      Tuning hyperparameters based on AUC...")
        
        # Get hyperparameter grid
        hp_config = self.hyperparameter_config
        model_grid = hp_config[f'{model_type.lower()}_grid']
        
        averaging_methods = ['logodds_average']
        best_hyperparameters = {}
        
        for avg_method in averaging_methods:
            print(f"        Tuning for {avg_method}...")
            
            # Train ensemble with different hyperparameters and get predictions for this averaging method
            best_score = -np.inf
            best_params = None
            
            # Grid search over hyperparameters
            for params in self._generate_param_combinations(model_grid):
                # Train models with these parameters
                trained_models = []
                for dataset in balanced_datasets:
                    model = self.model_factory.create_model(model_type, params)
                    model.fit(dataset['X'], dataset['y'])
                    trained_models.append(model)
                
                # Get ensemble predictions with this averaging method
                ensemble_proba = self._calculate_ensemble_probability(
                    trained_models, val_features['X'], avg_method
                )
                
                # Evaluate with AUC score
                try:
                    score = roc_auc_score(val_features['y'], ensemble_proba)
                except:
                    score = 0.5  # Default AUC for edge cases
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            best_hyperparameters[avg_method] = {
                'params': best_params,
                'score': best_score,
                'metric': 'auc'
            }
            
            print(f"         Best {avg_method}: {best_params} (AUC: {best_score:.4f})")
        
        return best_hyperparameters

    def _average_hyperparameters(self, all_hyperparameters: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Average hyperparameters from multiple inner splits."""
        print("         Averaging hyperparameters from 3 inner splits...")
        
        averaging_methods = ['logodds_average']
        final_hyperparameters = {}
        
        for avg_method in averaging_methods:
            # Collect parameters from all splits - handle nested structure
            all_params = []
            all_scores = []
            for hp in all_hyperparameters:
                if avg_method in hp and 'params' in hp[avg_method]:
                    all_params.append(hp[avg_method]['params'])
                    all_scores.append(hp[avg_method]['score'])
            
            # For numerical parameters, take the average
            # For categorical parameters, take the mode (most frequent)
            averaged_params = {}
            
            if all_params:
                # Get all parameter names
                all_param_names = set()
                for params in all_params:
                    all_param_names.update(params.keys())
                
                for param_name in all_param_names:
                    param_values = [params[param_name] for params in all_params if param_name in params and params[param_name] is not None]
                    
                    if not param_values:
                        continue
                        
                    # Check if parameter is numerical
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numerical: take mean
                        averaged_params[param_name] = np.mean(param_values)
                        # Only convert to int for parameters that should be integers (not C, gamma, etc.)
                        if param_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] and isinstance(param_values[0], int):
                            averaged_params[param_name] = int(round(averaged_params[param_name]))
                    else:
                        # Categorical: take mode (most frequent)
                        from collections import Counter
                        counter = Counter(param_values)
                        averaged_params[param_name] = counter.most_common(1)[0][0]
            
            final_hyperparameters[avg_method] = {
                'params': averaged_params,
                'score': np.mean(all_scores),
                'metric': 'auc'
            }
            
            print(f"           {avg_method}: {averaged_params} (avg AUC: {np.mean(all_scores):.4f})")
        
        return final_hyperparameters
    
    
    def _average_best_thresholds(self, all_best_thresholds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average best thresholds from multiple inner splits."""
        print("         Averaging best thresholds from 3 inner splits...")
        
        averaging_methods = ['logodds_average']
        final_best_thresholds = {}
        
        for avg_method in averaging_methods:
            # Collect thresholds and scores from all splits
            thresholds = [bt[avg_method]['threshold'] for bt in all_best_thresholds if avg_method in bt]
            scores = [bt[avg_method]['score'] for bt in all_best_thresholds if avg_method in bt]
            
            if thresholds:
                # Take the average threshold
                avg_threshold = np.mean(thresholds)
                avg_score = np.mean(scores)
                
                final_best_thresholds[avg_method] = {
                    'threshold': avg_threshold,
                    'score': avg_score,
                    'metric': 'f1'
                }
                
                print(f"           {avg_method}: threshold={avg_threshold:.3f} (avg F1: {avg_score:.4f})")
        
        return final_best_thresholds
    
    def _combine_inner_results(self, all_inner_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple inner splits for reporting."""
        # For now, return the first split's results structure but note that it represents combined results
        if all_inner_results:
            combined = all_inner_results[0].copy()
            # Add metadata about combination
            combined['_combined_from_splits'] = len(all_inner_results)
            return combined
        return {}
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_ensemble_probability(self, models: List, X: np.ndarray, 
                                      method: str) -> np.ndarray:
        """
        Calculate ensemble probabilities using logodds averaging method only.
        
        Args:
            models: List of trained models
            X: Input features
            method: 'logodds_average' (simple_average removed)
            
        Returns:
            Ensemble probabilities
        """
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
            raise ValueError(f"Unknown averaging method: {method} (only 'logodds_average' supported)")
        
        return ensemble_proba
    
    def _train_and_evaluate(self, balanced_datasets: List[Dict[str, Any]], 
                          features: Dict[str, Any],
                          best_hyperparameters: Dict[str, Dict[str, Any]], 
                          model_type: str) -> Dict[str, Any]:
        """
        Train final models and evaluate all combinations.
        
        Returns comprehensive results for all averaging×voting combinations.
        """
        print("      Training final models and evaluating all combinations...")
        
        results = {}
        
        # Get threshold and count ranges
        probability_thresholds = self.threshold_config['threshold_range']
        
        # Process each averaging method
        for avg_method, hp_info in best_hyperparameters.items():
            print(f"        Processing {avg_method}...")
            
            # Train models with best hyperparameters
            trained_models = []
            for dataset in balanced_datasets:
                model = self.model_factory.create_model(model_type, hp_info['params'])
                model.fit(dataset['X'], dataset['y'])
                trained_models.append(model)
            
            # Get ensemble probabilities for validation and test sets
            ensemble_proba = self._calculate_ensemble_probability(
                trained_models, features['X'], avg_method
            )
            
            # Initialize results for this averaging method (only soft_vote)
            results[avg_method] = {
                'hyperparameters': hp_info,
                'n_models': len(trained_models),
                'soft_vote': {}
            }
            
            # Process SOFT VOTE ONLY (F1-optimized probability-based)
            print(f"          Soft Vote: Testing {len(probability_thresholds)} thresholds (F1-optimized)...")
            for threshold in probability_thresholds:
                # Predictions based on probability threshold
                pred = (ensemble_proba >= threshold).astype(int)
                
                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(features['y'], pred, ensemble_proba)
                
                results[avg_method]['soft_vote'][f'threshold_{threshold:.2f}'] = {
                    'threshold': threshold,
                    'metrics': metrics,
                    'probabilities': ensemble_proba.copy()
                }
            
            print(f"       {avg_method}: {len(results[avg_method]['soft_vote'])} soft vote combinations")
        
        return results
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate PS-score with β=1 (PS1)
        ps1 = self._calculate_ps_score(precision, specificity, beta=1.0)
        
        metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'ps1': ps1,
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
    
    def _calculate_ps_score(self, precision: float, specificity: float, beta: float = 1.0) -> float:
        """
        Calculate PSβ score: PSβ = (1 + β²) × Precision × Specificity / (β² × Precision + Specificity)
        
        This is similar to F-beta score but balances precision and specificity instead of precision and recall.
        
        Args:
            precision: Precision score (0-1)
            specificity: Specificity score (0-1)  
            beta: Beta parameter (default 1.0 for PS1, equal weight to precision and specificity)
            
        Returns:
            PSβ score (0-1)
        """
        if precision == 0 and specificity == 0:
            return 0.0
        
        numerator = (1 + beta**2) * precision * specificity
        denominator = (beta**2 * precision) + specificity
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _select_best_thresholds_only(self, val_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select best thresholds for each averaging method based on F1 score from validation results.
        
        Returns:
            best_thresholds - dictionary with best threshold values for each averaging method
        """
        best_thresholds = {}
        optimization_metric = 'f1'  # Always use F1 for threshold optimization
        print(f"      Selecting best thresholds from validation results ({optimization_metric}-optimized)...")
        for avg_method in val_results.keys():
            if avg_method in ['hyperparameters', 'n_models']:
                continue
                
            # Find best threshold for soft_vote based on F1 score
            if 'soft_vote' in val_results[avg_method]:
                best_threshold_score = -np.inf
                best_threshold = None
                
                for threshold_key, threshold_data in val_results[avg_method]['soft_vote'].items():
                    score = threshold_data['metrics'][optimization_metric]
                    if score > best_threshold_score:
                        best_threshold_score = score
                        best_threshold = threshold_data['threshold']
                
                best_thresholds[avg_method] = {
                    'threshold': best_threshold,
                    'score': best_threshold_score,
                    'metric': optimization_metric
                }
            
            print(f"         {avg_method}: Best threshold={best_thresholds.get(avg_method, {}).get('threshold', 'N/A')} (F1={best_thresholds.get(avg_method, {}).get('score', 0):.3f})")
        
        return best_thresholds
    
    
    def _evaluate_on_test(self, balanced_datasets: List[Dict[str, Any]], 
                         test_features: Dict[str, Any],
                         best_hyperparameters: Dict[str, Dict[str, Any]], 
                         model_type: str, 
                         best_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate on test set with both best validation thresholds and fixed thresholds.
        """
        print("      Testing final models on test set...")
        
        n_datasets = len(balanced_datasets)
        results = {}
        
        # Process each averaging method
        for avg_method, hp_info in best_hyperparameters.items():
            print(f"        Testing {avg_method}...")
            
            # Train models with best hyperparameters on full balanced datasets
            trained_models = []
            for dataset in balanced_datasets:
                model = self.model_factory.create_model(model_type, hp_info['params'])
                model.fit(dataset['X'], dataset['y'])
                trained_models.append(model)
            
            # Get ensemble probabilities
            ensemble_proba = self._calculate_ensemble_probability(
                trained_models, test_features['X'], avg_method
            )
            
            
            # Initialize results for this averaging method
            results[avg_method] = {}
            
            # SOFT VOTE ONLY: Best threshold from validation + Fixed 0.5 threshold
            results[avg_method]['soft_vote'] = {}
            
            # Best threshold from validation
            if avg_method in best_thresholds:
                best_thresh = best_thresholds[avg_method]['threshold']
                soft_pred_best = (ensemble_proba >= best_thresh).astype(int)
                soft_metrics_best = self._calculate_comprehensive_metrics(
                    test_features['y'], soft_pred_best, ensemble_proba
                )
                
                results[avg_method]['soft_vote']['best_validation'] = {
                    'threshold': best_thresh,
                    'metrics': soft_metrics_best,
                    'probabilities': ensemble_proba.copy(),
                    'predictions': soft_pred_best,
                    'val_score': best_thresholds[avg_method]['score']
                }
            
            # Fixed 0.5 threshold
            soft_pred_fixed = (ensemble_proba >= 0.5).astype(int)
            soft_metrics_fixed = self._calculate_comprehensive_metrics(
                test_features['y'], soft_pred_fixed, ensemble_proba
            )
            
            results[avg_method]['soft_vote']['fixed_0.5'] = {
                'threshold': 0.5,
                'metrics': soft_metrics_fixed,
                'probabilities': ensemble_proba.copy(),
                'predictions': soft_pred_fixed
            }
            
            # Count voting completely removed
            
            # Print summary with actual threshold values and F1 scores (soft vote only)
            best_soft_f1 = results[avg_method]['soft_vote'].get('best_validation', {}).get('metrics', {}).get('f1', 0)
            best_soft_thresh = results[avg_method]['soft_vote'].get('best_validation', {}).get('threshold', 0.5)
            fixed_soft_f1 = results[avg_method]['soft_vote']['fixed_0.5']['metrics']['f1']
            best_soft_auc = results[avg_method]['soft_vote'].get('best_validation', {}).get('metrics', {}).get('auc', 0)
            fixed_soft_auc = results[avg_method]['soft_vote']['fixed_0.5']['metrics']['auc']
            
            print(f"         {avg_method}: Soft(best={best_soft_thresh:.3f}→F1={best_soft_f1:.3f}/AUC={best_soft_auc:.3f}, fixed=0.5→F1={fixed_soft_f1:.3f}/AUC={fixed_soft_auc:.3f})")
        
        return results
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        print("\n🔍 AGGREGATING RESULTS ACROSS FOLDS...")
        
        # Get all unique combinations
        avg_methods = []
        voting_methods = []
        combinations = []
        
        for fold_result in self.fold_results:
            if 'test_outer_results' in fold_result:
                for avg_method in fold_result['test_outer_results'].keys():
                    if avg_method not in avg_methods:
                        avg_methods.append(avg_method)
                    
                    for voting_method in fold_result['test_outer_results'][avg_method].keys():
                        if voting_method not in voting_methods:
                            voting_methods.append(voting_method)
                        
                        for combination in fold_result['test_outer_results'][avg_method][voting_method].keys():
                            combo_key = f"{avg_method}_{voting_method}_{combination}"
                            if combo_key not in combinations:
                                combinations.append(combo_key)
        
        print(f"   Found {len(avg_methods)} averaging methods × {len(voting_methods)} voting methods")
        print(f"   Total combinations: {len(combinations)}")
        
        # Calculate mean and std for each combination
        aggregated_results = {}
        
        for avg_method in avg_methods:
            aggregated_results[avg_method] = {}
            
            for voting_method in voting_methods:
                aggregated_results[avg_method][voting_method] = {}
                
                # Get all combinations for this avg×voting pair
                combo_keys = set()
                for fold_result in self.fold_results:
                    if ('test_outer_results' in fold_result and 
                        avg_method in fold_result['test_outer_results'] and 
                        voting_method in fold_result['test_outer_results'][avg_method]):
                        combo_keys.update(fold_result['test_outer_results'][avg_method][voting_method].keys())
                
                for combo_key in combo_keys:
                    # Collect metrics across folds
                    test_metrics_list = []
                    val_metrics_list = []
                    
                    for fold_result in self.fold_results:
                        if ('test_outer_results' in fold_result and
                            avg_method in fold_result['test_outer_results'] and 
                            voting_method in fold_result['test_outer_results'][avg_method] and
                            combo_key in fold_result['test_outer_results'][avg_method][voting_method]):
                            
                            combo_data = fold_result['test_outer_results'][avg_method][voting_method][combo_key]
                            test_metrics_list.append(combo_data['metrics'])
                            
                            # Try to get validation metrics if available
                            if ('val_inner_results' in fold_result and
                                avg_method in fold_result['val_inner_results'] and
                                voting_method in fold_result['val_inner_results'][avg_method]):
                                # For validation, we need to find corresponding threshold/count
                                val_data = fold_result['val_inner_results'][avg_method][voting_method]
                                
                                # Find matching validation combination
                                if 'best_validation' in combo_key:
                                    # Find the best from validation set
                                    best_val_metric = -1
                                    best_val_data = None
                                    for val_combo_key, val_combo_data in val_data.items():
                                        if val_combo_data['metrics']['f1'] > best_val_metric:
                                            best_val_metric = val_combo_data['metrics']['f1']
                                            best_val_data = val_combo_data['metrics']
                                    if best_val_data:
                                        val_metrics_list.append(best_val_data)
                                elif 'fixed_' in combo_key:
                                    # For fixed combinations, find corresponding threshold/count in validation
                                    if 'fixed_0.5' in combo_key:
                                        for val_combo_key, val_combo_data in val_data.items():
                                            if 'threshold_0.50' in val_combo_key:
                                                val_metrics_list.append(val_combo_data['metrics'])
                                                break
                                    elif 'fixed_majority' in combo_key:
                                        # Use the middle count from validation as approximation
                                        val_keys = list(val_data.keys())
                                        if val_keys:
                                            mid_key = val_keys[len(val_keys)//2]
                                            val_metrics_list.append(val_data[mid_key]['metrics'])
                    
                    if test_metrics_list:  # If we have data
                        # Calculate mean and std
                        test_mean_std = self._calculate_mean_std_metrics(test_metrics_list)
                        val_mean_std = self._calculate_mean_std_metrics(val_metrics_list)
                        
                        aggregated_results[avg_method][voting_method][combo_key] = {
                            'test_metrics': test_mean_std,
                            'val_metrics': val_mean_std,
                            'n_folds': len(test_metrics_list)
                        }
                        
                        # Add threshold/count info
                        if 'threshold_' in combo_key:
                            threshold = float(combo_key.split('_')[-1])
                            aggregated_results[avg_method][voting_method][combo_key]['threshold'] = threshold
                        elif 'count_' in combo_key:
                            count = int(combo_key.split('_')[-1])
                            aggregated_results[avg_method][voting_method][combo_key]['count'] = count
        
        return {
            'aggregated_results': aggregated_results,
            'fold_results': self.fold_results,
            'summary': {
                'n_folds': len(self.fold_results),
                'averaging_methods': avg_methods,
                'voting_methods': voting_methods,
                'total_combinations': len(combinations)
            }
        }
    
    def _calculate_mean_std_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate mean and standard deviation for metrics across folds."""
        if not metrics_list:
            return {}
        
        # Get all metric names
        metric_names = metrics_list[0].keys()
        
        result = {}
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in metrics_list]
            result[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        return result
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final summary of results."""
        summary = results['summary']
        
        print(f"\n📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"   Folds: {summary['n_folds']}")
        print(f"   Averaging methods: {summary['averaging_methods']}")
        print(f"   Voting methods: {summary['voting_methods']}")
        print(f"   Total combinations: {summary['total_combinations']}")
        
        # Show best combinations
        aggregated = results['aggregated_results']
        
        print(f"\n🏆 TOP PERFORMERS (Test F1):")
        all_combinations = []
        
        for avg_method in aggregated.keys():
            for voting_method in aggregated[avg_method].keys():
                for combo_key, combo_data in aggregated[avg_method][voting_method].items():
                    f1_mean = combo_data['test_metrics']['f1']['mean']
                    f1_std = combo_data['test_metrics']['f1']['std']
                    
                    all_combinations.append({
                        'name': f"{avg_method}_{voting_method}_{combo_key}",
                        'f1_mean': f1_mean,
                        'f1_std': f1_std,
                        'combo_data': combo_data
                    })
        
        # Sort by F1 mean
        all_combinations.sort(key=lambda x: x['f1_mean'], reverse=True)
        
        for i, combo in enumerate(all_combinations[:5]):  # Top 5
            print(f"   {i+1}. {combo['name']}: F1 = {combo['f1_mean']:.4f} ± {combo['f1_std']:.4f}")
    
    
    def plot_test_results(self, 
                          results: Dict[str, Any], 
                          title: str = 'Test Performance: F1-Optimized Logodds + Soft Vote',
                          show_only_0_5_threshold: bool = True):
        """
        Plot streamlined test results for F1-optimized configuration.
        
        Shows clean comparison of F1-optimized vs baseline thresholds.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not results or 'aggregated_results' not in results:
            print(" No aggregated results found for plotting test results")
            return
        
        aggregated = results['aggregated_results']
        
        # Streamlined metrics
        metrics_to_plot = ['f1', 'precision', 'specificity', 'recall', 'accuracy', 'balanced_accuracy', 'auc']
        
        # Set white background
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Single plot for logodds_average results
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.92)
        
        # Streamlined strategies - only show 0.5 threshold by default
        if show_only_0_5_threshold:
            strategies = [
                ('soft_vote', 'fixed_0.5', 'Fixed Threshold = 0.5', '#1976D2')  # Dark Blue
            ]
        else:
            strategies = [
                ('soft_vote', 'best_validation', 'F1-Optimized Threshold', '#C2185B'),  # Dark Pink
                ('soft_vote', 'fixed_0.5', 'Fixed Threshold = 0.5', '#1976D2')  # Dark Blue
            ]
        
        # Prepare data for grouped bar plot
        x_pos = np.arange(len(metrics_to_plot))
        bar_width = 0.4 if show_only_0_5_threshold else 0.3
        
        for strategy_idx, (vote_method, combo_key, strategy_label, color) in enumerate(strategies):
            values = []
            errors = []
            
            for metric in metrics_to_plot:
                value = 0
                error = 0
                
                if ('logodds_average' in aggregated and 
                    vote_method in aggregated['logodds_average'] and
                    combo_key in aggregated['logodds_average'][vote_method]):
                    test_metrics = aggregated['logodds_average'][vote_method][combo_key]['test_metrics']
                    value = test_metrics[metric]['mean']
                    error = test_metrics[metric]['std']
                
                values.append(value)
                errors.append(error)
            
            # Create bars for this strategy - center bars if only one strategy
            if show_only_0_5_threshold:
                bar_positions = x_pos
            else:
                bar_positions = x_pos + (strategy_idx - 0.5) * bar_width
            
            bars = ax.bar(bar_positions, values, bar_width, yerr=errors,
                         label=strategy_label, color=color, alpha=0.8, capsize=4)
            
            # Add value labels on all bars
            for bar, value, error, metric in zip(bars, values, errors, metrics_to_plot):
                if value > 0:  # Show values for all metrics
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                           f'{value:.2f}\n±{error:.2f}', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot], 
                          fontsize=14, fontweight='bold', rotation=45, ha='right')
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True, 
                 bbox_to_anchor=(1, 1.1), loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)
        
        # Set y-axis limits with some headroom
        all_values = [v for v in values if v > 0]
        if all_values:
            ax.set_ylim(0, max(all_values) * 1.2)
        
        # Adjust layout to prevent overlap with more space for title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    def extract_final_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and deliver key results including best hyperparameters and voting parameters.
        
        Args:
            results: Results from run() method
            
        Returns:
            Dictionary with best hyperparameters and voting parameters for each averaging method
        """
        from collections import Counter
        
        if 'fold_results' not in results:
            print(" No fold results found")
            return {}
        
        fold_results = results['fold_results']
        n_folds = len(fold_results)
        
        print(f" EXTRACTING FINAL RESULTS FROM {n_folds} FOLDS")
        print("=" * 60)
        
        final_results = {}
        averaging_methods = ['logodds_average']
        
        for avg_method in averaging_methods:
            print(f"\n {avg_method.upper().replace('_', ' ')}:")
            final_results[avg_method] = {}
            
            # Extract hyperparameters from all folds
            hyperparameters = []
            for fold_result in fold_results:
                if ('best_hyperparameters' in fold_result and 
                    avg_method in fold_result['best_hyperparameters']):
                    hp = fold_result['best_hyperparameters'][avg_method]['params']
                    hyperparameters.append(hp)
            
            # Average hyperparameters across folds
            if hyperparameters:
                # Get all unique parameter names
                all_param_names = set()
                for hp in hyperparameters:
                    all_param_names.update(hp.keys())
                
                averaged_hp = {}
                for param_name in all_param_names:
                    param_values = []
                    for hp in hyperparameters:
                        if param_name in hp:
                            param_values.append(hp[param_name])
                    
                    if param_values:
                        # Check if parameter is numeric
                        if isinstance(param_values[0], (int, float)):
                            # Average numeric parameters
                            averaged_value = np.mean(param_values)
                            # Round to appropriate precision
                            if isinstance(param_values[0], int):
                                averaged_hp[param_name] = int(round(averaged_value))
                            else:
                                averaged_hp[param_name] = round(averaged_value, 6)
                        else:
                            # For categorical parameters, use the most common value
                            param_counter = Counter(param_values)
                            averaged_hp[param_name] = param_counter.most_common(1)[0][0]
                
                final_results[avg_method]['best_hyperparameters'] = {
                    'params': averaged_hp,
                    'method': 'averaged_across_folds',
                    'all_fold_params': hyperparameters,
                    'n_folds': len(hyperparameters)
                }
                
                print(f"   Averaged Hyperparameters ({len(hyperparameters)} folds):")
                for param, value in averaged_hp.items():
                    print(f"      {param}: {value}")
            
            # Extract best thresholds from all folds (no counts)
            best_thresholds = []
            
            for fold_result in fold_results:
                # Soft vote only
                if ('best_thresholds' in fold_result and 
                    avg_method in fold_result['best_thresholds']):
                    threshold = fold_result['best_thresholds'][avg_method]['threshold']
                    best_thresholds.append(threshold)
            
            # Process soft vote results
            if best_thresholds:
                final_threshold = np.mean(best_thresholds)
                threshold_method = f"average of all unique values"
                
                final_results[avg_method]['best_soft_vote'] = {
                    'final_threshold': final_threshold,
                    'selection_method': threshold_method,
                    'all_fold_thresholds': best_thresholds
                }
                
                print(f"   Best Soft Vote: {final_threshold:.3f} ({threshold_method})")
        
        # Add summary comparison
        print(f"\n SUMMARY COMPARISON:")
        print("-" * 40)
        
        for avg_method in averaging_methods:
            if avg_method in final_results:
                method_name = avg_method.replace('_', ' ').title()
                print(f"\n{method_name}:")
                
                if 'best_hyperparameters' in final_results[avg_method]:
                    hp = final_results[avg_method]['best_hyperparameters']['params']
                    print(f"   Hyperparameters: {hp}")
                
                
                if 'best_soft_vote' in final_results[avg_method]:
                    threshold = final_results[avg_method]['best_soft_vote']['final_threshold']
                    print(f"   Soft Vote: {threshold:.3f} threshold")
        
        # Add recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        print("For production use, apply these parameters:")
        
        for avg_method in averaging_methods:
            if avg_method in final_results:
                method_name = avg_method.replace('_', ' ').title()
                print(f"\n{method_name} Ensemble:")
                
                if 'best_hyperparameters' in final_results[avg_method]:
                    print(f"   1. Train models with: {final_results[avg_method]['best_hyperparameters']['params']}")
                
                
                if 'best_soft_vote' in final_results[avg_method]:
                    threshold = final_results[avg_method]['best_soft_vote']['final_threshold']
                    print(f"   2. Soft voting: Use threshold ≥{threshold:.3f} on ensemble probability")
        
        return final_results
