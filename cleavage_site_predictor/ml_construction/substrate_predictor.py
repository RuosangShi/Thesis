#!/usr/bin/env python3
"""
Substrate Predictor (Clean Version)
===================================

Streamlined substrate prediction system that inherits from CleavagePredictor 
and implements protein-level prediction using logodds averaging with soft voting.

Features:
- Top-k pooling aggregation (k=1,3,5)  
- Poisson-binomial aggregation accounting for window count
- Threshold calibration for optimal balanced accuracy
- Clean evaluation and visualization tools
- Uses only logodds_average + soft_vote combination for optimal performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.model_selection import train_test_split

from .cleavage_predictor import CleavagePredictor
from ._util_substrate_predictor import create_clean_evaluation_plots


class SubstratePredictor(CleavagePredictor):
    """
    Streamlined substrate prediction system extending CleavagePredictor.

    Implements protein-level prediction using two aggregation strategies:
    - Top-k pooling (k=1,3,5) and Poisson-binomial aggregation
    - Threshold calibration for optimal balanced accuracy
    - Optional RSA filtering to focus on accessible cleavage sites (configurable RSA range)
    """
    
    def __init__(self, random_state: int = 42, filter_site_by_rsa: bool = False,
                 rsa_min: float = 0.4, rsa_max: float = 1.0):
        """
        Initialize SubstratePredictor.

        Args:
            random_state: Random seed for reproducibility
            filter_site_by_rsa: Whether to filter sites by RSA accessibility
            rsa_min: Minimum RSA threshold for accessible sites (default: 0.4)
            rsa_max: Maximum RSA threshold for accessible sites (default: 1.0)
        """
        super(SubstratePredictor, self).__init__(random_state=random_state)

        # RSA filtering configuration
        self.filter_site_by_rsa = filter_site_by_rsa
        self.rsa_range = (rsa_min, rsa_max)  # Configurable RSA range for accessible sites

        # Simplified aggregation parameters
        self.aggregation_params = {
            'top_k': [1, 3, 5],  # Top-k values to test
            'poisson_penalty': True  # Whether to penalize by window count
        }
    
    def fit_with_trainer_results(self, training_data: pd.DataFrame, 
                                trainer_results: Dict[str, Any], model_type: str = 'SVM') -> None:
        """
        Fit the substrate predictor using results from trainer.extract_final_results().
        
        Args:
            training_data: Training DataFrame with sequences and labels
            trainer_results: Dict from trainer.extract_final_results()
            model_type: Type of models to train ('SVM', 'RandomForest', 'XGBoost')
        """
        print(f" FITTING SUBSTRATE PREDICTOR WITH TRAINER RESULTS")
        print(f"   Training samples: {len(training_data)} ({(training_data['known_cleavage_site'] == 1).sum()} positive)")
        print(f"   Model type: {model_type}")
        
        # Parse trainer results into CleavagePredictor format
        parsed_params = self._parse_trainer_results(trainer_results)
        
        # Display parsed configuration
        self._display_trainer_configuration(trainer_results)
        
        # Call parent fit method with parsed parameters
        super(SubstratePredictor, self).fit(training_data, parsed_params, model_type)
        
        print(f"    SubstratePredictor fitted with optimal thresholds from trainer")
    
    def _parse_trainer_results(self, trainer_results: Dict) -> Dict:
        """Parse hyperparameters from trainer results."""
        parsed = {}
        
        # Extract logodds_average hyperparameters only
        if 'logodds_average' in trainer_results:
            method_config = trainer_results['logodds_average']
            
            if 'best_hyperparameters' in method_config:
                hp_data = method_config['best_hyperparameters']
                if 'params' in hp_data:
                    params = {}
                    for k, v in hp_data['params'].items():
                        if hasattr(v, 'item'):  # numpy scalar
                            params[k] = v.item()
                        else:
                            params[k] = v
                    parsed['logodds_average'] = params
                else:
                    parsed['logodds_average'] = method_config.copy()
            else:
                parsed['logodds_average'] = method_config.copy()
        
        # Extract soft voting threshold
        if 'logodds_average' in trainer_results:
            method_config = trainer_results['logodds_average']
            
            soft_threshold = 0.5  # default
            if 'best_soft_vote' in method_config:
                soft_data = method_config['best_soft_vote']
                if 'final_threshold' in soft_data:
                    threshold_val = soft_data['final_threshold']
                    soft_threshold = threshold_val.item() if hasattr(threshold_val, 'item') else threshold_val
            
            parsed['thresholds'] = {
                'soft_vote': [soft_threshold]
            }
        
        return parsed
    
    def _display_trainer_configuration(self, trainer_results: Dict):
        """Display the trainer configuration."""
        print("\n    OPTIMAL CONFIGURATION FROM TRAINER:")
        
        if 'logodds_average' in trainer_results:
            config = trainer_results['logodds_average']
            print(f"\n       Logodds Average Ensemble (F1-optimized):")
            
            if 'best_hyperparameters' in config:
                hp_data = config['best_hyperparameters']
                if 'params' in hp_data:
                    params = hp_data['params']
                    params_str = ', '.join([f"{k}: {v}" for k, v in params.items()])
                    print(f"         1. Train models with: {{{params_str}}}")
                    print(f"            Method: averaged_across_folds (5 folds)")
            
            if 'best_soft_vote' in config:
                soft_data = config['best_soft_vote']
                soft_thresh = soft_data.get('final_threshold', 'unknown')
                if hasattr(soft_thresh, 'item'):
                    soft_thresh = soft_thresh.item()
                selection_method = soft_data.get('selection_method', 'unknown')
                print(f"         2. Soft voting: Use threshold ≥{soft_thresh:.3f} on ensemble probability")
                print(f"            Selection: {selection_method}")
        print()
    
    def evaluate_with_calibration(self, substrate_predictions: Dict, nonsubstrate_predictions: Dict,
                                 performance_title: str = "Substrate Prediction: Threshold 0.5 vs Calibrated Comparison",
                                 confusion_title: str = "Confusion Matrices with Calibrated Thresholds", 
                                 cv_folds: int = 5, random_state: int = 42,
                                 bal_acc_weight: float = 1, f1_weight: float = 0):
        """
        Combined evaluation with calibration using cross-validation for robust statistics.
        
        The calibration optimizes a composite score that balances Balanced Accuracy and F1-score:
        Composite Score = bal_acc_weight × Balanced_Accuracy + f1_weight × F1_Score
        
        Args:
            substrate_predictions: Predictions for known substrates
            nonsubstrate_predictions: Predictions for known non-substrates
            performance_title: Title for performance comparison plot
            confusion_title: Title for confusion matrices plot
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            bal_acc_weight: Weight for balanced accuracy in composite score (default: 0.7)
            f1_weight: Weight for F1-score in composite score (default: 0.3)
            
        Returns:
            Dictionary with CV calibration results and evaluation metrics
        """
        print(" COMBINED EVALUATION WITH CROSS-VALIDATION CALIBRATION")
        print(f"   Optimization: {bal_acc_weight:.1f}×Bal_Acc + {f1_weight:.1f}×F1")
        print("=" * 70)
        
        # Run cross-validation calibration to get robust statistics
        cv_results = self._calibrate_with_cross_validation(
            substrate_predictions, nonsubstrate_predictions, 
            cv_folds, random_state, bal_acc_weight, f1_weight
        )
        
        # Create comparison plots with CV statistics (mean ± std)
        create_clean_evaluation_plots(
            substrate_predictions, nonsubstrate_predictions,
            performance_title, confusion_title,
            cv_results
        )
        
        return cv_results
    
    def predict_substrates(self, query_data: pd.DataFrame,
                          aggregation_strategies: List[str] = None,
                          custom_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Predict substrate status using protein-level aggregation of cleavage site predictions.

        Args:
            query_data: DataFrame with protein windows for prediction
            aggregation_strategies: List of strategies to use ['top_k_pooling', 'poisson_binomial']
            custom_thresholds: Custom probability thresholds (default: [0.5])

        Returns:
            Dictionary with aggregated predictions for each strategy
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before prediction")

        if aggregation_strategies is None:
            aggregation_strategies = ['top_k_pooling', 'poisson_binomial']

        print(f" SUBSTRATE PREDICTION")
        print(f"   Query samples: {len(query_data)}")
        print(f"   RSA filtering: {'Enabled' if self.filter_site_by_rsa else 'Disabled'}")
        print(f"   Aggregation strategies: {aggregation_strategies}")

        # Apply RSA filtering if enabled
        if self.filter_site_by_rsa:
            query_data_filtered = self._filter_sites_by_rsa(query_data)
            print(f"   After RSA filtering: {len(query_data_filtered)} samples (filtered out {len(query_data) - len(query_data_filtered)})")
        else:
            query_data_filtered = query_data

        # Get cleavage site predictions first
        cleavage_predictions = super(SubstratePredictor, self).predict(query_data_filtered, custom_thresholds)

        # Extract logodds_average soft_vote results
        if 'logodds_average' not in cleavage_predictions:
            raise ValueError("No logodds_average predictions found")

        window_predictions = cleavage_predictions['logodds_average']['soft_vote']

        # Group predictions by protein
        protein_windows = {}
        for idx, row in query_data_filtered.iterrows():
            protein_id = row['entry']
            if protein_id not in protein_windows:
                protein_windows[protein_id] = []
            protein_windows[protein_id].append(idx)

        print(f"   Grouped into {len(protein_windows)} proteins")

        # Apply aggregation strategies
        results = {}

        for strategy in aggregation_strategies:
            if strategy == 'top_k_pooling':
                results[strategy] = self._apply_top_k_pooling(
                    window_predictions, protein_windows, query_data_filtered
                )
            elif strategy == 'poisson_binomial':
                results[strategy] = self._apply_poisson_binomial(
                    window_predictions, protein_windows, query_data_filtered
                )

        print("Substrate prediction completed!")
        return results

    def _filter_sites_by_rsa(self, query_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter cleavage sites by average RSA (Relative Solvent Accessibility) values across the window.
        Uses DSSPProfiler (same as NodeFeatures) for consistency.

        Args:
            query_data: DataFrame with protein windows for prediction

        Returns:
            Filtered DataFrame containing only windows with average RSA in the specified range
        """
        from ..structural_analyzer.structural_profiler import DSSPProfiler

        print(f"    RSA Filtering Configuration:")
        print(f"      • RSA range: {self.rsa_range[0]:.1f} - {self.rsa_range[1]:.1f}")
        print(f"      • Before filtering: {len(query_data)} windows")

        # Initialize DSSP profiler (same as NodeFeatures)
        dssp_profiler = DSSPProfiler()

        try:
            # Extract RSA values with mean calculation using built-in DSSPProfiler functionality
            dssp_results = dssp_profiler.profile_dssp(
                query_data,
                include_rsa=True,
                include_secondary_structure=False,  # We only need RSA
                rsa_cal='Wilke',
                structure_source="alphafold",
                show_original_data=False,
                include_rsa_mean=True  # Use built-in RSA mean calculation
            )

            if 'rsa_mean' not in dssp_results.columns:
                print("       Warning: No RSA mean data found in DSSP results")
                return pd.DataFrame(columns=query_data.columns)

            filtered_indices = []
            rsa_stats = {
                'total': 0,
                'nan_values': 0,
                'below_min': 0,
                'above_max': 0,
                'in_range': 0
            }

            # Process each window using pre-calculated RSA means
            for idx, row in query_data.iterrows():
                try:
                    # Get pre-calculated average RSA for this window
                    avg_window_rsa = dssp_results.loc[idx, 'rsa_mean']
                    rsa_stats['total'] += 1

                    # Check RSA value and categorize
                    if np.isnan(avg_window_rsa):
                        rsa_stats['nan_values'] += 1
                    elif avg_window_rsa < self.rsa_range[0]:
                        rsa_stats['below_min'] += 1
                    elif avg_window_rsa > self.rsa_range[1]:
                        rsa_stats['above_max'] += 1
                    else:
                        # In accessible range
                        rsa_stats['in_range'] += 1
                        filtered_indices.append(idx)

                except Exception as e:
                    print(f"       Warning: Could not process window {idx}: {str(e)}")
                    continue

            # Return filtered DataFrame with reset index
            filtered_data = query_data.loc[filtered_indices].reset_index(drop=True)

            # Print detailed filtering results
            print(f"    RSA Filtering Results:")
            print(f"      • Before filtering: {len(query_data):,} windows")
            print(f"      • After filtering: {len(filtered_data):,} windows")
            print(f"      • Filtered out: {len(query_data) - len(filtered_data):,} windows ({(len(query_data) - len(filtered_data))/len(query_data)*100:.1f}%)")
            print(f"    RSA Distribution:")
            print(f"      • In range [{self.rsa_range[0]:.1f}-{self.rsa_range[1]:.1f}]: {rsa_stats['in_range']:,} ({rsa_stats['in_range']/rsa_stats['total']*100:.1f}%)")
            print(f"      • Below {self.rsa_range[0]:.1f} (buried): {rsa_stats['below_min']:,} ({rsa_stats['below_min']/rsa_stats['total']*100:.1f}%)")
            print(f"      • Above {self.rsa_range[1]:.1f} (over-exposed): {rsa_stats['above_max']:,} ({rsa_stats['above_max']/rsa_stats['total']*100:.1f}%)")
            print(f"      • NaN/missing: {rsa_stats['nan_values']:,} ({rsa_stats['nan_values']/rsa_stats['total']*100:.1f}%)")

            return filtered_data

        except Exception as e:
            print(f"     Error in RSA filtering: {str(e)}")
            print("     Returning unfiltered data")
            return query_data

    def _apply_top_k_pooling(self, window_predictions: Dict, protein_windows: Dict,
                           query_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply top-k pooling aggregation strategy."""
        results = {}
        
        for k in self.aggregation_params['top_k']:  # [1, 3, 5]
            k_results = {}
            
            for threshold_key, threshold_data in window_predictions.items():
                probabilities = threshold_data['probabilities']
                threshold_val = threshold_data['threshold']
                
                protein_scores = {}
                
                for protein_id, window_indices in protein_windows.items():
                    # Get probabilities for this protein's windows
                    protein_probs = [probabilities[idx] for idx in window_indices]
                    
                    # Top-k pooling: average of top k highest probabilities
                    # Formula: protein_score = (1/k) * sum(top_k(window_probabilities))
                    top_k_probs = sorted(protein_probs, reverse=True)[:k]
                    protein_score = np.mean(top_k_probs)
                    protein_scores[protein_id] = protein_score
                
                k_results[threshold_key] = {
                    'protein_scores': protein_scores,
                    'method': f'top_{k}_pooling'
                }
            
            results[f'top_{k}'] = k_results
        
        return results
    
    def _apply_poisson_binomial(self, window_predictions: Dict, protein_windows: Dict,
                              query_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply Poisson-binomial aggregation with window count penalty."""
        results = {}
        
        for threshold_key, threshold_data in window_predictions.items():
            probabilities = threshold_data['probabilities']
            threshold_val = threshold_data['threshold']
            
            protein_scores = {}
            
            for protein_id, window_indices in protein_windows.items():
                # Get probabilities for this protein's windows
                protein_probs = [probabilities[idx] for idx in window_indices]
                n_windows = len(protein_probs)
                
                # Poisson-binomial: probability that at least one window is positive
                # P(at least one positive) = 1 - P(all negative) = 1 - ∏(1 - pi)
                prob_all_negative = np.prod([1 - p for p in protein_probs])
                protein_score = 1 - prob_all_negative
                
                # Apply window count penalty if enabled
                if self.aggregation_params.get('poisson_penalty', True):
                    # Penalize proteins with fewer windows using logarithmic penalty
                    # Formula: penalty = 1 / (1 + 0.1 * log(n_windows))
                    penalty_factor = 1.0 / (1.0 + 0.1 * np.log(max(n_windows, 1)))
                    protein_score = protein_score * penalty_factor
                
                protein_scores[protein_id] = protein_score
            
            results[threshold_key] = {
                'protein_scores': protein_scores,
                'method': 'poisson_binomial',
                'window_penalty': self.aggregation_params.get('poisson_penalty', True)
            }
        
        return results
    
    def _calibrate_with_cross_validation(self, substrate_predictions: Dict, nonsubstrate_predictions: Dict,
                                       cv_folds: int = 5, random_state: int = 42,
                                       bal_acc_weight: float = 1, f1_weight: float = 0):
        """
        Perform cross-validation calibration to get robust threshold and performance statistics.
        
        Args:
            substrate_predictions: Predictions for known substrates
            nonsubstrate_predictions: Predictions for known non-substrates
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with cross-validation results including means and standard deviations
        """
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        
        print(f" Cross-Validation Calibration ({cv_folds} folds)")
        print("=" * 50)
        
        # Extract method data and create protein-level datasets (same as before)
        methods_data = {}
        
        # Process each aggregation method
        for strategy in ['top_k_pooling', 'poisson_binomial']:
            if strategy in substrate_predictions and strategy in nonsubstrate_predictions:
                
                sub_data = substrate_predictions[strategy]
                nonsub_data = nonsubstrate_predictions[strategy]
                configs = set(sub_data.keys()) & set(nonsub_data.keys())
                
                for config in configs:
                    try:
                        # Extract protein scores
                        if strategy == 'poisson_binomial':
                            sub_scores = list(sub_data['threshold_0.50']['protein_scores'].values())
                            nonsub_scores = list(nonsub_data['threshold_0.50']['protein_scores'].values())
                            method_name = 'poisson_binomial'
                        else:
                            sub_scores = list(sub_data[config]['threshold_0.50']['protein_scores'].values())
                            nonsub_scores = list(nonsub_data[config]['threshold_0.50']['protein_scores'].values())
                            method_name = f"top_{config.split('_')[1]}"
                        
                        # Create combined dataset
                        all_scores = np.array(sub_scores + nonsub_scores)
                        all_labels = np.array([1] * len(sub_scores) + [0] * len(nonsub_scores))
                        
                        methods_data[method_name] = {
                            'scores': all_scores,
                            'labels': all_labels,
                            'n_substrates': len(sub_scores),
                            'n_nonsubstrates': len(nonsub_scores)
                        }
                        
                        # For poisson_binomial, only process once
                        if strategy == 'poisson_binomial':
                            break
                            
                    except Exception as e:
                        print(f"Error processing {strategy}-{config}: {e}")
                        continue
        
        print(f" Processing {len(methods_data)} aggregation methods with CV")
        
        # Cross-validation calibration for each method
        cv_results = {}
        
        for method_name, data in methods_data.items():
            print(f"\n CV Calibrating {method_name.upper()}...")
            
            X, y = data['scores'], data['labels']
            
            # Initialize CV storage
            fold_thresholds = []
            fold_metrics = {
                'f1': [], 'balanced_accuracy': [], 'precision': [], 'recall': [], 'accuracy': []
            }
            fold_confusion_matrices = []  # Store confusion matrices for each fold
            
            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
                print(f"      Fold {fold+1}/{cv_folds}")
                
                # Split train+val vs test
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]
                
                # Further split train+val into train and val (70/30)
                val_size = int(0.3 * len(X_train_val))
                train_size = len(X_train_val) - val_size
                
                # Simple split (already stratified by outer CV)
                X_train, X_val = X_train_val[:train_size], X_train_val[train_size:]
                y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]
                
                print(f"         Val set: {len(y_val)} samples, {np.sum(y_val)} positive ({np.mean(y_val)*100:.1f}%)")
                
                # Threshold calibration on validation set - optimize Bal_Acc while keeping F1 high
                thresholds_to_test = np.arange(0.1, 0.95, 0.02)
                best_threshold = 0.5
                best_composite_score = 0.0
                best_bal_acc = 0.0
                best_f1 = 0.0
                
                from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score
                
                for threshold in thresholds_to_test:
                    y_pred_val = (X_val >= threshold).astype(int)
                    bal_acc = balanced_accuracy_score(y_val, y_pred_val)
                    f1 = f1_score(y_val, y_pred_val, zero_division=0)
                    
                    # Composite score: weighted combination of Bal_Acc and F1
                    composite_score = bal_acc_weight * bal_acc + f1_weight * f1
                    
                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_threshold = threshold
                        best_bal_acc = bal_acc
                        best_f1 = f1
                
                print(f"        ✨ Best threshold: {best_threshold:.3f} (composite: {best_composite_score:.3f})")
                
                # Test performance with best threshold
                y_pred_test = (X_test >= best_threshold).astype(int)
                
                # Calculate confusion matrix for this fold
                from sklearn.metrics import confusion_matrix
                cm_fold = confusion_matrix(y_test, y_pred_test)
                # Ensure 2x2 matrix even if some classes are missing
                if cm_fold.shape == (1, 1):
                    # Only one class predicted - expand to 2x2
                    if y_test.sum() == len(y_test):  # All positives
                        cm_fold = np.array([[0, 0], [0, cm_fold[0, 0]]])
                    else:  # All negatives
                        cm_fold = np.array([[cm_fold[0, 0], 0], [0, 0]])
                elif cm_fold.shape != (2, 2):
                    # Handle other edge cases
                    cm_fold = np.zeros((2, 2))
                
                # Store fold results
                fold_thresholds.append(best_threshold)
                fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred_test))
                fold_metrics['f1'].append(f1_score(y_test, y_pred_test, zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred_test, zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred_test, zero_division=0))
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred_test))
                fold_confusion_matrices.append(cm_fold)
            
            # Calculate CV statistics including confusion matrix statistics
            fold_cms = np.array(fold_confusion_matrices)  # Shape: (n_folds, 2, 2)
            cm_mean = np.mean(fold_cms, axis=0)  # Mean confusion matrix
            cm_std = np.std(fold_cms, axis=0)   # Std for each cell
            
            cv_results[method_name] = {
                'best_threshold': {
                    'mean': np.mean(fold_thresholds),
                    'std': np.std(fold_thresholds),
                    'values': fold_thresholds
                },
                'test_metrics': {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    } for metric, values in fold_metrics.items()
                },
                'confusion_matrix_cv': {
                    'mean': cm_mean,
                    'std': cm_std,
                    'individual_folds': fold_cms
                },
                'cv_folds': cv_folds
            }
            
        # Find overall best method based on CV balanced accuracy
        best_method = max(cv_results.keys(), 
                         key=lambda m: cv_results[m]['test_metrics']['balanced_accuracy']['mean'])
        best_result = cv_results[best_method]


        return {
            'cv_calibration_results': cv_results,
            'best_method': best_method,
            'best_result': best_result,
            'cv_folds': cv_folds,
            'type': 'cross_validation'
        }
    
    def _calibrate_and_evaluate_thresholds(self, substrate_predictions: Dict, nonsubstrate_predictions: Dict,
                                        val_split: float = 0.3, test_split: float = 0.3, random_state: int = 42,
                                        bal_acc_weight: float = 1, f1_weight: float = 0):
        """
        Comprehensive threshold calibration and evaluation system.
        
        1. Split substrate/non-substrate data into train/val/test
        2. Calibrate thresholds on validation set (optimize balanced accuracy)
        3. Evaluate on test set with calibrated thresholds
        4. Compare all methods to find the best combination
        
        Args:
            substrate_predictions: Predictions for known substrates
            nonsubstrate_predictions: Predictions for known non-substrates
            val_split: Proportion for validation split
            test_split: Proportion for test split
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with calibration results and best method
        """
        print(" THRESHOLD CALIBRATION SYSTEM")
        print("=" * 60)
        
        # Extract method data and create protein-level datasets
        methods_data = {}
        
        # Process each aggregation method
        for strategy in ['top_k_pooling', 'poisson_binomial']:
            if strategy in substrate_predictions and strategy in nonsubstrate_predictions:
                
                sub_data = substrate_predictions[strategy]
                nonsub_data = nonsubstrate_predictions[strategy]
                configs = set(sub_data.keys()) & set(nonsub_data.keys())
                
                for config in configs:
                    try:
                        # Extract protein scores
                        if strategy == 'poisson_binomial':
                            sub_scores = list(sub_data['threshold_0.50']['protein_scores'].values())
                            nonsub_scores = list(nonsub_data['threshold_0.50']['protein_scores'].values())
                            sub_proteins = list(sub_data['threshold_0.50']['protein_scores'].keys())
                            nonsub_proteins = list(nonsub_data['threshold_0.50']['protein_scores'].keys())
                            method_name = 'poisson_binomial'
                        else:
                            sub_scores = list(sub_data[config]['threshold_0.50']['protein_scores'].values())
                            nonsub_scores = list(nonsub_data[config]['threshold_0.50']['protein_scores'].values())
                            sub_proteins = list(sub_data[config]['threshold_0.50']['protein_scores'].keys())
                            nonsub_proteins = list(nonsub_data[config]['threshold_0.50']['protein_scores'].keys())
                            method_name = f"top_{config.split('_')[1]}"
                        
                        # Create combined dataset
                        all_scores = np.array(sub_scores + nonsub_scores)
                        all_labels = np.array([1] * len(sub_scores) + [0] * len(nonsub_scores))
                        all_proteins = sub_proteins + nonsub_proteins
                        
                        methods_data[method_name] = {
                            'scores': all_scores,
                            'labels': all_labels,
                            'proteins': all_proteins,
                            'n_substrates': len(sub_scores),
                            'n_nonsubstrates': len(nonsub_scores)
                        }
                        
                        # For poisson_binomial, only process once
                        if strategy == 'poisson_binomial':
                            break
                            
                    except Exception as e:
                        print(f"Error processing {strategy}-{config}: {e}")
                        continue
        
        print(f"Processing {len(methods_data)} aggregation methods")
        for method, data in methods_data.items():
            print(f"  {method}: {data['n_substrates']} substrates, {data['n_nonsubstrates']} non-substrates")
        
        # Calibrate thresholds for each method
        calibration_results = {}
        
        for method_name, data in methods_data.items():
            print(f"\n Calibrating {method_name.upper()}...")
            
            # Split data: train/val/test
            X, y = data['scores'], data['labels']
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state, stratify=y
            )
            
            # Second split: train/validation from remaining data
            val_size_adjusted = val_split / (1 - test_split)  # Adjust validation size
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
            )
            
            print(f"  Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Threshold calibration on validation set - optimize Bal_Acc while keeping F1 high
            thresholds_to_test = np.arange(0.1, 0.95, 0.02)  # Fine-grained search
            best_threshold = 0.5
            best_composite_score = 0.0
            best_bal_acc = 0.0
            best_f1 = 0.0
            threshold_results = []
            
            from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score
            
            print(f"     Testing {len(thresholds_to_test)} thresholds from {thresholds_to_test[0]:.2f} to {thresholds_to_test[-1]:.2f}")
            print(f"     Validation set: {len(y_val)} samples, {np.sum(y_val)} positive ({np.mean(y_val)*100:.1f}%)")
            
            for i, threshold in enumerate(thresholds_to_test):
                y_pred_val = (X_val >= threshold).astype(int)
                bal_acc = balanced_accuracy_score(y_val, y_pred_val)
                f1 = f1_score(y_val, y_pred_val, zero_division=0)
                
                # Composite score: weighted combination of Bal_Acc and F1
                composite_score = bal_acc_weight * bal_acc + f1_weight * f1
                
                threshold_results.append({
                    'threshold': threshold,
                    'balanced_accuracy': bal_acc,
                    'f1': f1,
                    'composite_score': composite_score,
                    'accuracy': accuracy_score(y_val, y_pred_val),
                    'precision': precision_score(y_val, y_pred_val, zero_division=0),
                    'recall': recall_score(y_val, y_pred_val, zero_division=0)
                })
                
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_threshold = threshold
                    best_bal_acc = bal_acc
                    best_f1 = f1
               
            print(f"     Best threshold found: {best_threshold:.3f} (composite score: {best_composite_score:.3f})")
            
            # Test performance with best threshold
            y_pred_test = (X_test >= best_threshold).astype(int)
            test_metrics = {
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
                'f1': f1_score(y_test, y_pred_test, zero_division=0),
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, zero_division=0),
                'recall': recall_score(y_test, y_pred_test, zero_division=0)
            }
            
            calibration_results[method_name] = {
                'best_threshold': best_threshold,
                'best_val_balanced_accuracy': best_bal_acc,
                'test_metrics': test_metrics,
                'threshold_results': threshold_results,
                'data_splits': {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test)
                }
            }
            
            print(f"     Best threshold: {best_threshold:.3f} (optimized for {bal_acc_weight:.1f}×Bal_Acc + {f1_weight:.1f}×F1)")
            print(f"     Val performance: Bal.Acc={best_bal_acc:.3f}, F1={best_f1:.3f}")
            print(f"     Test performance: Bal.Acc={test_metrics['balanced_accuracy']:.3f}, F1={test_metrics['f1']:.3f}")
        
        # Find overall best method
        best_method = max(calibration_results.keys(), 
                         key=lambda m: calibration_results[m]['test_metrics']['balanced_accuracy'])
        best_result = calibration_results[best_method]
        
        
        return {
            'calibration_results': calibration_results,
            'best_method': best_method,
            'best_threshold': best_result['best_threshold'],
            'best_performance': best_result['test_metrics'],
            'recommendation': {
                'method': best_method,
                'threshold': best_result['best_threshold'],
                'expected_balanced_accuracy': best_result['test_metrics']['balanced_accuracy'],
                'expected_f1': best_result['test_metrics']['f1']
            }
        }
    
