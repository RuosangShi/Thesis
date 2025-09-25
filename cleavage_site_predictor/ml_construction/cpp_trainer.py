#!/usr/bin/env python3
"""
CPP Feature Optimization Trainer
===============================

This module implements a grid search trainer for CPP (Conjoint Triad Patterns) features
to find the optimal combination of window size and number of features (n_filter).

Uses the existing DataLoader and ML pipeline architecture for consistency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .config import get_feature_config
from .trainer import EnsembleTrainer
from .ml_models import MLModelFactory


class CPPFeatureTrainer(EnsembleTrainer):
    """
    Specialized trainer that extends EnsembleTrainer for CPP feature optimization.
    Uses proper cross-validation and hyperparameter tuning but evaluates with 0.5 threshold.
    """
    
    def __init__(self, cpp_n_filter: int, random_state: int = 42):
        """
        Initialize CPP feature trainer.
        
        Args:
            cpp_n_filter: Number of CPP features to use
            random_state: Random state for reproducibility
        """
        super().__init__(random_state=random_state)
        self.cpp_n_filter = cpp_n_filter
        
        # Override feature config to use only CPP features
        self.feature_config = self._create_cpp_only_config(cpp_n_filter)
    
    def _create_cpp_only_config(self, cpp_n_filter: int) -> Dict[str, Any]:
        """Create feature configuration with only CPP features enabled."""
        config = get_feature_config()
        
        # Disable all features except CPP
        config['include_sequence_features'] = False
        config['include_structure_features'] = False
        config['include_plm_embeddings'] = False
        config['include_weighted_metrix'] = False
        config['include_cksaap'] = False
        config['include_coevolution_patterns'] = False
        config['use_fimo'] = False
        
        # Enable only CPP with specified n_filter
        config['include_cpp'] = True
        config['cpp_n_filter'] = cpp_n_filter
        
        return config
    
    def _process_outer_fold(self, train_outer: pd.DataFrame, test_outer: pd.DataFrame, 
                          fold_idx: int, model_type: str, use_group_based_split: bool) -> Dict[str, Any]:
        """Override to use custom feature configuration and extract 0.5 threshold performance."""
        
        print(f"   Train outer: {len(train_outer)} samples ({(train_outer['known_cleavage_site'] == 1).sum()} pos)")
        print(f"   Test outer: {len(test_outer)} samples ({(test_outer['known_cleavage_site'] == 1).sum()} pos)")

        # Step 1: Perform 3 inner splits (same as parent)
        all_inner_results = []
        all_hyperparameters = []
        all_best_thresholds = []
        
        # Create cross-validation splits (same logic as parent class)
        if use_group_based_split:
            from sklearn.model_selection import GroupKFold
            print("   Step 1: Perform 3 inner splits using GroupKFold (5-fold, use first 3)")
            inner_kfold = GroupKFold(n_splits=5)
            inner_groups = train_outer['entry']
            inner_splits = list(inner_kfold.split(train_outer, train_outer['known_cleavage_site'], groups=inner_groups))
        else:
            from sklearn.model_selection import StratifiedKFold
            print("   Step 1: Perform 3 inner splits using StratifiedKFold (5-fold, use first 3)")
            inner_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state + fold_idx)
            inner_splits = list(inner_kfold.split(train_outer, train_outer['known_cleavage_site']))
        
        # Process inner splits with CPP-only feature config
        for inner_split_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_splits[:3]):
            print(f"      Inner split {inner_split_idx + 1}/3:")
            
            train_inner = train_outer.iloc[train_inner_idx].reset_index(drop=True)
            val_inner = train_outer.iloc[val_inner_idx].reset_index(drop=True)
            
            print(f"         Train inner: {len(train_inner)} samples ({(train_inner['known_cleavage_site'] == 1).sum()} pos)")
            print(f"         Val inner: {len(val_inner)} samples ({(val_inner['known_cleavage_site'] == 1).sum()} pos)")

            # Step 2: Feature extraction with CPP-only config
            print(f"         Feature extraction (fit on train_inner) - CPP only (n_filter={self.cpp_n_filter})")
            feature_extractor = DataLoader(feature_config=self.feature_config, verbose=False)
            train_inner_features = feature_extractor.fit_transform(train_inner)
            val_inner_features = feature_extractor.transform(val_inner)
            
            # Continue with standard processing...
            print("         Create balanced subdatasets for train_inner")
            train_inner_data = train_inner.copy()
            inner_balanced_datasets = self._create_balanced_feature_datasets(train_inner_features, train_inner_data)
            print(f"         Created {len(inner_balanced_datasets)} balanced subdatasets")

            print("         Hyperparameter tuning (optimizing AUC)")
            best_hyperparameters = self._tune_hyperparameters_auc(
                inner_balanced_datasets, val_inner_features, model_type
            )
            
            print("         Threshold selection (optimizing F1)")
            val_inner_results = self._train_and_evaluate(
                inner_balanced_datasets, val_inner_features, 
                best_hyperparameters, model_type
            )

            print("         Select best thresholds from validation")
            best_thresholds = self._select_best_thresholds_only(val_inner_results)
            
            # Store results
            all_inner_results.append(val_inner_results)
            all_hyperparameters.append(best_hyperparameters)
            all_best_thresholds.append(best_thresholds)
            
            print(f"         Inner split {inner_split_idx + 1} completed")
        
        # Step 2: Average results from 3 inner splits
        print("   Step 2: Average results from 3 inner splits")
        final_hyperparameters = self._average_hyperparameters(all_hyperparameters)
        final_best_thresholds = self._average_best_thresholds(all_best_thresholds)
        
        print(f"         Final hyperparameters: {final_hyperparameters}")
        print(f"         Final best thresholds: {final_best_thresholds}")

        combined_val_inner_results = self._combine_inner_results(all_inner_results)
        
        # Step 3: Feature extraction for train_outer with CPP-only config
        print(f"   Step 3: Feature extraction for train_outer (refit on full train_outer) - CPP only (n_filter={self.cpp_n_filter})")
        train_outer_feature_extractor = DataLoader(feature_config=self.feature_config, verbose=False)
        train_outer_features = train_outer_feature_extractor.fit_transform(train_outer)
        test_outer_features = train_outer_feature_extractor.transform(test_outer)
        
        # Step 4: Generate balanced datasets for train_outer  
        print("   Step 4: Generate balanced datasets for train_outer")
        train_outer_data = train_outer.copy()
        train_outer_balanced_datasets = self._create_balanced_feature_datasets(train_outer_features, train_outer_data)
        print(f"         Created {len(train_outer_balanced_datasets)} balanced subdatasets")
        
        # Step 5: Test on test_outer with averaged best parameters
        print("   Step 5: Test on test_outer with averaged best parameters")
        test_outer_results = self._evaluate_on_test(
            train_outer_balanced_datasets, test_outer_features, 
            final_hyperparameters, model_type, final_best_thresholds
        )

        # Step 6: ADDITIONAL - Extract performance with 0.5 threshold
        print("   Step 6: Extract performance with 0.5 default threshold")
        test_outer_results_05 = self._evaluate_on_test_with_05_threshold(
            train_outer_balanced_datasets, test_outer_features, 
            final_hyperparameters, model_type
        )

        return {
            'fold_idx': fold_idx,
            'best_hyperparameters': final_hyperparameters,
            'best_thresholds': final_best_thresholds,
            'val_inner_results': combined_val_inner_results,
            'test_outer_results': test_outer_results,
            'test_outer_results_05': test_outer_results_05,  # NEW: 0.5 threshold results
            'all_inner_results': all_inner_results,
            'data_info': {
                'n_inner_splits': 3,
                'test_outer': len(test_outer),
                'n_subdatasets': len(train_outer_balanced_datasets),
                'cpp_n_filter': self.cpp_n_filter
            },
            'feature_config_used': self.feature_config
        }
    
    def _evaluate_on_test_with_05_threshold(self, balanced_datasets: List[Dict], test_features: Dict,
                                          hyperparameters: Dict, model_type: str) -> Dict[str, Any]:
        """Evaluate on test set using fixed 0.5 threshold instead of optimized thresholds."""
        
        # Only process logodds_average method
        avg_method = 'logodds_average'
        
        if avg_method not in hyperparameters:
            return {}
        
        print(f"         Evaluating with 0.5 threshold using {avg_method}")
        
        # Get hyperparameters for this averaging method
        method_hyperparams = hyperparameters[avg_method]
        
        # Train models on all balanced datasets
        trained_models = []
        for i, dataset in enumerate(balanced_datasets):
            X_train = dataset['X']
            y_train = dataset['y']
            
            # Create and train model with best hyperparameters
            model = self.model_factory.create_model(model_type, method_hyperparams['params'])
            model.fit(X_train, y_train)
            trained_models.append(model)
        
        # Make predictions on test set
        test_probabilities = []
        for model in trained_models:
            test_proba = model.predict_proba(test_features['X'])[:, 1]
            test_probabilities.append(test_proba)
        
        # Average probabilities using logodds averaging
        test_probabilities = np.array(test_probabilities)
        
        # Log-odds averaging
        epsilon = 1e-15
        test_probabilities = np.clip(test_probabilities, epsilon, 1 - epsilon)
        logodds = np.log(test_probabilities / (1 - test_probabilities))
        avg_logodds = np.mean(logodds, axis=0)
        avg_proba = 1 / (1 + np.exp(-avg_logodds))
        
        # Use fixed 0.5 threshold
        y_pred = (avg_proba >= 0.5).astype(int)
        y_true = test_features['y']
        
        # Calculate metrics with 0.5 threshold
        metrics = {
            'f1': f1_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, avg_proba),
            'threshold_used': 0.5
        }
        
        return {
            avg_method: {
                'soft_vote': {
                    'threshold_05': {
                        'test_metrics': metrics
                    }
                }
            }
        }


class CPPTrainer:
    """
    Grid search trainer for CPP features optimization using existing DataLoader architecture.
    
    This trainer tests different combinations of window sizes and CPP feature counts
    to find the optimal configuration for cleavage site prediction.
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Initialize the CPP trainer.
        
        Args:
            random_state: Random state for reproducibility
            verbose: Whether to print detailed progress information
        """
        self.random_state = random_state
        self.verbose = verbose
        self.results = {}
        self.best_config = None
        
        # Parameter grids - will be extracted from data
        self.window_sizes = None  # Will be extracted from DataFrame
        self.n_filters = [150, 100, 75, 50, 40, 30, 20, 10]
        
        # Initialize model factory
        self.model_factory = MLModelFactory()
        
        if self.verbose:
            print(" CPP FEATURE TRAINER INITIALIZED")
            print("=" * 60)
            print(f"   Window sizes: Will be extracted from data")
            print(f"   N_filter values to test: {len(self.n_filters)} configurations")
            print("=" * 60)
    
    def set_parameter_grid(self, window_sizes: List[int] = None, n_filters: List[int] = None):
        """
        Set custom parameter grids for optimization.
        
        Args:
            window_sizes: List of window sizes to test
            n_filters: List of n_filter values to test
        """
        if window_sizes is not None:
            self.window_sizes = window_sizes
        if n_filters is not None:
            self.n_filters = n_filters
        
        if self.verbose:
            print(f"Parameter grid updated:")
            print(f"   Window sizes: {self.window_sizes}")
            print(f"   N_filters: {self.n_filters}")
            if self.window_sizes is not None:
                print(f"   Total combinations: {len(self.window_sizes) * len(self.n_filters)}")
    
    def _create_cpp_only_config(self, cpp_n_filter: int) -> Dict[str, Any]:
        """
        Create feature configuration with only CPP features enabled.
        
        Args:
            cpp_n_filter: Number of CPP features to extract
            
        Returns:
            Feature configuration dictionary
        """
        config = get_feature_config()
        
        # Disable all features except CPP
        config['include_sequence_features'] = False
        config['include_structure_features'] = False
        config['include_plm_embeddings'] = False
        config['include_weighted_metrix'] = False
        config['include_cksaap'] = False
        config['include_coevolution_patterns'] = False
        config['use_fimo'] = False
        
        # Enable only CPP with specified n_filter
        config['include_cpp'] = True
        config['cpp_n_filter'] = cpp_n_filter
        
        return config
    
    def _extract_features_for_config(self, windows_df: pd.DataFrame, cpp_n_filter: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using DataLoader with CPP-only configuration.
        
        Args:
            windows_df: DataFrame with window sequences
            cpp_n_filter: Number of CPP features to extract
            
        Returns:
            Tuple of (X, y) - features and labels
        """
        # Create CPP-only configuration
        cpp_config = self._create_cpp_only_config(cpp_n_filter)
        
        # Initialize DataLoader with CPP-only config
        data_loader = DataLoader(feature_config=cpp_config, verbose=False)
        
        # Extract features using existing pipeline
        features = data_loader.fit_transform(windows_df)
        
        # Get feature matrix and labels
        X = features['X']  # Feature matrix
        y = windows_df['known_cleavage_site'].values
        
        return X, y
    
    def _analyze_data_distribution(self, cs_windows: pd.DataFrame) -> None:
        """
        Analyze the data distribution across window sizes and extract window sizes.
        
        Args:
            cs_windows: DataFrame containing windows with different lengths
        """
        if 'window_size' not in cs_windows.columns:
            print(" Error: 'window_size' column not found in DataFrame")
            return
        
        # Extract unique window sizes from the data
        self.window_sizes = sorted(cs_windows['window_size'].unique())
        
        if self.verbose:
            print(" DATA DISTRIBUTION ANALYSIS")
            print("=" * 80)
            print(f"   Total windows in dataset: {len(cs_windows)}")
            print(f"   Window sizes found: {self.window_sizes}")
            print(f"   Total combinations to test: {len(self.window_sizes)} × {len(self.n_filters)} = {len(self.window_sizes) * len(self.n_filters)}")
            print()
            print("   Window Size Distribution:")
            print("   " + "-" * 50)
            print(f"   {'Size':<6} {'Total':<8} {'Positive':<10} {'Negative':<10} {'Pos%':<8}")
            print("   " + "-" * 50)
            
            total_pos = 0
            total_neg = 0
            
            for window_size in self.window_sizes:
                size_data = cs_windows[cs_windows['window_size'] == window_size]
                n_total = len(size_data)
                n_pos = (size_data['known_cleavage_site'] == 1).sum()
                n_neg = n_total - n_pos
                pos_pct = (n_pos / n_total * 100) if n_total > 0 else 0
                
                total_pos += n_pos
                total_neg += n_neg
                
                print(f"   {window_size:<6} {n_total:<8} {n_pos:<10} {n_neg:<10} {pos_pct:<8.1f}%")
            
            print("   " + "-" * 50)
            print(f"   {'Total':<6} {total_pos + total_neg:<8} {total_pos:<10} {total_neg:<10} {(total_pos/(total_pos + total_neg)*100):<8.1f}%")
            print("=" * 80)
    
    def _evaluate_configuration(self, windows_df: pd.DataFrame, 
                               window_size: int, n_filter: int,
                               model_type: str = 'SVM', outer_folds: int = 5,
                               use_group_based_split: bool = True) -> Dict[str, float]:
        """
        Evaluate a specific window_size and n_filter configuration using proper trainer.
        
        Args:
            windows_df: DataFrame with windows for this specific window size
            window_size: Window size used
            n_filter: Number of CPP features
            model_type: Model type to use ('SVM', 'RandomForest', 'XGBoost')
            outer_folds: Number of CV folds
            use_group_based_split: Whether to use group-based CV
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Create specialized trainer for this configuration
        cpp_trainer = CPPFeatureTrainer(cpp_n_filter=n_filter, random_state=self.random_state)
        
        # Run full training pipeline
        results = cpp_trainer.run(
            data=windows_df,
            use_group_based_split=use_group_based_split,
            model_type=model_type,
            outer_folds=outer_folds
        )
        
        # Extract 0.5 threshold performance from aggregated results
        performance_05 = self._extract_05_threshold_performance(results)
        
        return {
            'window_size': window_size,
            'n_filter': n_filter,
            'n_samples': len(windows_df),
            'n_positive': (windows_df['known_cleavage_site'] == 1).sum(),
            'n_negative': (windows_df['known_cleavage_site'] == 0).sum(),
            # 0.5 threshold performance (what we want)
            'f1_05_mean': performance_05.get('f1_mean', 0),
            'f1_05_std': performance_05.get('f1_std', 0),
            'balanced_accuracy_05_mean': performance_05.get('balanced_accuracy_mean', 0),
            'balanced_accuracy_05_std': performance_05.get('balanced_accuracy_std', 0),
            'auc_05_mean': performance_05.get('auc_mean', 0),
            'auc_05_std': performance_05.get('auc_std', 0),
        }
    
    def _extract_05_threshold_performance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract 0.5 threshold performance from trainer results."""
        if 'aggregated_results' not in results:
            return {}
        
        aggregated = results['aggregated_results']
        
        # Look for 0.5 threshold results in logodds_average method
        if ('logodds_average' in aggregated and 
            'soft_vote' in aggregated['logodds_average'] and
            'fixed_0.5' in aggregated['logodds_average']['soft_vote']):
            
            metrics = aggregated['logodds_average']['soft_vote']['fixed_0.5']['test_metrics']
            return {
                'f1_mean': metrics.get('f1', {}).get('mean', 0),
                'f1_std': metrics.get('f1', {}).get('std', 0),
                'balanced_accuracy_mean': metrics.get('balanced_accuracy', {}).get('mean', 0),
                'balanced_accuracy_std': metrics.get('balanced_accuracy', {}).get('std', 0),
                'auc_mean': metrics.get('auc', {}).get('mean', 0),
                'auc_std': metrics.get('auc', {}).get('std', 0),
            }
        
        return {}
    
    def train_and_evaluate(self, cs_windows: pd.DataFrame) -> Dict[str, Any]:
        """
        Train and evaluate all combinations of window sizes and n_filters.
        
        Args:
            cs_windows: DataFrame containing windows with different lengths
                       Must have columns: 'sequence', 'known_cleavage_site', 'window_size'
        
        Returns:
            Dictionary containing all results and best configuration
        """
        # Step 1: Analyze data distribution and extract window sizes
        self._analyze_data_distribution(cs_windows)
        
        if self.window_sizes is None or len(self.window_sizes) == 0:
            print(" No valid window sizes found in the data")
            return {}
        
        if self.verbose:
            print(" STARTING CPP FEATURE GRID SEARCH")
        
        # Store all results
        all_results = []
        best_f1 = 0
        best_config = None
        
        # Grid search over all combinations
        total_combinations = len(self.window_sizes) * len(self.n_filters)
        current_combination = 0
        
        for window_size in self.window_sizes:
            print('***###'*40)
            print(f'Training window_size: {window_size}')
            # Filter windows for this specific window size
            size_windows = cs_windows[cs_windows['window_size'] == window_size].copy().reset_index(drop=True)
            
            if len(size_windows) == 0:
                if self.verbose:
                    print(f" No windows found for window size {window_size}, skipping...")
                continue
            
            if self.verbose:
                print(f"\n Processing window size: {window_size}")
                print(f"   Available windows: {len(size_windows)}")
                print(f"   Positive: {(size_windows['known_cleavage_site'] == 1).sum()}")
                print(f"   Negative: {(size_windows['known_cleavage_site'] == 0).sum()}")
            
            # Check if we have both classes
            if size_windows['known_cleavage_site'].nunique() < 2:
                if self.verbose:
                    print(f" Only one class found for window size {window_size}, skipping...")
                continue
            
            for n_filter in self.n_filters:
                current_combination += 1
                
                if self.verbose:
                    progress = (current_combination / total_combinations) * 100
                    print('***###'*20)
                    print(f"    Testing n_filter={n_filter} ({current_combination}/{total_combinations}, {progress:.1f}%)")
                
                try:
                    # Evaluate this configuration using full trainer pipeline
                    result = self._evaluate_configuration(
                        size_windows, window_size, n_filter,
                        model_type='SVM',  # Use SVM for consistency
                        outer_folds=5,
                        use_group_based_split=True
                    )
                    all_results.append(result)
                    
                    # Track best configuration based on 0.5 threshold F1 score
                    if result['f1_05_mean'] > best_f1:
                        best_f1 = result['f1_05_mean']
                        best_config = result.copy()
                    
                    if self.verbose:
                        print(f"      F1(0.5): {result['f1_05_mean']:.3f}±{result['f1_05_std']:.3f}, "
                              f"Bal_Acc(0.5): {result['balanced_accuracy_05_mean']:.3f}±{result['balanced_accuracy_05_std']:.3f}, "
                              f"AUC(0.5): {result['auc_05_mean']:.3f}±{result['auc_05_std']:.3f}")
                
                except Exception as e:
                    print(f"    Error processing window_size={window_size}, n_filter={n_filter}: {str(e)}")
                    continue
        
        # Store results
        self.results = {
            'all_results': all_results,
            'best_config': best_config,
            'parameter_grid': {
                'window_sizes': self.window_sizes,
                'n_filters': self.n_filters
            },
            'dataset_info': {
                'total_windows': len(cs_windows),
                'n_positive': (cs_windows['known_cleavage_site'] == 1).sum(),
                'n_negative': (cs_windows['known_cleavage_site'] == 0).sum(),
                'unique_window_sizes': sorted(cs_windows['window_size'].unique().tolist())
            }
        }
        self.best_config = best_config
        
        return self.results
    
    
    def create_heatmap(self, metric: str = 'balanced_accuracy_05_mean', save_path: str = None, 
                      figsize: Tuple[int, int] = (14, 8), percentage: bool = True, 
                      title: str = None):
        """
        Create a heatmap visualization.
        
        Args:
            metric: Metric to visualize ('f1_05_mean', 'balanced_accuracy_05_mean', 'auc_05_mean')
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            percentage: Whether to display values as percentages
            title: Custom title for the plot (optional)
        """
        if not self.results or not self.results['all_results']:
            print(" No results available for plotting. Run train_and_evaluate() first.")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results['all_results'])
        
        # Create pivot table for heatmap
        heatmap_data = results_df.pivot_table(
            index='n_filter', 
            columns='window_size', 
            values=metric, 
            aggfunc='mean'
        )
        
        # Sort indices to match the provided image style
        heatmap_data = heatmap_data.sort_index(ascending=False)  # n_filter high to low
        heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))  # window_size low to high
        
        # Convert to percentage if requested
        if percentage:
            heatmap_data = heatmap_data * 100
        
        # Create the plot to match the provided image style
        plt.figure(figsize=figsize)
        
        # Create heatmap with similar styling to the provided image
        fmt_str = '.0f' if percentage else '.2f'
        
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=fmt_str,
            cmap='RdYlGn',  # Red-Yellow-Green colormap like the image
            center=None,
            cbar_kws={
                'label': f'{self._get_metric_label(metric)} {"(%)" if percentage else ""}',
                'shrink': 0.8
            },
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'size': 14, 'weight': 'bold'},  # Increased from 8 to 14
            square=False
        )
        
        # Get colorbar and increase its label font size
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # Colorbar tick labels
        cbar.set_label(f'{self._get_metric_label(metric)} {"(%)" if percentage else ""}', 
                      size=16, weight='bold')  # Colorbar title
        
        # Customize the plot title
        if title:
            plt.title(title, fontsize=18, fontweight='bold', pad=25)  # Increased from 14 to 18
        else:
            plt.title(f'Evaluation CPP features ({self._get_metric_label(metric)})', 
                     fontsize=18, fontweight='bold', pad=25)  # Increased from 14 to 18
        plt.xlabel('Window Size', fontsize=16, fontweight='bold')  # Increased from 12 to 16
        plt.ylabel('Number of features (top n)', fontsize=16, fontweight='bold')  # Increased from 12 to 16
        
        # Improve tick labels
        plt.xticks(rotation=0, fontsize=14, weight='bold')  # Increased from 10 to 14, added bold
        plt.yticks(rotation=0, fontsize=14, weight='bold')  # Increased from 10 to 14, added bold
        
        # No markers - keep it as a clean heatmap
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" Heatmap saved to: {save_path}")
        
        plt.show()
        
        # Print additional information
        if self.verbose:
            print(f"\n HEATMAP STATISTICS:")
            print(f"   Metric: {self._get_metric_label(metric)}")
            best_score = heatmap_data.max().max()
            worst_score = heatmap_data.min().min()
            mean_score = heatmap_data.mean().mean()
            
            if percentage:
                print(f"   Best score: {best_score:.1f}%")
                print(f"   Worst score: {worst_score:.1f}%")
                print(f"   Mean score: {mean_score:.1f}%")
            else:
                print(f"   Best score: {best_score:.4f}")
                print(f"   Worst score: {worst_score:.4f}")
                print(f"   Mean score: {mean_score:.4f}")
            
            if self.best_config:
                print(f"   Best config: Window Size {self.best_config['window_size']}, N_Filter {self.best_config['n_filter']}")
    
    def _get_metric_label(self, metric: str) -> str:
        """Get human-readable label for metric."""
        labels = {
            # 0.5 threshold metrics (primary)
            'f1_05_mean': 'F1 Score',
            'balanced_accuracy_05_mean': 'Balanced Accuracy', 
            'auc_05_mean': 'AUC',
            # Optimized threshold metrics (for reference)
            'f1_opt_mean': 'F1 Score (Optimized)',
            'balanced_accuracy_opt_mean': 'Balanced Accuracy (Optimized)',
            'auc_opt_mean': 'AUC (Optimized)',
            # Legacy support
            'f1_cv_mean': 'F1 Score',
            'balanced_accuracy_cv_mean': 'Balanced Accuracy',
            'auc_cv_mean': 'AUC',
        }
        return labels.get(metric, metric)
    