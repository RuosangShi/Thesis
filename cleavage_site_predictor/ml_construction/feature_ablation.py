#!/usr/bin/env python3
"""
Feature Ablation Study Module
=============================

This module implements systematic feature group removal to understand 
the contribution of different feature types to cleavage site prediction.

Key Features:
1. Progressive feature group removal
2. Performance comparison across different feature combinations
3. Statistical significance testing
4. Visualization of feature importance
5. Group-based cross-validation support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import copy
import warnings
warnings.filterwarnings('ignore')

from .trainer import EnsembleTrainer
from .config import get_feature_config
from .data_loader import DataLoader
from ._util_ablation import plot_progressive_ablation_results


class FeatureAblationTrainer(EnsembleTrainer):
    """
    Extended EnsembleTrainer that supports custom feature configurations for ablation studies.
    """
    
    def __init__(self, feature_config: Dict[str, Any], random_state=42):
        """
        Initialize trainer with custom feature configuration.
        
        Args:
            feature_config: Custom feature configuration for ablation
            random_state: Random state for reproducibility
        """
        super().__init__(random_state=random_state)
        self.custom_feature_config = feature_config
        
        enabled_features = [k for k, v in feature_config.items() if isinstance(v, bool) and v]
        disabled_features = [k for k, v in feature_config.items() if isinstance(v, bool) and not v]
        print(f"   Custom config: {len(enabled_features)} enabled, {len(disabled_features)} disabled")
    
    def _process_outer_fold(self, train_outer: pd.DataFrame, test_outer: pd.DataFrame, 
                          fold_idx: int, model_type: str, use_group_based_split: bool) -> Dict[str, Any]:
        """Override to use custom feature configuration in data loading."""
        
        print(f"   Train outer: {len(train_outer)} samples ({(train_outer['known_cleavage_site'] == 1).sum()} pos)")
        print(f"   Test outer: {len(test_outer)} samples ({(test_outer['known_cleavage_site'] == 1).sum()} pos)")

        # Step 1: Perform 3 inner splits
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
        
        # Process inner splits with custom feature config
        for inner_split_idx, (train_inner_idx, val_inner_idx) in enumerate(inner_splits[:3]):
            print(f"      Inner split {inner_split_idx + 1}/3:")
            
            train_inner = train_outer.iloc[train_inner_idx].reset_index(drop=True)
            val_inner = train_outer.iloc[val_inner_idx].reset_index(drop=True)
            
            print(f"         Train inner: {len(train_inner)} samples ({(train_inner['known_cleavage_site'] == 1).sum()} pos)")
            print(f"         Val inner: {len(val_inner)} samples ({(val_inner['known_cleavage_site'] == 1).sum()} pos)")

            # Step 2: Feature extraction with CUSTOM CONFIG
            print("         Feature extraction (fit on train_inner) - USING CUSTOM CONFIG")
            feature_extractor = DataLoader(feature_config=self.custom_feature_config, verbose=False)
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
            
            print("         Counts/Threshold selection (optimizing PS1)")
            val_inner_results = self._train_and_evaluate(
                inner_balanced_datasets, val_inner_features, 
                best_hyperparameters, model_type
            )

            print("         Using fixed threshold 0.5 for ablation study")
            best_thresholds = self._get_fixed_thresholds()
            
            # Store results
            all_inner_results.append(val_inner_results)
            all_hyperparameters.append(best_hyperparameters)
            all_best_thresholds.append(best_thresholds)
            
            print(f"         Inner split {inner_split_idx + 1} completed")
        
        # Step 2: Average results from 3 inner splits (same as parent)
        print("   Step 2: Average results from 3 inner splits")
        final_hyperparameters = self._average_hyperparameters(all_hyperparameters)
        final_best_thresholds = self._average_best_thresholds(all_best_thresholds)
        
        print(f"         Final hyperparameters: {final_hyperparameters}")
        print(f"         Final best thresholds: {final_best_thresholds}")

        combined_val_inner_results = self._combine_inner_results(all_inner_results)
        
        # Step 3: Feature extraction for train_outer with CUSTOM CONFIG
        print("   Step 3: Feature extraction for train_outer (refit on full train_outer) - USING CUSTOM CONFIG")
        train_outer_feature_extractor = DataLoader(feature_config=self.custom_feature_config, verbose=False)
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
            },
            'feature_config_used': self.custom_feature_config
        }
    
    def _get_fixed_thresholds(self) -> Dict[str, Any]:
        """
        Return fixed threshold of 0.5 for ablation studies instead of optimizing.
        
        Returns:
            Dictionary with fixed thresholds for each averaging method
        """
        fixed_thresholds = {}
        averaging_methods = ['logodds_average']
        
        for avg_method in averaging_methods:
            fixed_thresholds[avg_method] = {
                'threshold': 0.5,
                'score': 0.0,  # Not applicable for fixed threshold
                'metric': 'fixed'
            }
        
        return fixed_thresholds


class FeatureAblationStudy:
    """
    Systematic feature ablation study for cleavage site prediction.
    
    This class implements progressive removal of feature groups to understand
    their individual and combined contributions to model performance.
    """
    
    def __init__(self, random_state=42, verbose=True):
        """
        Initialize the feature ablation study.
        
        Args:
            random_state: Random state for reproducibility
            verbose: Whether to print detailed progress information
        """
        self.random_state = random_state
        self.verbose = verbose
        
        # Define feature groups based on the config structure
        self.feature_groups = self._define_feature_groups()
        
        # Storage for results
        self.ablation_results = {}
        self.performance_summary = {}
        
        if self.verbose:
            print(" FEATURE ABLATION STUDY INITIALIZED")
            print("=" * 60)
            self._print_feature_groups()
    
    def _define_feature_groups(self) -> Dict[str, Dict[str, Any]]:
        """
        Define feature groups for systematic ablation based on actual implementation.
        
        Node Features: 2,456 dimensions per residue → mean pooled to window level
        Global Features: 1,764 dimensions per window
        Total: 4,220 dimensions per window
        
        Returns:
            Dictionary with feature group definitions matching actual implementation
        """
        feature_groups = {
            'sequence_features': {
                'name': 'Sequence-based Node Features',
                'description': 'PAM/BLOSUM profiles (10) + AAindex physicochemical descriptors (118), mean-pooled across 20 residues',
                'config_keys': ['include_sequence_features'],
                'estimated_dims': 128,  # 128 per residue, pooled to window level
                'feature_type': 'node'
            },
            'structure_features': {
                'name': 'Structure-based Node Features', 
                'description': 'Conformational angles (10), distances (2), RSA (1), secondary structure (8), flexibility (1), depth (1)',
                'config_keys': ['include_structure_features'],
                'estimated_dims': 23,  # 23 per residue, pooled to window level
                'feature_type': 'node'
            },
            'plm_embeddings': {
                'name': 'Protein Language Model Embeddings',
                'description': 'Combined ESM-2 (1,280 dims) + ProtT5 (1,024 dims) embeddings per residue, mean-pooled to window level',
                'config_keys': ['include_plm_embeddings'],
                'estimated_dims': 2304,  # 1,280 + 1,024 per residue, pooled to window level
                'feature_type': 'node'
            },
            'weighted_matrices': {
                'name': 'Weighted Matrices',
                'description': 'Position weight matrices and PSSM features (14 dims per window)',
                'config_keys': ['include_weighted_metrix'],
                'estimated_dims': 14,
                'feature_type': 'global'
            },
            'cksaap_composition': {
                'name': 'CKSAAP Composition Features',
                'description': 'Composition of k-spaced amino acid pairs (1,600 dims per window)',
                'config_keys': ['include_cksaap'],
                'estimated_dims': 1600,
                'feature_type': 'global'
            },
            'cpp_patterns': {
                'name': 'CPP Features', 
                'description': 'Conjoint triad patterns (100 dims per window)',
                'config_keys': ['include_cpp'],
                'estimated_dims': 100,
                'feature_type': 'global'
            },
            'coevolution_patterns': {
                'name': 'Coevolutionary Patterns',
                'description': 'Evolutionary covariation patterns (50 dims per window)',
                'config_keys': ['include_coevolution_patterns'],
                'estimated_dims': 50,
                'feature_type': 'global'
            }
            # Note: FIMO features are intentionally excluded as use_fimo=False in config
        }
        
        return feature_groups
    
    def _print_feature_groups(self):
        """Print defined feature groups."""
        print(" DEFINED FEATURE GROUPS:")
        total_dims = 0
        for group_id, group_info in self.feature_groups.items():
            dims = group_info['estimated_dims']
            total_dims += dims
            print(f"   {group_info['name']}: {dims} dims")
            print(f"      {group_info['description']}")
        
        print(f"\n Total estimated dimensions: {total_dims}")
        print("=" * 60)
    
    def run_ablation_study(self, 
                          data: pd.DataFrame, 
                          model_type: str = 'SVM',
                          use_group_based_split: bool = True,
                          outer_folds: int = 5,
                          study_type: str = 'progressive') -> Dict[str, Any]:
        """
        Run the complete feature ablation study.
        
        Args:
            data: Input DataFrame with sequence windows
            model_type: ML model to use ('SVM', 'RandomForest', 'XGBoost')
            use_group_based_split: Whether to use protein-based CV splits
            outer_folds: Number of outer CV folds
            study_type: Type of study ('progressive', 'individual', 'combinatorial')
        
        Returns:
            Dictionary containing comprehensive ablation results
        """
        
        if study_type == 'progressive':
            results = self._run_progressive_ablation(
                data, model_type, use_group_based_split, outer_folds
            )
        elif study_type == 'individual':
            results = self._run_individual_ablation(
                data, model_type, use_group_based_split, outer_folds
            )
        elif study_type == 'combinatorial':
            results = self._run_combinatorial_ablation(
                data, model_type, use_group_based_split, outer_folds
            )
        else:
            raise ValueError(f"Unknown study type: {study_type}")
        
        # Generate comprehensive summary
        self.performance_summary = self._generate_performance_summary(results)
        
        
        return {
            'ablation_results': results,
            'performance_summary': self.performance_summary,
            'feature_groups': self.feature_groups,
            'study_config': {
                'study_type': study_type,
                'model_type': model_type,
                'use_group_based_split': use_group_based_split,
                'outer_folds': outer_folds,
                'n_samples': len(data),
                'n_positive': (data['known_cleavage_site'] == 1).sum()
            }
        }
    
    def _run_progressive_ablation(self, data: pd.DataFrame, model_type: str, 
                                use_group_based_split: bool, outer_folds: int) -> Dict[str, Any]:
        """
        Run progressive ablation: start with all features, remove one group at a time.
        """
        print("\n PROGRESSIVE FEATURE ABLATION")
        print("   Strategy: Start with ALL features, remove one group at a time")
        
        results = {}
        
        # Step 0a: AA index only baseline
        print("\n Step 0a: AA index only baseline")
        aa_index_config = self._create_single_feature_config('sequence_features')
        aa_index_results = self._train_with_config(
            data, aa_index_config, model_type, use_group_based_split, outer_folds, 'baseline_aa_index'
        )
        results['baseline_aa_index'] = aa_index_results
        
        # Step 0b: CPP only baseline
        print("\n Step 0b: CPP only baseline")
        cpp_config = self._create_single_feature_config('cpp_patterns')  
        cpp_results = self._train_with_config(
            data, cpp_config, model_type, use_group_based_split, outer_folds, 'baseline_cpp'
        )
        results['baseline_cpp'] = cpp_results
        
        # Step 0c: PLM only baseline
        print("\n Step 0c: PLM only baseline")
        plm_config = self._create_single_feature_config('plm_embeddings')
        plm_results = self._train_with_config(
            data, plm_config, model_type, use_group_based_split, outer_folds, 'baseline_plm'
        )
        results['baseline_plm'] = plm_results
        
        
        # Step 1: Full model with all features
        print("\n Step 1: Full model with ALL features")
        full_model_config = self._create_feature_config(remove_groups=[])
        full_model_results = self._train_with_config(
            data, full_model_config, model_type, use_group_based_split, outer_folds, 'full_model'
        )
        results['full_model'] = full_model_results
        
        # Step 2: Remove each group individually
        print(f"\n Step 2: Remove each feature group individually ({len(self.feature_groups)} experiments)")
        
        for group_id, group_info in self.feature_groups.items():
            print(f"\n   Removing: {group_info['name']}")
            
            # Create config with this group removed
            ablation_config = self._create_feature_config(remove_groups=[group_id])
            
            # Train and evaluate
            group_results = self._train_with_config(
                data, ablation_config, model_type, use_group_based_split, 
                outer_folds, f'remove_{group_id}'
            )
            
            results[f'remove_{group_id}'] = group_results
        
        return results
    
    def _run_individual_ablation(self, data: pd.DataFrame, model_type: str,
                                use_group_based_split: bool, outer_folds: int) -> Dict[str, Any]:
        """
        Run individual ablation: use only one feature group at a time.
        """
        print("\n INDIVIDUAL FEATURE ABLATION")
        print("   Strategy: Use ONLY one feature group at a time")
        
        results = {}
        
        # Test each group individually
        for group_id, group_info in self.feature_groups.items():
            print(f"\n   Using ONLY: {group_info['name']}")
            
            # Create config with only this group enabled
            all_other_groups = [g for g in self.feature_groups.keys() if g != group_id]
            ablation_config = self._create_feature_config(remove_groups=all_other_groups)
            
            # Train and evaluate
            group_results = self._train_with_config(
                data, ablation_config, model_type, use_group_based_split,
                outer_folds, f'only_{group_id}'
            )
            
            results[f'only_{group_id}'] = group_results
        
        return results
    
    def _run_combinatorial_ablation(self, data: pd.DataFrame, model_type: str,
                                  use_group_based_split: bool, outer_folds: int) -> Dict[str, Any]:
        """
        Run combinatorial ablation: test important combinations.
        """
        print(" COMBINATORIAL FEATURE ABLATION")
        print("   Strategy: Test important feature combinations")
        
        results = {}
        
        # Define important combinations to test (using actual feature group keys)
        combinations = {
            'sequence_structure': ['sequence_features', 'structure_features'],
            'sequence_plm': ['sequence_features', 'plm_embeddings'],
            'structure_plm': ['structure_features', 'plm_embeddings'],
            'core_features': ['sequence_features', 'structure_features', 'plm_embeddings'],
            'traditional_features': ['sequence_features', 'structure_features', 'cksaap_composition', 'cpp_patterns'],
            'modern_features': ['plm_embeddings', 'coevolution_patterns'],
            'composition_only': ['cksaap_composition', 'cpp_patterns'],
            'minimal_effective': ['sequence_features', 'structure_features']  # Minimal but potentially effective
        }
        
        for combo_name, included_groups in combinations.items():
            print(f"\n   Testing combination: {combo_name}")
            print(f"      Included groups: {[self.feature_groups[g]['name'] for g in included_groups]}")
            
            # Create config with only these groups enabled
            excluded_groups = [g for g in self.feature_groups.keys() if g not in included_groups]
            ablation_config = self._create_feature_config(remove_groups=excluded_groups)
            
            # Train and evaluate
            combo_results = self._train_with_config(
                data, ablation_config, model_type, use_group_based_split,
                outer_folds, f'combo_{combo_name}'
            )
            
            results[f'combo_{combo_name}'] = combo_results
        
        return results
    
    def _create_feature_config(self, remove_groups: List[str]) -> Dict[str, Any]:
        """
        Create a feature configuration with specified groups removed.
        
        Args:
            remove_groups: List of feature group IDs to remove
        
        Returns:
            Modified feature configuration
        """
        # Start with default config
        config = get_feature_config()
        
        # IMPORTANT: Always keep FIMO disabled during ablation studies
        config['use_fimo'] = False
        
        # Remove specified feature groups
        for group_id in remove_groups:
            if group_id in self.feature_groups:
                group_info = self.feature_groups[group_id]
                for config_key in group_info['config_keys']:
                    if config_key in config:
                        config[config_key] = False
        
        return config
    
    def _create_single_feature_config(self, feature_group_id: str) -> Dict[str, Any]:
        """
        Create a configuration using only a single feature group.
        
        Args:
            feature_group_id: ID of the single feature group to enable
        
        Returns:
            Feature configuration with only the specified group enabled
        """
        # Start with default config but disable all features
        config = get_feature_config()
        
        # Disable everything first
        config['use_fimo'] = False
        config['include_sequence_features'] = False
        config['include_structure_features'] = False
        config['include_plm_embeddings'] = False
        config['include_weighted_metrix'] = False
        config['include_cksaap'] = False
        config['include_cpp'] = False
        config['include_coevolution_patterns'] = False
        
        # Enable only the specified feature group
        if feature_group_id in self.feature_groups:
            group_info = self.feature_groups[feature_group_id]
            for config_key in group_info['config_keys']:
                if config_key in config:
                    config[config_key] = True
        
        return config
    
    
    def _train_with_config(self, data: pd.DataFrame, feature_config: Dict[str, Any],
                          model_type: str, use_group_based_split: bool, outer_folds: int,
                          experiment_name: str) -> Dict[str, Any]:
        """
        Train model with specific feature configuration.
        """
        if self.verbose:
            enabled_features = [k for k, v in feature_config.items() 
                              if isinstance(v, bool) and v]
            disabled_features = [k for k, v in feature_config.items()
                               if isinstance(v, bool) and not v]
            print(f"      Training with {len(enabled_features)} enabled, {len(disabled_features)} disabled features...")
            if disabled_features:
                print(f"      Disabled: {', '.join(disabled_features)}")
        
        # Create trainer with custom feature config
        trainer = FeatureAblationTrainer(
            feature_config=feature_config,
            random_state=self.random_state
        )
        
        # Run training with the custom configuration
        results = trainer.run(
            data=data,
            use_group_based_split=use_group_based_split,
            model_type=model_type,
            outer_folds=outer_folds
        )
        
        # Add metadata about the experiment
        results['experiment_info'] = {
            'name': experiment_name,
            'feature_config': feature_config,
            'enabled_features': [k for k, v in feature_config.items() 
                               if isinstance(v, bool) and v],
            'disabled_features': [k for k, v in feature_config.items()
                                if isinstance(v, bool) and not v]
        }
        
        return results
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary across all experiments.
        """
        print("\n GENERATING PERFORMANCE SUMMARY...")
        
        summary = {
            'experiments': {},
            'rankings': {},
            'feature_importance': {},
            'statistical_analysis': {}
        }
        
        # Extract key metrics for each experiment
        for exp_name, exp_results in results.items():
            if 'aggregated_results' in exp_results:
                # Extract performance metrics
                exp_summary = self._extract_experiment_metrics(exp_results)
                summary['experiments'][exp_name] = exp_summary
        
        # Generate rankings
        summary['rankings'] = self._generate_rankings(summary['experiments'])
        
        # Calculate feature importance
        summary['feature_importance'] = self._calculate_feature_importance(results)
        
        return summary
    
    def _extract_experiment_metrics(self, exp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from experiment results."""
        aggregated = exp_results['aggregated_results']
        
        metrics_summary = {}
        
        # Only process logodds_average method
        if 'logodds_average' in aggregated:
            avg_method = 'logodds_average'
            method_summary = {}
            
            # Extract best performing combination
            best_f1 = 0
            best_combo = None
            
            # Only process soft_vote method
            if 'soft_vote' in aggregated[avg_method]:
                vote_method = 'soft_vote'
                for combo_key, combo_data in aggregated[avg_method][vote_method].items():
                    f1_mean = combo_data['test_metrics']['f1']['mean']
                    if f1_mean > best_f1:
                        best_f1 = f1_mean
                        best_combo = {
                            'vote_method': vote_method,
                            'combo_key': combo_key,
                            'metrics': combo_data['test_metrics']
                                }
                
                method_summary['best_combination'] = best_combo
                metrics_summary[avg_method] = method_summary
        
        return metrics_summary
    
    def _generate_rankings(self, experiments: Dict[str, Any]) -> Dict[str, List]:
        """Generate performance rankings."""
        rankings = {
            'f1_score': [],
            'auc_score': [],
            'ps1_score': []
        }
        
        # Collect scores for ranking
        exp_scores = []
        for exp_name, exp_data in experiments.items():
            if 'logodds_average' in exp_data:
                best_combo = exp_data['logodds_average'].get('best_combination')
                if best_combo:
                    metrics = best_combo['metrics']
                    exp_scores.append({
                        'experiment': exp_name,
                        'f1': metrics['f1']['mean'],
                        'auc': metrics['auc']['mean'],
                        'ps1': metrics['ps1']['mean'] if 'ps1' in metrics else 0.0  # ps1 may not exist in streamlined version
                    })
        
        # Sort by each metric
        for metric in ['f1', 'auc', 'ps1']:
            sorted_scores = sorted(exp_scores, key=lambda x: x[metric], reverse=True)
            rankings[f'{metric}_score'] = [(exp['experiment'], exp[metric]) for exp in sorted_scores]
        
        return rankings
    
    def _calculate_feature_importance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relative importance of each feature group."""
        importance = {}
        
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        if 'full_model' in results:
            full_model_f1 = self._extract_best_f1(results['full_model'])
            
            for group_id, group_info in self.feature_groups.items():
                remove_key = f'remove_{group_id}'
                if remove_key in results:
                    removed_f1 = self._extract_best_f1(results[remove_key])
                    
                    # Importance = performance drop when removed
                    importance_score = full_model_f1 - removed_f1
                    importance[group_id] = {
                        'name': group_info['name'],
                        'importance_score': importance_score,
                        'full_model_f1': full_model_f1,
                        'removed_f1': removed_f1,
                        'performance_drop': importance_score
                    }
        
        return importance
    
    def _extract_best_f1(self, results: Dict[str, Any]) -> float:
        """Extract best F1 score from experiment results."""
        if 'aggregated_results' not in results:
            return 0.0
        
        aggregated = results['aggregated_results']
        best_f1 = 0.0
        
        # Only process logodds_average method
        if 'logodds_average' in aggregated:
            avg_method = 'logodds_average'
            # Only process soft_vote method
            if 'soft_vote' in aggregated[avg_method]:
                vote_method = 'soft_vote'
                for combo_data in aggregated[avg_method][vote_method].values():
                    f1_mean = combo_data['test_metrics']['f1']['mean']
                    best_f1 = max(best_f1, f1_mean)
        
        return best_f1
    
    def plot_progressive_ablation_results(self, progressive_results: Dict[str, Any], 
                                         save_path: Optional[str] = None, title: Optional[str] = 'Progressive Feature Ablation: Impact of Removing Each Feature Group\n Baseline (All Features) vs. Remove One Group at a Time'):
        """
        Create comprehensive plots for progressive ablation results using external utility.
        """
        plot_progressive_ablation_results(progressive_results, save_path, title)
    
    def extract_results_to_csv(self, progressive_results: Dict[str, Any], save_path: str = "ablation_results.csv") -> pd.DataFrame:
        """
        Extract ablation results from progressive_results and save to CSV.
        
        Args:
            progressive_results: Your progressive ablation results dictionary
            save_path: Path to save the CSV file
        
        Returns:
            pandas.DataFrame: DataFrame containing all results
        """
        
        # Extract experiments from performance_summary
        if 'performance_summary' not in progressive_results or 'experiments' not in progressive_results['performance_summary']:
            print(" No experiments found in progressive_results['performance_summary']['experiments']")
            return None
        
        experiments = progressive_results['performance_summary']['experiments']
        
        # List to store all experiment data
        results_data = []
        
        # Metrics to extract
        metrics = ['f1', 'auc', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'ps1', 'accuracy', 'mcc']
        
        print(f" Extracting results from {len(experiments)} experiments...")
        
        for exp_name, exp_data in experiments.items():
            print(f"   Processing: {exp_name}")
            
            # Initialize row data
            row_data = {
                'experiment': exp_name,
                'experiment_type': self._classify_experiment(exp_name)
            }
            
            # Extract metrics from logodds_average -> best_combination -> metrics
            if ('logodds_average' in exp_data and 
                'best_combination' in exp_data['logodds_average'] and
                'metrics' in exp_data['logodds_average']['best_combination']):
                
                experiment_metrics = exp_data['logodds_average']['best_combination']['metrics']
                
                # Extract each metric's mean and std only
                for metric in metrics:
                    if metric in experiment_metrics:
                        metric_data = experiment_metrics[metric]
                        if isinstance(metric_data, dict):
                            row_data[f'{metric}_mean'] = float(metric_data.get('mean', 0))
                            row_data[f'{metric}_std'] = float(metric_data.get('std', 0))
                    else:
                        # Fill missing metrics with zeros
                        row_data[f'{metric}_mean'] = 0.0
                        row_data[f'{metric}_std'] = 0.0
            
            results_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(results_data)
        
        # Sort by experiment type and then by f1_mean (descending)
        df = df.sort_values(['experiment_type', 'f1_mean'], ascending=[True, False])
        
        # Round numeric columns to 4 decimal places
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        print(f" Results saved to: {save_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        return df
    
    def _classify_experiment(self, exp_name: str) -> str:
        """Classify experiment type for sorting."""
        if exp_name == 'full_model':
            return 'A_Full_Model'
        elif exp_name.startswith('baseline_'):
            return 'B_Baselines'
        elif exp_name.startswith('remove_'):
            return 'C_Removals'
        else:
            return 'D_Other'

