import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import aaanalysis as aa
from .model_trainer import ModelTrainer, ModelVisualizer
from matplotlib import pyplot as plt
from .cdhit import CDHit

class CPPPipeline:
    """
    Main CPP (Comparative Physicochemical Properties) Pipeline class with Nested Cross-Validation.
    
    Implements a complete nested cross-validation framework for unbiased model evaluation:
    - Outer loop: 5-fold CV for final performance estimation
    - Inner loop: 5-fold CV for hyperparameter tuning  
    - CPP features extracted only on training data
    - Hyperparameter selection based on most frequent combination across outer folds
    - Final results reported with mean and standard deviation
    
    Usage:
    ```python
    pipeline = CPPPipeline(df_seq_sub, df_seq_nonsub, df_seq_ref)
    results = pipeline.run_nested_cv()
    ```
    """
    
    def __init__(self, df_seq_sub, df_seq_nonsub, df_seq_ref):
        """
        Initialize the CPP Pipeline with sequence data.
        
        Args:
            df_seq_sub: DataFrame containing substrate sequences
            df_seq_nonsub: DataFrame containing non-substrate sequences  
            df_seq_ref: DataFrame containing reference sequences for dPULearn
        """
        self.df_seq_sub = df_seq_sub
        self.df_seq_nonsub = df_seq_nonsub
        self.df_seq_ref = df_seq_ref
        self.cd_hit = CDHit()
    
    def _cdhit_filter(self, filter_type='partial', 
                      similarity_threshold=0.4, jmd_n_len=50, jmd_c_len=10, list_parts=['tmd','jmd_n', 'jmd_c']):
        '''
        Filter the sequences using CD-HIT.
        '''
        df_clust_sub = self.cd_hit.filter_seq(self.df_seq_sub, filter_type=filter_type,
                                        similarity_threshold=similarity_threshold, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=list_parts)
        df_clust_nonsub = self.cd_hit.filter_seq(self.df_seq_nonsub, filter_type=filter_type,
                                        similarity_threshold=similarity_threshold, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=list_parts)
        df_clust_ref = self.cd_hit.filter_seq(self.df_seq_ref, filter_type=filter_type,
                                        similarity_threshold=similarity_threshold, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=list_parts)

        df_seq_clustered_sub = pd.merge(self.df_seq_sub, df_clust_sub[['entry']], on='entry', how='inner').reset_index(drop=True)
        df_seq_clustered_nonsub = pd.merge(self.df_seq_nonsub, df_clust_nonsub[['entry']], on='entry', how='inner').reset_index(drop=True)
        df_seq_clustered_ref = pd.merge(self.df_seq_ref, df_clust_ref[['entry']], on='entry', how='inner').reset_index(drop=True)

        return df_seq_clustered_sub, df_seq_clustered_nonsub, df_seq_clustered_ref
    

    def _get_cpp_scales(self, n_clusters=133):
        '''cpp: sub as test; ref as ref'''

        # Load scales
        df_scales = aa.load_scales() # 586 scales in total
        # Obtain redundancy-reduced set of 133 scales (need to be changed)
        aac = aa.AAclust()
        X = np.array(df_scales).T
        scales = aac.fit(X, names=list(df_scales), n_clusters=n_clusters).medoid_names_
        df_scales = df_scales[scales]
        return df_scales
    

    def _cpp_run(self, df_scales, df_seq_clustered_sub, df_seq_clustered_ref, n_filter=100, 
                 jmd_n_len=50, jmd_c_len=10, list_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c'],
                 plot_cpp=True):
        '''
        Run CPP, obtain feature dataframe df_feat
        '''
        sf = aa.SequenceFeature()
        df_seq_clustered_sub_ref = pd.concat([df_seq_clustered_sub, df_seq_clustered_ref]).reset_index(drop=True)
        labels_sub_ref = df_seq_clustered_sub_ref['label'].to_list()
        df_parts_clustered_sub_ref = sf.get_df_parts(df_seq=df_seq_clustered_sub_ref,
                                list_parts=list_parts,
                                jmd_n_len=jmd_n_len,
                                jmd_c_len=jmd_c_len)

        # Create 100 baseline features (Scale values averaged over TMD)
        cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts_clustered_sub_ref, accept_gaps=True)
        df_feat = cpp.run(labels=labels_sub_ref,n_filter=n_filter,label_test=1, label_ref=2)
        X_sub_ref = sf.feature_matrix(df_parts=df_parts_clustered_sub_ref, features=df_feat["feature"],accept_gaps=True)

        if plot_cpp:
            self._cpp_plot(X_sub_ref, labels_sub_ref, df_feat, jmd_n_len, jmd_c_len, plot_type='ranking')
            self._cpp_plot(X_sub_ref, labels_sub_ref, df_feat, jmd_n_len, jmd_c_len, plot_type='profile')
            self._cpp_plot(X_sub_ref, labels_sub_ref, df_feat, jmd_n_len, jmd_c_len, plot_type='feature_map')

        return df_feat
    
    def _cpp_plot(self, X_sub_ref, labels_sub_ref, df_feat,jmd_n_len=50, jmd_c_len=10, plot_type='ranking'):
        '''
        Plot CPP features.
        ''' 
        tm = aa.TreeModel()
        tm.fit(X_sub_ref, labels=labels_sub_ref)
        df_feat_new = tm.add_feat_importance(df_feat=df_feat)
        cpp_plot = aa.CPPPlot(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_gaps=True)
        if plot_type == 'ranking':
            aa.plot_settings(short_ticks=True, weight_bold=False)
            cpp_plot.ranking(df_feat=df_feat_new)
        elif plot_type == 'profile':
            aa.plot_settings(font_scale=0.9)
            cpp_plot.profile(df_feat=df_feat_new)
        elif plot_type == 'feature_map':
            aa.plot_settings(font_scale=0.65, weight_bold=False)
            cpp_plot.feature_map(df_feat=df_feat_new)
        plt.show()

    def _cpp_shapmap(self, df_seq_clustered_sub_ref, labels_sub_ref, df_feat, jmd_n_len=50, jmd_c_len=10, 
                     names="APP", uniprot_id="P05067",
                     list_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c']):
        '''
        Plot CPP SHAP map.
        '''
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(df_seq=df_seq_clustered_sub_ref,
                                list_parts=list_parts,
                                jmd_n_len=jmd_n_len,
                                jmd_c_len=jmd_c_len)
        X_sub_ref = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"],accept_gaps=True)
        se = aa.ShapModel()
        se.fit(X_sub_ref, labels=labels_sub_ref, label_target_class=1)
        df_feat = se.add_sample_mean_dif(X_sub_ref, labels=labels_sub_ref, df_feat=df_feat, 
                                         label_ref=2,
                                        sample_positions=0, names=names)
        df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=0, names=names)

        _df_parts = sf.get_df_parts(df_seq=df_seq_clustered_sub_ref, list_parts=["tmd", "jmd_c", "jmd_n"])
        _args_seq = _df_parts.loc[uniprot_id].to_dict()   # Accession number of APP
        args_seq = {key + "_seq": _args_seq[key] for key in _args_seq}

        cpp_plot = aa.CPPPlot()
        aa.plot_settings(short_ticks=True, weight_bold=False)

        # CPP heatmap (sample level)
        fs = aa.plot_gcfs()
        aa.plot_settings(font_scale=0.65, weight_bold=False)
        cpp_plot.heatmap(df_feat=df_feat, shap_plot=True,
                        col_val="mean_dif_"+names, **args_seq, name_test=names)
        plt.title("CPP heatmap for "+names, fontsize=fs+5, weight="bold")
        plt.show()
        return df_feat

    def _cpp_profile(self, df_feat, df_seq, jmd_n_len=50, jmd_c_len=10, 
                     list_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c'], accept_gaps=True):
        """
        Extract CPP feature matrix from sequences using pre-computed features.
        
        Args:
            df_feat: DataFrame containing CPP features from feature selection
            df_seq: DataFrame containing sequences to profile
            
        Returns:
            numpy.ndarray: Feature matrix X with shape (n_sequences, n_features)
        """
        sf = aa.SequenceFeature()
        df_parts = sf.get_df_parts(
            df_seq=df_seq,
            list_parts=list_parts,
            jmd_n_len=jmd_n_len,
            jmd_c_len=jmd_c_len
        )
        feature_matrix = sf.feature_matrix(
            df_parts=df_parts, 
            features=df_feat["feature"], 
            accept_gaps=accept_gaps
        )
        return feature_matrix
    
    def _add_dpu_labels(self, X, X_labels, X_ref):
        """
        Apply dPULearn to identify reliable negatives from reference sequences.
        
        Label mapping:
        - 0: Original negatives (non-substrates)
        - 1: Positives (substrates) 
        - 2: Unlabeled (reference sequences)
        - 3: dPULearn negatives (reference sequences identified as reliable negatives)

        Returns:
            tuple: (X_final, labels_final)
                - X_final: Extended X with dPULearn negatives
                - labels_final: Extended X_labels with dPULearn negatives
        """
        # Separate positive and negative samples
        mask_positives = (np.array(X_labels) == 1)
        mask_negatives = (np.array(X_labels) == 0)
        X_positives = X[mask_positives]
        X_negatives = X[mask_negatives]
        
        # Prepare data for dPULearn
        X_pos_unl = np.concatenate([X_positives, X_ref], axis=0)
        labels_pos_unl = np.hstack([
            np.ones(len(X_positives), dtype=int),      # 1: positive
            np.full(len(X_ref), 2, dtype=int)          # 2: unlabeled
        ])
        
        # Apply dPULearn
        dpul = aa.dPULearn()
        n_neg_to_extract = len(X_positives) - len(X_negatives)
        dpul.fit(X=X_pos_unl, labels=labels_pos_unl, n_unl_to_neg=n_neg_to_extract)
        
        # Get results
        labels_after_dpu = dpul.labels_
        mask_changed_to_neg = (labels_after_dpu != labels_pos_unl)
        labels_with_dpu_marked = labels_after_dpu.copy()
        labels_with_dpu_marked[mask_changed_to_neg] = 3
        
        # Extract newly identified negatives
        X_dpu_negatives = X_pos_unl[labels_with_dpu_marked == 3]
        
        # Construct final data
        X_aug = np.concatenate([X, X_dpu_negatives], axis=0)
        labels_aug = np.hstack([
            X_labels,
            np.full(len(X_dpu_negatives), 3, dtype=int)
        ])
        mask_dpu_negatives = (np.array(labels_aug) == 3)
        
        # Log results
        print(f"dPULearn Results:")
        print(f"  - Reliable negatives identified: {len(X_dpu_negatives)}")
        print(f"  - Label distribution: {dict(pd.Series(labels_aug).value_counts().sort_index())}")
        
        return X_aug, labels_aug, mask_dpu_negatives
    
    def _outer_fold_split(self, df_combined, n_splits=5, random_state=42):
        """
        Split data into outer folds for nested CV.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        outer_folds = []
        
        for train_idx, test_idx in skf.split(df_combined, df_combined['label']):
            df_train = df_combined.iloc[train_idx].reset_index(drop=True)
            df_test = df_combined.iloc[test_idx].reset_index(drop=True)
            outer_folds.append((df_train, df_test))
        
        return outer_folds
    

    
    def _process_outer_fold(self, outer_fold_idx, df_train, df_test, df_ref, 
                           jmd_n_len=50, jmd_c_len=10, cpp_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c'],
                           n_clusters=133, n_filter=100, plot_cpp=False, apply_dpu=True, inner_folds=5):
        """
        Process a single outer fold: CPP feature selection, dPULearn, inner CV for hyperparameter tuning.
        """
        print(f"\n=== Processing Outer Fold {outer_fold_idx + 1} ===")
        
        # Step 1: CPP feature selection using training substrates vs reference
        df_train_sub = df_train[df_train['label'] == 1].reset_index(drop=True)
        df_scales = self._get_cpp_scales(n_clusters=n_clusters)
        
        print("  Running CPP feature selection...")
        df_feat = self._cpp_run(
            df_scales=df_scales,
            df_seq_clustered_sub=df_train_sub,
            df_seq_clustered_ref=df_ref,
            n_filter=n_filter,
            jmd_n_len=jmd_n_len,
            jmd_c_len=jmd_c_len,
            list_parts=cpp_parts,
            plot_cpp=plot_cpp
        )
        
        # Step 2: Extract features for all datasets using selected features
        print("  Extracting features...")
        X_train = self._cpp_profile(df_feat, df_train, jmd_n_len, jmd_c_len, cpp_parts)
        X_test = self._cpp_profile(df_feat, df_test, jmd_n_len, jmd_c_len, cpp_parts)
        X_ref = self._cpp_profile(df_feat, df_ref, jmd_n_len, jmd_c_len, cpp_parts)
        
        y_train = df_train['label'].values
        y_test = df_test['label'].values
        
        # Step 3: Apply dPULearn to augment training data
        if apply_dpu:
            print("  Applying dPULearn...")
            X_train_aug, y_train_aug, mask_dpu_negatives = self._add_dpu_labels(X_train, y_train, X_ref) 
        else:
            print("  Applying dPULearn is disabled, using unbalanced training data")
            X_train_aug = X_train
            y_train_aug = y_train
            mask_dpu_negatives = None
        
        # Step 4: Inner CV for hyperparameter tuning
        print("  Running inner CV for hyperparameter tuning...")
        
        # Initialize trainer for hyperparameter tuning
        trainer = ModelTrainer(random_state=42, auto_tune=True)
        
        # Convert dPULearn labels (3) to negative labels (0) for training
        y_train_aug_for_training = y_train_aug.copy()
        y_train_aug_for_training[y_train_aug_for_training == 3] = 0
        
        # Tune hyperparameters on augmented training data
        best_params = trainer._tune_hyperparameters(X_train_aug, y_train_aug_for_training, cv=inner_folds)
        
        print(f"    Best parameters from tuning: {best_params}")

        # Step 5: Train final models on full training data with best parameters
        print("  Training final models with best parameters...")
        
        final_models = trainer.train_final_models(
            X_train_aug, y_train_aug_for_training, best_params
        )
        
        # Step 6: Evaluate on test set
        print("  Evaluating on test set...")
        test_results = trainer.evaluate_final_models(final_models, X_test, y_test)
        
        # Return results for this outer fold
        return {
            'outer_fold': outer_fold_idx + 1,
            'best_params': best_params,
            'test_results': test_results,
            'feature_info': {
                'features': df_feat,
                'n_augmented_samples': len(X_train_aug) - len(X_train)
            }
        }
    
    def run_nested_cv(self, filter_type='partial', similarity_threshold=0.4, 
                     jmd_n_len=50, jmd_c_len=10, cdhit_parts=['tmd','jmd_n', 'jmd_c'],
                     n_clusters=133, cpp_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c'],
                     n_filter=100, plot_cpp=False, outer_folds=5, apply_dpu=True, inner_folds=5):
        """
        Run complete nested cross-validation pipeline.
        
        Args:
            filter_type: CD-HIT filtering type ('partial' or 'full')
            similarity_threshold: Similarity threshold for clustering
            jmd_n_len: N-terminal juxtamembrane domain length
            jmd_c_len: C-terminal juxtamembrane domain length  
            cdhit_parts: Parts to include in CD-HIT
            n_clusters: Number of clusters for CPP scales
            cpp_parts: Parts to include in CPP
            n_filter: Number of top features to select
            plot_cpp: Whether to plot CPP results
            outer_folds: Number of outer folds
            apply_dpu: Whether to apply dPULearn
            inner_folds: Number of inner folds for hyperparameter tuning
        Returns:
            dict: Complete nested CV results with statistics
        """
        print("=== Starting Nested Cross-Validation Pipeline ===")
        
        # Step 1: CD-HIT filtering
        print("\n=== Step 1: CD-HIT Filtering ===")
        df_seq_clustered_sub, df_seq_clustered_nonsub, df_seq_clustered_ref = self._cdhit_filter(
            filter_type=filter_type,
            similarity_threshold=similarity_threshold,
            jmd_n_len=jmd_n_len,
            jmd_c_len=jmd_c_len,
            list_parts=cdhit_parts
        )
        
        # Step 2: Combine substrates and non-substrates
        print("\n=== Step 2: Data Preparation ===")
        df_combined = pd.concat([df_seq_clustered_sub, df_seq_clustered_nonsub]).reset_index(drop=True)
        
        print(f"Total sequences: {len(df_combined)}")
        print(f"Substrates: {len(df_seq_clustered_sub)}")
        print(f"Non-substrates: {len(df_seq_clustered_nonsub)}")
        print(f"Reference sequences: {len(df_seq_clustered_ref)}")
        
        # Step 3: Create outer folds
        print(f"\n=== Step 3: Creating {outer_folds} Outer Folds ===")
        outer_fold_splits = self._outer_fold_split(df_combined, n_splits=outer_folds, random_state=42)
        
        # Step 4: Process each outer fold
        print("\n=== Step 4: Processing Outer Folds ===")
        outer_fold_results = []
        
        for outer_fold_idx, (df_train, df_test) in enumerate(outer_fold_splits):
            fold_result = self._process_outer_fold(
                outer_fold_idx, df_train, df_test, df_seq_clustered_ref,
                jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, cpp_parts=cpp_parts,
                n_clusters=n_clusters, n_filter=n_filter, plot_cpp=plot_cpp,
                apply_dpu=apply_dpu, inner_folds=inner_folds
            )
            outer_fold_results.append(fold_result)
        
        # Step 5: Analyze results across all outer folds
        print("\n=== Step 5: Analyzing Results Across Outer Folds ===")
        
        # Collect all hyperparameters
        all_params = [result['best_params'] for result in outer_fold_results]
        
        # Find most frequent hyperparameters (convert dict to tuple for hashing)
        def dict_to_tuple(d):
            return tuple(sorted(d.items())) if d else ()
        
        param_tuples = [dict_to_tuple(params) for params in all_params]
        most_common = Counter(param_tuples).most_common(1)[0][0]
        best_params_final = {
            'model': dict(most_common)
        }
        
        # Collect test results
        test_results = [result['test_results']['model'] for result in outer_fold_results]
        
        # Calculate statistics
        def calculate_stats(results, metric):
            values = [r[metric] for r in results]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Calculate statistics for all metrics
        metrics = ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'mcc']
        
        final_results = {
            'model': {}
        }
        
        for metric in metrics:
            final_results['model'][metric] = calculate_stats(test_results, metric)
        
        # Step 6: Generate final report
        print("\n=== Step 6: Final Results Summary ===")
        print(f"Most frequent hyperparameters across {outer_folds} outer folds:")
        print(f"  Model: {best_params_final['model']}")
        
        print(f"\nTest Performance (Mean ± Std across {outer_folds} folds):")
        for metric in metrics:
            model_mean = final_results['model'][metric]['mean']
            model_std = final_results['model'][metric]['std']
            
            print(f"  {metric.upper()}:")
            print(f"    Model: {model_mean:.4f} ± {model_std:.4f}")
            
            if model_mean > 0:
                improvement = (model_mean - model_mean) / model_mean * 100
                print(f"    Improvement: {improvement:.2f}%")
        
        # Compile complete results
        complete_results = {
            'config': {
                'filter_type': filter_type,
                'similarity_threshold': similarity_threshold,
                'outer_folds': outer_folds,
                'inner_folds': 5,
                'n_clusters': n_clusters,
                'n_filter': n_filter,
                'jmd_n_len': jmd_n_len,
                'jmd_c_len': jmd_c_len,
                'cpp_parts': cpp_parts
            },
            'data_info': {
                'total_sequences': len(df_combined),
                'substrates': len(df_seq_clustered_sub),
                'non_substrates': len(df_seq_clustered_nonsub),
                'reference': len(df_seq_clustered_ref)
            },
            'outer_fold_results': outer_fold_results,
            'best_hyperparameters': best_params_final,
            'final_performance': final_results,
            'hyperparameter_analysis': {
                'params_frequency': Counter(param_tuples)
            }
        }
        
        print("\n=== Nested Cross-Validation Complete ===")
        return complete_results
    
    def run_leave_one_out(self, filter_type='partial', similarity_threshold=0.4, 
                         jmd_n_len=50, jmd_c_len=10, cdhit_parts=['tmd','jmd_n', 'jmd_c'],
                         n_clusters=133, cpp_parts=['tmd','jmd_n_tmd_n', 'tmd_c_jmd_c'],
                         n_filter=100, plot_cpp=False, apply_dpu=True, inner_folds=5):
        """
        Run Leave One Out Cross-Validation with CPP feature engineering and dPULearn.
        
        This method reuses the _process_outer_fold function from nested CV to ensure
        consistent processing logic and compatible result formats.
        
        Args:
            filter_type: CD-HIT filtering type ('partial' or 'full')
            similarity_threshold: Similarity threshold for clustering
            jmd_n_len: N-terminal juxtamembrane domain length
            jmd_c_len: C-terminal juxtamembrane domain length  
            cdhit_parts: Parts to include in CD-HIT
            n_clusters: Number of clusters for CPP scales
            cpp_parts: Parts to include in CPP
            n_filter: Number of top features to select
            plot_cpp: Whether to plot CPP results
            apply_dpu: Whether to apply dPULearn
            inner_folds: Number of inner CV folds for hyperparameter tuning
            
        Returns:
            dict: Complete LOOCV results with same format as nested CV
        """
        print("="*80)
        print("LEAVE ONE OUT CROSS-VALIDATION")
        print("="*80)
        
        # Step 1: Data preprocessing and filtering
        print("\n=== Step 1: Data Preprocessing ===")
        
        # Apply CD-HIT filtering
        print(f"Applying CD-HIT filtering (threshold: {similarity_threshold})...")
        df_seq_clustered_sub, df_seq_clustered_nonsub, df_seq_clustered_ref = self._cdhit_filter(
            filter_type=filter_type, similarity_threshold=similarity_threshold, 
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=cdhit_parts
        )
        
        print(f"After filtering - Substrates: {len(df_seq_clustered_sub)}, "
              f"Non-substrates: {len(df_seq_clustered_nonsub)}, "
              f"Reference: {len(df_seq_clustered_ref)}")
        
        # Combine data and create labels
        df_combined = pd.concat([df_seq_clustered_sub, df_seq_clustered_nonsub], ignore_index=True)
        n_samples = len(df_combined)
        
        print(f"Total samples for LOOCV: {n_samples}")
        
        # Step 2: Leave One Out Cross-Validation using _process_outer_fold
        print(f"\n=== Step 2: Leave One Out Cross-Validation ===")
        print(f"Running {n_samples} iterations using nested CV processing logic...")
        
        # Storage for results
        outer_fold_results = []
        
        # Run LOOCV iterations - each sample becomes a "fold"
        for loo_idx in range(n_samples):
            print(f"\nLOOCV Iteration {loo_idx + 1}/{n_samples}: Processing sample {loo_idx}")
            
            # Create train/test split (leave one out)
            test_idx = loo_idx
            train_indices = list(range(n_samples))
            train_indices.remove(test_idx)
            
            df_train = df_combined.iloc[train_indices].reset_index(drop=True)
            df_test = df_combined.iloc[[test_idx]].reset_index(drop=True)
            
            print(f"  Training set size: {len(df_train)}")
            print(f"  Test sample: {df_test['entry'].iloc[0]} (label: {df_test['label'].iloc[0]})")
            
            # Process this "fold" using existing outer fold logic
            fold_result = self._process_outer_fold(
                outer_fold_idx=loo_idx,
                df_train=df_train,
                df_test=df_test,
                df_ref=df_seq_clustered_ref,
                jmd_n_len=jmd_n_len,
                jmd_c_len=jmd_c_len,
                cpp_parts=cpp_parts,
                n_clusters=n_clusters,
                n_filter=n_filter,
                plot_cpp=plot_cpp,
                apply_dpu=apply_dpu,
                inner_folds=inner_folds
            )
            
            outer_fold_results.append(fold_result)
            
            if (loo_idx + 1) % 10 == 0:
                print(f"  Completed {loo_idx + 1}/{n_samples} iterations")
        
        # Step 3: Aggregate results (using same logic as nested CV)
        print("\n=== Step 3: Aggregating Results ===")
        
        # Helper functions (same as nested CV)
        def dict_to_tuple(d):
            return tuple(sorted(d.items()))
        
        # NEW: Proper LOO evaluation - aggregate all predictions first, then calculate metrics
        print("  Aggregating all predictions for comprehensive evaluation...")
        
        # Collect all predictions and true labels from all LOO iterations
        all_predictions = []
        all_probabilities = []
        all_y_true = []
        
        for fold_result in outer_fold_results:
            # Extract predictions from outer_fold_results[i]['test_results']['model']['predictions']
            predictions = fold_result['test_results']['model']['predictions']
            probabilities = fold_result['test_results']['model']['probabilities']
            
            # Extract true labels from outer_fold_results[i]['test_results']['test_data']['y_test']
            y_true = fold_result['test_results']['test_data']['y_test']
            
            # Append to aggregated lists
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_y_true.extend(y_true)
        
        print(f"  Total predictions collected: {len(all_predictions)}")
        print(f"  Total true labels collected: {len(all_y_true)}")
        
        # Calculate comprehensive metrics on aggregated data
        from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score, 
                                   recall_score, precision_score, matthews_corrcoef, f1_score)
        
        aggregated_metrics = {
            'accuracy': accuracy_score(all_y_true, all_predictions),
            'auc': roc_auc_score(all_y_true, all_probabilities),
            'balanced_accuracy': balanced_accuracy_score(all_y_true, all_predictions),
            'precision': precision_score(all_y_true, all_predictions, zero_division=0),
            'recall': recall_score(all_y_true, all_predictions, zero_division=0),
            'f1': f1_score(all_y_true, all_predictions, zero_division=0),
            'mcc': matthews_corrcoef(all_y_true, all_predictions)
        }
        
        # Format results to match nested CV structure
        final_results = {}
        for metric, value in aggregated_metrics.items():
            final_results[metric] = {
                'mean': value,
                'std': 0.0,  # No std for LOOCV - deterministic result
                'values': [value]  # Single aggregated value
            }
        
        # Aggregate hyperparameters
        all_best_params = [result['best_params'] for result in outer_fold_results]
        param_tuples = [dict_to_tuple(params) for params in all_best_params]
        param_counter = Counter(param_tuples)
        
        # Get most frequent hyperparameters
        most_common_params = dict(param_counter.most_common(1)[0][0])
        
        # Step 4: Generate final report
        print("\n=== Step 4: Final Results Summary ===")
        print(f"Leave One Out Cross-Validation Results (n={n_samples}):")
        print(f"  Metrics calculated on aggregated predictions across all {n_samples} iterations:")
        
        for metric, stats in final_results.items():
            print(f"  {metric.upper()}: {stats['mean']:.4f}")
        
        print(f"\nCorrect classifications: {sum(np.array(all_predictions) == np.array(all_y_true))}/{len(all_y_true)}")
        print(f"Error rate: {1 - final_results['accuracy']['mean']:.4f}")
        
        print(f"\nHyperparameter Analysis:")
        print(f"  Total unique combinations: {len(param_counter)}")
        print(f"  Most frequent parameters: {most_common_params}")
        print(f"  Used in {param_counter.most_common(1)[0][1]}/{n_samples} iterations")
        
        # Compile complete results (same format as nested CV)
        loocv_results = {
            'method': 'leave_one_out',
            'config': {
                'filter_type': filter_type,
                'similarity_threshold': similarity_threshold,
                'outer_folds': n_samples,  # Each sample is a "fold"
                'inner_folds': inner_folds,
                'n_clusters': n_clusters,
                'n_filter': n_filter,
                'jmd_n_len': jmd_n_len,
                'jmd_c_len': jmd_c_len,
                'cpp_parts': cpp_parts,
                'apply_dpu': apply_dpu
            },
            'data_info': {
                'total_sequences': n_samples,
                'substrates': len(df_seq_clustered_sub),
                'non_substrates': len(df_seq_clustered_nonsub),
                'reference': len(df_seq_clustered_ref)
            },
            'outer_fold_results': outer_fold_results,
            'best_hyperparameters': {
                'model': most_common_params
            },
            'final_performance': {
                'model': final_results
            },
            'hyperparameter_analysis': {
                'params_frequency': param_counter
            }
        }
        
        print("\n=== Leave One Out Cross-Validation Complete ===")
        print(f"Results format matches nested CV - compatible with all visualization functions")
        
        return loocv_results

    def plot_loocv_results(self, loocv_results,
                          title="Leave One Out Cross-Validation Results",
                          subtitle="CPP + dPULearn Pipeline",
                          figsize=(10, 6)):
        """
        Plot Leave One Out cross-validation results.
        
        Note: Since LOOCV now uses the same result format as nested CV,
        this method now delegates to the standard plotting function.
        
        Args:
            loocv_results: Dictionary containing LOOCV results
            title: Main title for the plot
            subtitle: Subtitle describing the application
            figsize: Figure size tuple
        """
        # LOOCV now has the same format as nested CV, so use the standard plot function
        return self.plot_nested_cv_results(
            loocv_results,
            title=title,
            subtitle=subtitle,
            figsize=figsize,
            show_hyperparams=True
        )

    
    def plot_nested_cv_results(self, complete_results, 
                              title="Nested Cross-Validation Results",
                              subtitle="Random Forest Classification",
                              figsize=(12, 8),
                              show_hyperparams=True):
        """
        Plot comprehensive nested cross-validation results.
        
        Args:
            complete_results: Dictionary containing complete nested CV results
            title: Main title for the plot
            subtitle: Subtitle describing the specific application
            figsize: Figure size tuple
            show_hyperparams: Whether to show hyperparameters panel
        """
        # Extract data
        best_params = complete_results['best_hyperparameters']['model']
        final_results = complete_results['final_performance']['model']
        
        return ModelVisualizer.plot_results(
            best_params, final_results, title, subtitle, figsize, show_hyperparams
        )

    def analyze_feature_robustness(self, complete_results, 
                                 title="CPP Feature Selection Robustness",
                                 figsize=(16, 12),
                                 top_n_features=20):
        """
        Analyze the consistency of CPP feature selection across outer folds.
        
        Args:
            complete_results: Dictionary containing complete nested CV results
            title: Main title for the analysis
            figsize: Figure size tuple  
            top_n_features: Number of top consistent features to highlight
            
        Returns:
            dict: Analysis results including robustness metrics
        """
        return ModelVisualizer.analyze_feature_robustness(
            complete_results, title, figsize, top_n_features
        )
    
    def analyze_hyperparameter_stability(self, loocv_results, 
                                        title="LOOCV Hyperparameter Stability Analysis",
                                        figsize=(14, 8)):
        """
        Analyze the stability of hyperparameter selection across LOOCV iterations.
        
        Args:
            loocv_results: Dictionary containing LOOCV results
            title: Main title for the analysis
            figsize: Figure size tuple  
            
        Returns:
            dict: Analysis results including stability metrics
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract hyperparameter data from outer fold results
        outer_fold_results = loocv_results['outer_fold_results']
        all_hyperparams = [result['best_params'] for result in outer_fold_results]
        hyperparams_counter = loocv_results['hyperparameter_analysis']['params_frequency']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Hyperparameter combination frequency
        top_combinations = hyperparams_counter.most_common(10)
        combinations = [f"Config {i+1}" for i in range(len(top_combinations))]
        frequencies = [count for _, count in top_combinations]
        
        axes[0,0].bar(combinations, frequencies, color='skyblue')
        axes[0,0].set_title('Top 10 Hyperparameter Combinations', fontweight='bold')
        axes[0,0].set_xlabel('Configuration')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Individual hyperparameter stability
        # Count frequency of each hyperparameter value
        param_stability = {}
        for param_dict in all_hyperparams:
            for param, value in param_dict.items():
                if param not in param_stability:
                    param_stability[param] = {}
                if value not in param_stability[param]:
                    param_stability[param][value] = 0
                param_stability[param][value] += 1
        
        # Plot stability for each parameter
        param_names = list(param_stability.keys())
        stability_scores = []
        
        for param in param_names:
            # Calculate stability as max frequency / total iterations
            max_freq = max(param_stability[param].values())
            stability = max_freq / len(all_hyperparams)
            stability_scores.append(stability)
        
        axes[0,1].bar(param_names, stability_scores, color='lightcoral')
        axes[0,1].set_title('Individual Parameter Stability', fontweight='bold')
        axes[0,1].set_xlabel('Hyperparameter')
        axes[0,1].set_ylabel('Stability Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)
        
        # 3. Hyperparameter evolution over iterations
        n_iterations = len(outer_fold_results)
        iteration_indices = range(1, n_iterations + 1)
        
        # Track one key parameter over iterations (e.g., max_depth)
        if 'max_depth' in param_names:
            key_param = 'max_depth'
        else:
            key_param = param_names[0]
        
        key_param_values = []
        for result in outer_fold_results:
            value = result['best_params'][key_param]
            # Convert None to a numerical value for plotting
            if value is None:
                value = -1
            key_param_values.append(value)
        
        axes[1,0].scatter(iteration_indices, key_param_values, alpha=0.6, color='green')
        axes[1,0].set_title(f'{key_param} Evolution Over Iterations', fontweight='bold')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel(f'{key_param} Value')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Stability metrics summary
        axes[1,1].axis('off')
        
        # Calculate overall stability metrics
        total_combinations = len(hyperparams_counter)
        most_frequent_count = hyperparams_counter.most_common(1)[0][1]
        overall_stability = most_frequent_count / len(all_hyperparams)
        
        stability_text = f"""
        Stability Metrics:
        
        • Total Iterations: {len(all_hyperparams)}
        • Unique Combinations: {total_combinations}
        • Most Frequent Count: {most_frequent_count}
        • Overall Stability: {overall_stability:.3f}
        
        Individual Parameter Stability:
        """
        
        for param, score in zip(param_names, stability_scores):
            stability_text += f"\n• {param}: {score:.3f}"
        
        axes[1,1].text(0.1, 0.9, stability_text, transform=axes[1,1].transAxes, 
                      fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Return analysis results
        analysis_results = {
            'overall_stability': overall_stability,
            'total_combinations': total_combinations,
            'most_frequent_count': most_frequent_count,
            'parameter_stability': {param: max(param_stability[param].values()) / len(all_hyperparams) 
                                  for param in param_names},
            'stability_scores': dict(zip(param_names, stability_scores))
        }
        
        return analysis_results

    @staticmethod
    def compare_multiple_results(*complete_results_list, 
                               labels=None,
                               title="Nested Cross-Validation Comparison",
                               subtitle="Performance Comparison Across Different Conditions",
                               figsize=(12, 6)):
        """
        Compare multiple nested cross-validation results in one comprehensive plot.
        
        Args:
            *complete_results_list: Variable number of complete_results dictionaries
            labels: List of labels for each result (if None, will use default labels)
            title: Main title for the comparison
            subtitle: Subtitle describing the comparison
            figsize: Figure size tuple
            
        Returns:
            dict: Dictionary containing figure and comparison results including hyperparameters
            
        Example:
            # Compare 3 different conditions
            results = CPPPipeline.compare_multiple_results(
                results1, results2, results3,
                labels=["With dPULearn", "Without dPULearn", "Baseline"],
                title="dPULearn Impact Analysis",
                subtitle="Comparison of Different Training Strategies"
            )
            
            # Access hyperparameters
            print(results['hyperparameters'])
            
            # Access overall winner
            print(results['overall_winner'])
        """
        return ModelVisualizer.compare_multiple_results(
            *complete_results_list, 
            labels=labels,
            title=title,
            subtitle=subtitle,
            figsize=figsize
        )


