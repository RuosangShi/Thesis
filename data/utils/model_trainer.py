'''
Model Trainer

labels:
- 0: Original negatives (non-substrates)
- 1: Positives (substrates) 
- 2: Unlabeled (reference sequences)
- 3: dPULearn negatives (reference sequences identified as reliable negatives)


Input:
X: Substrate, Non-substrate, dPULearn negatives
y: 0, 1, 3 (0: Original negatives, 1: Positives, 2: Unlabeled, 3: dPULearn negatives)

Output:
- model: trained model
- metrics: performance metrics

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc, 
                            balanced_accuracy_score, recall_score, precision_score, 
                            matthews_corrcoef, f1_score)
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import seaborn as sns


class ModelTrainer:
    """
    Handles model training, evaluation, and visualization for CPP pipeline.
    Supports both original and dPULearn-augmented training approaches.
    """
    
    def __init__(self, random_state=42, n_estimators=100, auto_tune=True):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.auto_tune = auto_tune
        self.best_params_ = None
        
        # Define hyperparameter grid for tuning
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    def _tune_hyperparameters(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of CV folds
            
        Returns:
            dict: Best parameters found
        """
        if not self.auto_tune:
            return {'n_estimators': self.n_estimators}
        
        print("    Tuning hyperparameters...")
        
        # Create base model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Perform grid search with CV
        grid_search = GridSearchCV(
            base_model, 
            self.param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best parameters: {grid_search.best_params_}")
        print(f"    Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
 
    def train_final_models(self, X, y, best_params):
        """
        Train final models on complete CV dataset.
        
        Args:
            X, y: CV data, could be augmented or not (dPULearn processed or not)
            best_params: Best hyperparameters from CV
            
        Returns:
            dict: Final trained models and training data info
        """
        print("Training final models on full CV data...")
        print(f"Using best parameters: {best_params}")
        
        # Final Model
        final_model = RandomForestClassifier(
            random_state=self.random_state, 
            n_jobs=-1,
            **best_params
        )
        final_model.fit(X, y)
        
        return {
            'model': final_model,
            'best_params': best_params,
            'training_data': {
                'X': X,
                'y': y
            }
        }
    
    def evaluate_final_models(self, final_models, X_test, y_test):
        """
        Evaluate final models on independent test set.
        
        Args:
            final_models: Dictionary of trained models and training data
            X_test, y_test: Independent test data
            
        Returns:
            dict: Test evaluation results
        """
        final_model = final_models['model']
        
        # Evaluate model on independent test set (comprehensive metrics)
        test_pred = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_metrics = self._calculate_comprehensive_metrics(y_test, test_pred, test_proba)
        
        test_results = {
            'model': {
                'model': final_model,
                'metrics': test_metrics,
                'accuracy': test_metrics['accuracy'],
                'auc': test_metrics['auc'],
                'balanced_accuracy': test_metrics['balanced_accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'mcc': test_metrics['mcc'],
                'predictions': test_pred,
                'probabilities': test_proba
            },
            'test_data': {
                'X_test': X_test,
                'y_test': y_test
            }
        }
        
        print(f"Final Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}, MCC: {test_metrics['mcc']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, Precision: {test_metrics['precision']:.4f}, "
              f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        
        return test_results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            dict: Comprehensive metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba)
        }


class ModelVisualizer:
    """
    Handles visualization of model performance and ROC curves.
    """
    
    @staticmethod
    def plot_results(best_params, final_results,
                     title="Nested Cross-Validation Results",
                     subtitle="Random Forest Classification with dPULearn",
                     figsize=(12, 8),
                     show_hyperparams=True):
        """
        Create a simple visualization of nested cross-validation results.
        
        Args:
            complete_results: Dictionary containing complete nested CV results
            title: Main title for the plot
            subtitle: Subtitle describing the specific application
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            show_hyperparams: Whether to show hyperparameters panel
        """
        # Set up the plotting style
        plt.style.use('default')
        
        # Create figure
        if show_hyperparams:
            fig, (ax_params, ax_main) = plt.subplots(2, 1, figsize=figsize, 
                                                   gridspec_kw={'height_ratios': [1, 3]})
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=figsize)
        
        # Main title
        fig.suptitle(f'{title}\n{subtitle}', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot hyperparameters if requested
        if show_hyperparams:
            ax_params.axis('off')
            
            # Create hyperparameter text
            param_text = "Best Hyperparameters: "
            param_list = []
            for param, value in best_params.items():
                param_list.append(f"{param.replace('_', ' ').title()}: {value}")
            param_text += " • ".join(param_list)
            
            ax_params.text(0.5, 0.5, param_text, transform=ax_params.transAxes,
                          fontsize=10, ha='center', va='center', fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
        
        # Metrics to plot
        metrics = ['auc', 'accuracy', 'balanced_accuracy', 'f1', 'mcc', 'precision', 'recall']
        metric_labels = ['AUC', 'Accuracy', 'Balanced Accuracy', 'F1 Score', 'MCC', 'Precision', 'Recall']
        
        # Extract means and standard deviations
        means = [final_results[metric]['mean'] for metric in metrics]
        stds = [final_results[metric]['std'] for metric in metrics]
        
        # Plot points and error bars
        x_pos = np.arange(len(metrics))
        
        # Plot points
        points = ax_main.scatter(x_pos, means, s=100, color='#2E86AB', alpha=0.8, zorder=3)
        
        # Connect points with dashed line
        ax_main.plot(x_pos, means, color='#2E86AB', linestyle='--', linewidth=2, alpha=0.6, zorder=2)
        
        # Plot error bars (dashed lines)
        for i, (mean, std) in enumerate(zip(means, stds)):
            # Vertical error bars
            ax_main.plot([i, i], [mean - std, mean + std], 
                        color='#A23B72', linestyle='--', linewidth=2, alpha=0.7)
            # Horizontal caps
            ax_main.plot([i-0.05, i+0.05], [mean - std, mean - std], 
                        color='#A23B72', linestyle='-', linewidth=2, alpha=0.7)
            ax_main.plot([i-0.05, i+0.05], [mean + std, mean + std], 
                        color='#A23B72', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add compact value labels on points
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax_main.annotate(f'{mean:.3f}±{std:.3f}', 
                           (i, mean), 
                           textcoords="offset points", 
                           xytext=(0,12), 
                           ha='center', 
                           fontsize=9, 
                           fontweight='bold')
        
        # Customize plot
        ax_main.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax_main.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
        ax_main.set_xticks(x_pos)
        ax_main.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        
        # Make y-axis tick labels bold
        ax_main.tick_params(axis='y', labelsize=10)
        for tick in ax_main.get_yticklabels():
            tick.set_fontweight('bold')

        # Set y-axis limits with padding
        y_min = min([m - s for m, s in zip(means, stds)]) - 0.05
        y_max = max([m + s for m, s in zip(means, stds)]) + 0.1
        ax_main.set_ylim(y_min, y_max)
        
        # Add horizontal reference lines
        ax_main.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax_main.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=2, label='0.5 Reference')
        
        # Color-code points based on performance
        for i, (metric, mean) in enumerate(zip(metrics, means)):
            if metric in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
                if mean >= 0.8:
                    color = 'green'
                elif mean >= 0.6:
                    color = 'orange'
                else:
                    color = 'red'
            else:  # MCC
                if mean >= 0.3:
                    color = 'green'
                elif mean >= 0.0:
                    color = 'orange'
                else:
                    color = 'red'
            
            # Add colored circle around point
            circle = plt.Circle((i, mean), 0.03, color=color, fill=False, linewidth=3, alpha=0.8)
            ax_main.add_patch(circle)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                      markersize=10, label='Mean Score', alpha=0.8),
            plt.Line2D([0], [0], color='#A23B72', linestyle='--', linewidth=2, 
                      label='±1 Std Dev', alpha=0.7),
            plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, 
                      label='0.5 Reference', alpha=0.5)
        ]
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()


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
        """
        n_results = len(complete_results_list)
        if n_results < 2:
            raise ValueError("At least 2 results are required for comparison")
        
        # Generate default labels if not provided
        if labels is None:
            labels = [f"Condition {i+1}" for i in range(n_results)]
        elif len(labels) != n_results:
            raise ValueError("Number of labels must match number of results")
        
        # Extract data from all results (both nested CV and LOOCV now use same format)
        all_best_params = []
        all_final_results = []
        all_configs = []
        
        for complete_results in complete_results_list:
            # All results now have consistent format
            all_best_params.append(complete_results['best_hyperparameters']['model'])
            all_final_results.append(complete_results['final_performance']['model'])
            all_configs.append(complete_results['config'])
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create figure with single plot
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)
        
        # Main title
        fig.suptitle(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', y=0.95)
        
        # Define colors for different conditions
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B4A9C', '#2F9B69', '#FF6B6B', '#4ECDC4']
        if n_results > len(colors):
            # Generate more colors if needed
            import matplotlib.cm as cm
            colors = [cm.Set3(i/n_results) for i in range(n_results)]
        
        # Metrics to plot
        metrics = ['auc', 'accuracy', 'balanced_accuracy', 'f1', 'mcc', 'precision', 'recall']
        metric_labels = ['AUC', 'Accuracy', 'Bal. Acc.', 'F1', 'MCC', 'Precision', 'Recall']
        
        # Set up the main plot
        x_pos = np.arange(len(metrics))
        width = 0.7 / n_results  # Slightly narrower bars for better spacing
        
        # Plot bars for each condition
        for i, (final_results, label, color) in enumerate(zip(all_final_results, labels, colors[:n_results])):
            means = [final_results[metric]['mean'] for metric in metrics]
            stds = [final_results[metric]['std'] for metric in metrics]
            
            # Calculate x positions for this condition
            x_offset = x_pos + (i - (n_results-1)/2) * width
            
            # Plot bars
            bars = ax_main.bar(x_offset, means, width, 
                             label=label, color=color, alpha=0.8, 
                             edgecolor='black', linewidth=0.8)
            
            # Plot error bars
            ax_main.errorbar(x_offset, means, yerr=stds, 
                           fmt='none', color='black', alpha=0.7, capsize=2, linewidth=1)
        
        # Connect points with dashed lines for each condition
        for i, (final_results, color) in enumerate(zip(all_final_results, colors[:n_results])):
            means = [final_results[metric]['mean'] for metric in metrics]
            x_offset = x_pos + (i - (n_results-1)/2) * width
            ax_main.plot(x_offset, means, color=color, linestyle='--', 
                        linewidth=1.5, alpha=0.8, marker='o', markersize=3)
        
        # Customize main plot
        ax_main.set_xlabel('Performance Metrics', fontsize=11, fontweight='bold')
        ax_main.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax_main.set_xticks(x_pos)
        ax_main.set_xticklabels(metric_labels, rotation=0, ha='center', fontsize=10, fontweight='bold')
        ax_main.grid(True, alpha=0.3, axis='y')
        ax_main.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Make y-axis tick labels bold
        ax_main.tick_params(axis='y', labelsize=9)
        for tick in ax_main.get_yticklabels():
            tick.set_fontweight('bold')
        
        # Add horizontal reference lines
        ax_main.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Zero Baseline')
        ax_main.axhline(y=0.5, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Random Baseline')
        
        # Set y-axis limits with better spacing (allow negative values for MCC)
        all_means = []
        all_stds = []
        for final_results in all_final_results:
            for metric in metrics:
                all_means.append(final_results[metric]['mean'])
                all_stds.append(final_results[metric]['std'])
        
        # Calculate y-axis limits allowing for negative MCC values
        # Handle NaN values by filtering them out
        valid_data = []
        for m, s in zip(all_means, all_stds):
            if not (np.isnan(m) or np.isnan(s)):
                valid_data.extend([m - s, m + s])
        
        if valid_data:  # Only proceed if we have valid data
            y_min_data = min(valid_data)
            y_max_data = max(valid_data)
            
            # Allow negative values (important for MCC which can be negative)
            y_min = max(-1, y_min_data - 0.05)  # MCC can go down to -1
            y_max = min(1, y_max_data + 0.08)   # Most metrics max at 1
        else:
            # Fallback if all data is NaN
            y_min = -1
            y_max = 1
            print("Warning: All metric values are NaN, using default y-axis limits [-1, 1]")
        
        ax_main.set_ylim(y_min, y_max)
        
        # Add text annotations for best performers
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Handle NaN values when finding best performer
            metric_means = [results[metric]['mean'] for results in all_final_results]
            # Replace NaN with negative infinity for argmax to work properly
            valid_means = [m if not np.isnan(m) else -np.inf for m in metric_means]
            
            if any(m != -np.inf for m in valid_means):  # Only if we have at least one valid value
                best_idx = np.argmax(valid_means)
                best_score = all_final_results[best_idx][metric]['mean']
                best_std = all_final_results[best_idx][metric]['std']
                
                # Only add annotation if both score and std are not NaN
                if not (np.isnan(best_score) or np.isnan(best_std)):
                    x_offset = x_pos[i] + (best_idx - (n_results-1)/2) * width
                    ax_main.annotate(f'★', xy=(x_offset, best_score + best_std), 
                                   xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                                   fontsize=8, color='gold', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print concise comparison
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for metric, label in zip(metrics, metric_labels):
            row = {'Metric': label}
            for i, (results, condition_label) in enumerate(zip(all_final_results, labels)):
                mean = results[metric]['mean']
                std = results[metric]['std']
                # Handle NaN values in formatting
                if np.isnan(mean) or np.isnan(std):
                    row[condition_label] = 'NaN±NaN'
                else:
                    row[condition_label] = f'{mean:.3f}±{std:.3f}'
            
            # Add best performer (handle NaN values)
            metric_means = [results[metric]['mean'] for results in all_final_results]
            valid_means = [m if not np.isnan(m) else -np.inf for m in metric_means]
            
            if any(m != -np.inf for m in valid_means):
                best_idx = np.argmax(valid_means)
                row['Winner'] = labels[best_idx]
            else:
                row['Winner'] = 'N/A (all NaN)'
            comparison_data.append(row)
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Overall winner summary
        metric_winners = []
        for metric in metrics:
            metric_means = [results[metric]['mean'] for results in all_final_results]
            valid_means = [m if not np.isnan(m) else -np.inf for m in metric_means]
            
            if any(m != -np.inf for m in valid_means):
                best_idx = np.argmax(valid_means)
                metric_winners.append(best_idx)
        
        from collections import Counter
        if metric_winners:  # Only if we have valid winners
            winner_counts = Counter(metric_winners)
            overall_winner_idx = winner_counts.most_common(1)[0][0]
            print(f"\nOverall Best: {labels[overall_winner_idx]} (wins {winner_counts[overall_winner_idx]}/{len(metric_winners)} valid metrics)")
        else:
            print(f"\nOverall Best: N/A (all metrics have NaN values)")
            overall_winner_idx = 0  # Default to first label for consistency
        
        # Prepare hyperparameters information
        hyperparameters_info = {}
        for i, (params, label) in enumerate(zip(all_best_params, labels)):
            hyperparameters_info[label] = params
        
        # Prepare comparison results
        if metric_winners:
            overall_winner_metrics_won = winner_counts[overall_winner_idx]
            metric_winners_dict = dict(zip(metric_labels[:len(metric_winners)], [labels[i] for i in metric_winners]))
        else:
            overall_winner_metrics_won = 0
            metric_winners_dict = {label: 'N/A (NaN)' for label in metric_labels}
        
        comparison_results = {
            'figure': fig,
            'comparison_table': comparison_df,
            'hyperparameters': hyperparameters_info,
            'overall_winner': {
                'label': labels[overall_winner_idx],
                'metrics_won': overall_winner_metrics_won,
                'total_metrics': len(metrics)
            },
            'metric_winners': metric_winners_dict,
            'labels': labels,
            'colors': colors[:n_results]
        }
        
        return comparison_results

    
    @staticmethod
    def analyze_feature_robustness(complete_results, 
                                 title="Feature Selection Consistency Distribution",
                                 figsize=(16, 10),
                                 top_n_features=50):
        """
        Analyze and visualize feature selection consistency across outer folds.
        Creates a beautiful and clear visualization optimized for 100 features.
        
        Args:
            complete_results: Dictionary containing complete nested CV results
            title: Main title for the analysis
            figsize: Figure size tuple
            top_n_features: Number of top consistent features to highlight
            
        Returns:
            dict: Analysis results including robustness metrics and df_feat_robust
        """
        
        # Extract feature information from each fold
        fold_features = {}
        all_features = set()
        all_df_feat = []
        
        for fold_result in complete_results['outer_fold_results']:
            fold_id = fold_result['outer_fold']
            df_feat = fold_result['feature_info']['features']
            features = df_feat['feature'].tolist()
            fold_features[fold_id] = set(features)
            all_features.update(features)
            all_df_feat.append(df_feat)
        
        n_folds = len(fold_features)
        all_features = list(all_features)
        n_total_features = len(all_features)
        
        # Calculate feature frequency across folds
        feature_counts = {}
        for feature in all_features:
            count = sum(1 for fold_feats in fold_features.values() if feature in fold_feats)
            feature_counts[feature] = count
        
        # Sort features by consistency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get features that appear in ALL folds
        robust_features = [feature for feature, count in sorted_features if count == n_folds]
        
        # Create df_feat_robust with features that appear in all folds
        if robust_features and len(all_df_feat) > 0:
            # Use the first fold's df_feat as template and filter for robust features
            df_feat_robust = all_df_feat[0][all_df_feat[0]['feature'].isin(robust_features)].copy()
            df_feat_robust = df_feat_robust.reset_index(drop=True)
            print(f"\nGenerated df_feat_robust with {len(df_feat_robust)} robust features (appear in all {n_folds} folds)")
        else:
            df_feat_robust = None
            print(f"\nNo features appear in all {n_folds} folds")
        
        # Calculate robustness metrics
        features_in_all_folds = len(robust_features)
        features_in_majority = sum(1 for count in feature_counts.values() if count >= (n_folds + 1) // 2)
        
        # Create beautiful and clean single plot visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Main title with enhanced styling
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        
        # Feature consistency distribution (enhanced and more readable)
        overlap_counts = [0] * (n_folds + 1)
        for count in feature_counts.values():
            overlap_counts[count] += 1
        
        x_pos = range(1, n_folds + 1)
        # Beautiful gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(x_pos)))
        bars = ax.bar(x_pos, overlap_counts[1:], color=colors, alpha=0.8, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        # Enhanced value labels with better positioning
        max_height = max(overlap_counts) if overlap_counts else 1
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max_height*0.03,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontsize=14, fontweight='bold', color='black')
        
        # Improved axis formatting
        ax.set_xlabel('Number of Folds Feature Appears In', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_ylabel('Number of Features', fontsize=16, fontweight='bold', labelpad=15)
        
        # Better tick formatting - only show key x-axis values to avoid crowding
        if n_folds <= 10:
            # Show all folds if 10 or fewer
            tick_positions = x_pos
        elif n_folds <= 20:
            # Show every 2nd fold if 20 or fewer
            tick_positions = [i for i in x_pos if i % 2 == 0 or i == 1 or i == n_folds]
        else:
            # For many folds, show at intervals of 5
            interval = max(1, n_folds // 10)  # Aim for ~10 ticks
            if interval >= 5:
                interval = ((interval + 4) // 5) * 5  # Round to nearest 5
            tick_positions = [i for i in x_pos if i % interval == 0 or i == 1 or i == n_folds]
            
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{i}' for i in tick_positions], fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        
        # Improved grid and styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Highlight the "all folds" bar with better positioning
        if overlap_counts[n_folds] > 0:
            bars[-1].set_color('#FF6B6B')
            bars[-1].set_alpha(0.9)
            bars[-1].set_linewidth(3)
            ax.text(n_folds, overlap_counts[n_folds] + max_height*0.15,
                   f'Robust Features\n({overlap_counts[n_folds]})', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#FF6B6B', alpha=0.8, edgecolor='darkred'))
        
        # Add summary statistics box
        summary_text = f"""
• Total Features Found: {n_total_features:,}
• Features per Fold: {complete_results['config']['n_filter']}
• Robust Features (all folds): {features_in_all_folds}
• Robustness Rate: {features_in_all_folds/complete_results['config']['n_filter']*100:.1f}%"""
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=11, ha='left', va='top', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.7", facecolor='lightblue', alpha=0.9, edgecolor='navy'))
        
        # Set better y-axis limits with padding
        ax.set_ylim(0, max_height * 1.3)
        ax.set_xlim(0.4, n_folds + 0.6)
        
        plt.tight_layout()
        plt.show()
        
        # Enhanced detailed analysis printout
        print("\n" + "="*100)
        print("FEATURE ROBUSTNESS ANALYSIS RESULTS")
        print("="*100)
        
        # Calculate pairwise overlaps
        pairwise_overlaps = []
        fold_ids = list(fold_features.keys())
        for i in range(len(fold_ids)):
            for j in range(i+1, len(fold_ids)):
                overlap = len(fold_features[fold_ids[i]] & fold_features[fold_ids[j]])
                pairwise_overlaps.append(overlap)
        
        avg_pairwise_overlap = np.mean(pairwise_overlaps)
        std_pairwise_overlap = np.std(pairwise_overlaps)
        
        print(f"   • Total Unique Features: {n_total_features:,}")
        print(f"   • Features per Fold: {complete_results['config']['n_filter']}")
        print(f"   • Features in ALL folds: {features_in_all_folds} ({features_in_all_folds/complete_results['config']['n_filter']*100:.1f}%)")
        print(f"   • Features in majority: {features_in_majority} ({features_in_majority/complete_results['config']['n_filter']*100:.1f}%)")
        print(f"   • Average pairwise overlap: {avg_pairwise_overlap:.1f} ± {std_pairwise_overlap:.1f}")
        print(f"   • Robustness Score: {avg_pairwise_overlap/complete_results['config']['n_filter']*100:.0f}%")
        
        # Robustness assessment
        robustness_score = avg_pairwise_overlap/complete_results['config']['n_filter']
        if robustness_score > 0.7:
            assessment = "EXCELLENT - Very stable feature selection"
        elif robustness_score > 0.5:
            assessment = "GOOD - Moderately stable feature selection"
        elif robustness_score > 0.3:
            assessment = "MODERATE - Some variability in feature selection"
        else:
            assessment = "POOR - High variability in feature selection"
        
        print(f"   • Assessment: {assessment}")
        
        # Feature categories analysis
        print(f"\nFEATURE CATEGORIES ANALYSIS:")
        try:
            all_categories = []
            for fold_result in complete_results['outer_fold_results']:
                categories = fold_result['feature_info']['features']['category'].tolist()
                all_categories.extend(categories)
            
            from collections import Counter
            category_counts = Counter(all_categories)
            print("   Most common feature categories:")
            for cat, count in category_counts.most_common(5):
                print(f"     • {cat}: {count} selections ({count/(len(all_categories))*100:.1f}%)")
        except:
            print("   Category analysis not available")
        
        # Return comprehensive analysis results
        analysis_results = {
            'total_features': n_total_features,
            'features_per_fold': complete_results['config']['n_filter'],
            'features_in_all_folds': features_in_all_folds,
            'features_in_majority': features_in_majority,
            'avg_pairwise_overlap': avg_pairwise_overlap,
            'robustness_score': robustness_score,
            'robust_features': robust_features,
            'most_consistent_features': sorted_features[:50],
            'pairwise_overlaps': pairwise_overlaps,
            'df_feat_robust': df_feat_robust  # New: DataFrame with robust features
        }
        
        return analysis_results