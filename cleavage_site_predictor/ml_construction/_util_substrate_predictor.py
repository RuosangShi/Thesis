#!/usr/bin/env python3
"""
Substrate Predictor Plotting Utilities
====================================

Plotting functions for substrate prediction evaluation and threshold calibration.
Enhanced with bold formatting and .2f value labels as requested.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score


def create_clean_evaluation_plots(substrate_predictions: Dict, nonsubstrate_predictions: Dict,
                                performance_title: str = "Performance Comparison of Substrate Prediction Methods",
                                confusion_title: str = "Confusion Matrices: Substrate vs Non-Substrate Classification",
                                calibration_results: Dict = None):
    """
    Create two plots: 1) Performance comparison (threshold 0.5 vs calibrated), 2) Calibrated confusion matrices.
    
    Args:
        substrate_predictions: Predictions for substrate proteins
        nonsubstrate_predictions: Predictions for non-substrate proteins
        performance_title: Title for performance comparison plot
        confusion_title: Title for confusion matrices plot
        calibration_results: Calibration results with optimized thresholds
    """
    
    # Extract method data from predictions structure
    methods_data = {}
    
    # Build methods list from actual data structure
    if 'top_k_pooling' in substrate_predictions:
        for k in ['top_1', 'top_3', 'top_5']:
            if k in substrate_predictions['top_k_pooling']:
                method_name = f'top_{k.split("_")[1]}'
                sub_data = substrate_predictions['top_k_pooling'][k]['threshold_0.50']['protein_scores']
                nonsub_data = nonsubstrate_predictions['top_k_pooling'][k]['threshold_0.50']['protein_scores']
                
                # Combine data
                y_prob = []
                y_true = []
                for protein_id, score in sub_data.items():
                    y_prob.append(float(score))
                    y_true.append(1)
                for protein_id, score in nonsub_data.items():
                    y_prob.append(float(score))
                    y_true.append(0)
                
                methods_data[method_name] = {
                    'label': f'Top-{k.split("_")[1]}',
                    'y_true': np.array(y_true),
                    'y_prob': np.array(y_prob)
                }
    
    if 'poisson_binomial' in substrate_predictions:
        sub_data = substrate_predictions['poisson_binomial']['threshold_0.50']['protein_scores']
        nonsub_data = nonsubstrate_predictions['poisson_binomial']['threshold_0.50']['protein_scores']
        
        y_prob = []
        y_true = []
        for protein_id, score in sub_data.items():
            y_prob.append(float(score))
            y_true.append(1)
        for protein_id, score in nonsub_data.items():
            y_prob.append(float(score))
            y_true.append(0)
        
        methods_data['poisson_binomial'] = {
            'label': 'Poisson Binomial',
            'y_true': np.array(y_true),
            'y_prob': np.array(y_prob)
        }
    
    # PLOT 1: Performance Comparison (Threshold 0.5 vs Calibrated)
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    
    methods = list(methods_data.keys())
    method_labels = [methods_data[m]['label'] for m in methods]
    
    # Calculate metrics for threshold 0.5
    f1_05 = []
    auc_05 = []
    bal_acc_05 = []
    
    # Calculate metrics for calibrated thresholds
    f1_cal = []
    auc_cal = []
    bal_acc_cal = []
    calibrated_thresholds = []
    
    for method in methods:
        y_true = methods_data[method]['y_true']
        y_prob = methods_data[method]['y_prob']
        
        # Metrics at threshold 0.5
        y_pred_05 = (y_prob >= 0.5).astype(int)
        f1_05.append(f1_score(y_true, y_pred_05))
        auc_05.append(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5)
        bal_acc_05.append(balanced_accuracy_score(y_true, y_pred_05))
        
        # Metrics at calibrated threshold (handle CV results)
        # Check for CV results first (nested structure)
        method_data = None
        if calibration_results and 'cv_calibration_results' in calibration_results and method in calibration_results['cv_calibration_results']:
            method_data = calibration_results['cv_calibration_results'][method]
        elif calibration_results and method in calibration_results:
            method_data = calibration_results[method]
            
        if method_data:
            # Check if this is CV results or single-split results
            if 'mean' in str(method_data.get('best_threshold', {})):
                # CV results with mean and std
                cal_threshold = method_data['best_threshold']['mean']
            else:
                # Single-split results
                cal_threshold = method_data['best_threshold']
                
            calibrated_thresholds.append(cal_threshold)
            y_pred_cal = (y_prob >= cal_threshold).astype(int)
            f1_cal.append(f1_score(y_true, y_pred_cal))
            auc_cal.append(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5)
            bal_acc_cal.append(balanced_accuracy_score(y_true, y_pred_cal))
        else:
            calibrated_thresholds.append(0.5)
            f1_cal.append(f1_05[-1])
            auc_cal.append(auc_05[-1])
            bal_acc_cal.append(bal_acc_05[-1])
    
    # Extract error bar data for CV results
    f1_cal_err = []
    auc_cal_err = []
    bal_acc_cal_err = []
    threshold_errors = []
    
    for method in methods:
        # Check for CV results first (nested structure)
        method_data = None
        if calibration_results and 'cv_calibration_results' in calibration_results and method in calibration_results['cv_calibration_results']:
            method_data = calibration_results['cv_calibration_results'][method]
        elif calibration_results and method in calibration_results:
            method_data = calibration_results[method]
            
        if method_data:
            # Check if CV results available
            if 'mean' in str(method_data.get('best_threshold', {})):
                # CV results - extract std deviations
                threshold_errors.append(method_data['best_threshold']['std'])
                f1_cal_err.append(method_data['test_metrics']['f1']['std'])
                auc_cal_err.append(0.0)  # AUC std not computed in CV
                bal_acc_cal_err.append(method_data['test_metrics']['balanced_accuracy']['std'])
            else:
                # Single-split results - no error bars
                threshold_errors.append(0.0)
                f1_cal_err.append(0.0)
                auc_cal_err.append(0.0)
                bal_acc_cal_err.append(0.0)
        else:
            threshold_errors.append(0.0)
            f1_cal_err.append(0.0)
            auc_cal_err.append(0.0)
            bal_acc_cal_err.append(0.0)
    
    # Create grouped bar plot
    x = np.arange(len(methods))
    width = 0.12
    
    # Threshold 0.5 results (no error bars)
    bars1 = ax1.bar(x - 1.5*width, f1_05, width, label='F1 (θ=0.5)', alpha=0.7, color='#1f77b4')
    bars2 = ax1.bar(x - 0.5*width, auc_05, width, label='AUC (θ=0.5)', alpha=0.7, color='#ff7f0e')
    bars3 = ax1.bar(x + 0.5*width, bal_acc_05, width, label='Bal_Acc (θ=0.5)', alpha=0.7, color='#2ca02c')
    
    # Calibrated results (with error bars if CV)
    bars4 = ax1.bar(x + 1.5*width, f1_cal, width, yerr=f1_cal_err, capsize=3,
                   label='F1 (Calibrated)', alpha=0.9, color='#1f77b4', edgecolor='black', linewidth=1)
    bars5 = ax1.bar(x + 2.5*width, auc_cal, width, yerr=auc_cal_err, capsize=3,
                   label='AUC (Calibrated)', alpha=0.9, color='#ff7f0e', edgecolor='black', linewidth=1)
    bars6 = ax1.bar(x + 3.5*width, bal_acc_cal, width, yerr=bal_acc_cal_err, capsize=3,
                   label='Bal_Acc (Calibrated)', alpha=0.9, color='#2ca02c', edgecolor='black', linewidth=1)
    
    # Add value labels with .2f format
    all_bars = [bars1, bars2, bars3, bars4, bars5, bars6]
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Add threshold information as text (with CV statistics if available)
    threshold_text = "Calibrated Thresholds:\n"
    is_cv_results = False
    
    for i, (method, threshold) in enumerate(zip(method_labels, calibrated_thresholds)):
        # Check for CV results first (nested structure)
        method_data = None
        if calibration_results and 'cv_calibration_results' in calibration_results and methods[i] in calibration_results['cv_calibration_results']:
            method_data = calibration_results['cv_calibration_results'][methods[i]]
        elif calibration_results and methods[i] in calibration_results:
            method_data = calibration_results[methods[i]]
            
        if method_data:
            if 'mean' in str(method_data.get('best_threshold', {})):
                # CV results - show mean ± std
                thresh_std = threshold_errors[i]
                threshold_text += f"{method}: θ={threshold:.2f}±{thresh_std:.2f} "
                is_cv_results = True
            else:
                # Single-split results
                threshold_text += f"{method}: θ={threshold:.2f} "
        else:
            threshold_text += f"{method}: θ={threshold:.2f} "
    
    if is_cv_results:
        threshold_text += "\n(CV mean±std shown)"
    
    # Position text box
    ax1.text(0.02, 0.98, threshold_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontweight='bold')
    
    ax1.set_xlabel('Aggregation Methods', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title(performance_title, fontweight='bold', fontsize=14, pad=20)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(method_labels, fontweight='bold')
    
    legend1 = ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    for text in legend1.get_texts():
        text.set_fontweight('bold')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Make tick labels bold
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    # PLOT 2: Confusion Matrices with Calibrated Thresholds (with CV deviations if available)
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        if i >= 4:  # Only show first 4 methods
            break
            
        ax = axes[i]
        
        # Check if CV confusion matrix data is available
        # Handle both direct calibration_results and nested cv_calibration_results
        method_results = None
        if calibration_results and method in calibration_results:
            method_results = calibration_results[method]
        elif (calibration_results and 'cv_calibration_results' in calibration_results and 
              method in calibration_results['cv_calibration_results']):
            method_results = calibration_results['cv_calibration_results'][method]
        
        if method_results and 'confusion_matrix_cv' in method_results:
            
            # Use CV confusion matrix with mean ± std
            cv_cm_data = method_results['confusion_matrix_cv']
            cm_mean = cv_cm_data['mean']
            cm_std = cv_cm_data['std']
            
            # Create custom annotations with mean ± std
            annot_labels = np.array([
                [f"{cm_mean[0,0]:.1f}±{cm_std[0,0]:.1f}", f"{cm_mean[0,1]:.1f}±{cm_std[0,1]:.1f}"],
                [f"{cm_mean[1,0]:.1f}±{cm_std[1,0]:.1f}", f"{cm_mean[1,1]:.1f}±{cm_std[1,1]:.1f}"]
            ])
            
            # Create heatmap with mean values and custom annotations
            sns.heatmap(cm_mean, annot=annot_labels, fmt='', cmap='Blues', cbar=False,
                       xticklabels=['Non-Substrate', 'Substrate'],
                       yticklabels=['Non-Substrate', 'Substrate'], ax=ax,
                       annot_kws={'fontsize': 10, 'fontweight': 'bold'})
            
            # Title with threshold information (CV mean ± std)
            threshold_mean = calibrated_thresholds[i]
            threshold_std = threshold_errors[i]
            method_label = methods_data[method]['label']
            if threshold_std > 0:
                ax.set_title(f'{method_label}\nThreshold: {threshold_mean:.2f}±{threshold_std:.2f}\n(CV mean±std)', 
                           fontweight='bold', fontsize=11)
            else:
                ax.set_title(f'{method_label}\nThreshold: {threshold_mean:.2f}', 
                           fontweight='bold', fontsize=11)
                
        else:
            # Fallback: regular confusion matrix without CV
            y_true = methods_data[method]['y_true']
            y_prob = methods_data[method]['y_prob']
            
            # Use calibrated threshold
            threshold = calibrated_thresholds[i]
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create regular heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=['Non-Substrate', 'Substrate'],
                       yticklabels=['Non-Substrate', 'Substrate'], ax=ax)
            
            # Title with threshold information
            method_label = methods_data[method]['label']
            ax.set_title(f'{method_label}\nThreshold: {threshold:.2f}', fontweight='bold', fontsize=12)
        
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        
        # Make tick labels bold
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
    
    # Hide extra subplots
    for i in range(len(methods), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(confusion_title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


