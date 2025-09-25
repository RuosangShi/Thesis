#!/usr/bin/env python3
"""
Updated Feature Ablation Plotting Utilities with Two-Plot Layout and Error Bars
==============================================================================

This module contains improved plotting functions for feature ablation studies.
Creates two separate plots: one for removal experiments, one for baseline comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional

def plot_progressive_ablation_results(progressive_results: Dict[str, Any], 
                                    save_path: Optional[str] = None, title: Optional[str] = 'Progressive Feature Ablation: Impact of Removing Each Feature Group\n Baseline (All Features) vs. Remove One Group at a Time'):
    """
    Create two comprehensive plots for progressive ablation results with error bars.
    
    Plot 1: Full model vs removal experiments
    Plot 2: Full model vs single-feature baselines
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    print(" Creating Progressive Ablation Plots...")
    
    # Use performance_summary.experiments path directly
    if 'performance_summary' in progressive_results and 'experiments' in progressive_results['performance_summary']:
        aggregated = progressive_results['performance_summary']['experiments']
    else:
        print(" Expected structure: progressive_results['performance_summary']['experiments']")
        return
    
    if not aggregated:
        print(" No results found for progressive ablation plotting")
        return
    
    print(f"    Available experiments: {list(aggregated.keys())}")
    
    # Extract full model performance
    full_model_names = ['full_model', 'baseline_all', 'all_features']
    full_model_data = None
    
    for full_model_name in full_model_names:
        if full_model_name in aggregated:
            full_model_data = aggregated[full_model_name]
            print(f"    Found full model: '{full_model_name}'")
            break
    
    if not full_model_data:
        print(f" No full model results found. Tried: {full_model_names}")
        return
    
    key_metrics = ['f1', 'auc', 'balanced_accuracy']
    metric_colors = ['#E74C3C', '#3498DB', '#F39C12']  # Red, Blue, Orange
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=False)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    
    # PLOT 1: Full Model vs Removal Experiments
    _create_removal_plot(ax1, aggregated, full_model_data, key_metrics, metric_colors)
    
    # PLOT 2: Full Model vs Single-Feature Baselines  
    _create_baseline_plot(ax2, aggregated, full_model_data, key_metrics, metric_colors)
    
    # Create shared legend
    handles, labels = ax1.get_legend_handles_labels()
    if handles:  # Only create shared legend if we have handles
        legend = fig.legend(handles, labels,
                           loc='lower center', bbox_to_anchor=(0.5, -0.02),
                           ncol=len(labels), fontsize=14, frameon=True, fancybox=True, shadow=False)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        # Remove individual legends from subplots if they exist
        legend1 = ax1.get_legend()
        if legend1:
            legend1.remove()
        legend2 = ax2.get_legend()
        if legend2:
            legend2.remove()

    # Adjust layout to leave space for title and bottom legend
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.07, right=0.98, wspace=0.2)
    if save_path:
        plt.savefig(f"{save_path}_progressive_ablation_comprehensive.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.show()
    
    print(" Progressive ablation plots complete!")
    if save_path:
        print(f"📁 Progressive ablation plots saved as: {save_path}_progressive_ablation_comprehensive.png")


def _create_removal_plot(ax, aggregated, full_model_data, key_metrics, metric_colors):
    """Create plot comparing full model vs removal experiments."""
    
    # Extract removal experiments
    removal_experiments = []
    for exp_name in aggregated.keys():
        if exp_name.startswith('remove_'):
            removal_experiments.append(exp_name)
    
    if not removal_experiments:
        ax.text(0.5, 0.5, 'No removal experiments found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Full Model vs Feature Removal', fontweight='bold', fontsize=16)
        return
    
    print(f"    Found {len(removal_experiments)} removal experiments")
    
    # Collect data - NO SORTING
    experiment_names = ['Full Model']
    experiment_data = {'Full Model': full_model_data}
    
    # Add removal experiments in original order
    for exp_name in removal_experiments:
        # Create display name from experiment name
        display_name = exp_name.replace('remove_', 'Remove ').replace('_', ' ').title()
        experiment_names.append(display_name)
        experiment_data[display_name] = aggregated[exp_name]
    
    _plot_with_error_bars(ax, experiment_names, experiment_data, key_metrics, metric_colors,
                         title='Full Model vs Feature Removal',
                         xlabel='Feature Configurations')


def _create_baseline_plot(ax, aggregated, full_model_data, key_metrics, metric_colors):
    """Create plot comparing full model vs single-feature baselines."""
    
    # Look for baseline experiments
    baseline_experiments = []
    baseline_names_map = {
        'baseline_aa_index': 'AAIndex Only',
        'baseline_cpp': 'CPP Only', 
        'baseline_plm': 'PLM Only'
    }
    
    for exp_name, display_name in baseline_names_map.items():
        if exp_name in aggregated:
            baseline_experiments.append((exp_name, display_name))
    
    if not baseline_experiments:
        ax.text(0.5, 0.5, 'No baseline experiments found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Full Model vs Single-Feature Baselines', fontweight='bold', fontsize=16)
        return
    
    print(f"    Found {len(baseline_experiments)} baseline experiments")
    
    # Collect data - NO SORTING
    experiment_names = ['Full Model']
    experiment_data = {'Full Model': full_model_data}
    
    # Add baselines in original order
    for exp_name, display_name in baseline_experiments:
        experiment_names.append(display_name)
        experiment_data[display_name] = aggregated[exp_name]
    
    _plot_with_error_bars(ax, experiment_names, experiment_data, key_metrics, metric_colors,
                         title='Full Model vs Single-Feature Baselines',
                         xlabel='Model Configurations')


def _plot_with_error_bars(ax, experiment_names, experiment_data, key_metrics, metric_colors, 
                         title, xlabel):
    """Create bar plot with error bars for given experiments."""
    
    # Extract mean and std for each metric and experiment
    means = {metric: [] for metric in key_metrics}
    stds = {metric: [] for metric in key_metrics}
    
    for exp_name in experiment_names:
        exp_data = experiment_data[exp_name]
        
        # Extract metrics with error bars from aggregated results
        for metric in key_metrics:
            mean_val, std_val = _extract_metric_with_error(exp_data, metric)
            means[metric].append(mean_val)
            stds[metric].append(std_val)
    
    # Create grouped bar chart
    x_pos = np.arange(len(experiment_names)) * 1.15
    bar_width = 0.24
    
    for i, (metric, color) in enumerate(zip(key_metrics, metric_colors)):
        bars = ax.bar(x_pos + i * bar_width, means[metric], bar_width,
                     yerr=stds[metric], capsize=6,
                     label=f'{metric.upper()}', 
                     color=color, alpha=0.85, edgecolor='black', linewidth=0.6) 
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, means[metric], stds[metric]):
            if abs(mean_val) > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., 
                       bar.get_height() + std_val + 0.005,
                       f'{mean_val:.2f}', #\n±{std_val:.2f}
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=13)
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=13)
    ax.set_title(title, fontweight='bold', pad=10, fontsize=15)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(experiment_names, rotation=20, ha='right', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    
    # Increase y-axis limit for legend space
    current_ylim = ax.get_ylim()
    y_top = max(1.0, current_ylim[1] * 1.1)
    ax.set_ylim(0.0, y_top)

    # Remove per-axes legend to rely on shared legend
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Cleaner look
    ax.grid(True, alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _extract_metric_with_error(exp_data, metric):
    """Extract metric mean and std from experiment data."""
    
    # Extract from: logodds_average -> best_combination -> metrics
    if ('logodds_average' in exp_data and 
        'best_combination' in exp_data['logodds_average'] and
        'metrics' in exp_data['logodds_average']['best_combination']):
        
        metrics = exp_data['logodds_average']['best_combination']['metrics']
        
        if metric in metrics:
            metric_data = metrics[metric]
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mean_val = float(metric_data['mean'])
                std_val = float(metric_data.get('std', 0))
                print(f"    Found {metric}: {mean_val:.2f}±{std_val:.2f}")
                return mean_val, std_val
    
    print(f"    Could not find {metric}")
    return 0.0, 0.0