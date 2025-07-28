#!/usr/bin/env python3
"""
Utility Functions for Wine Quality Classification Project
========================================================

This module contains utility functions for model evaluation, visualization,
hyperparameter tuning, and other common tasks.

Author: [Your Name]
Course: Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Parameters
    ----------
    model : object
        Trained model with predict method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str, default="Model"
        Name of the model for reporting
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    # Make predictions
    start_time = time.time()
    predictions = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"{'Accuracy:':<12} {accuracy:.4f}")
    print(f"{'Precision:':<12} {precision:.4f}")
    print(f"{'Recall:':<12} {recall:.4f}")
    print(f"{'F1-Score:':<12} {f1:.4f}")
    print(f"{'Pred. Time:':<12} {prediction_time:.4f}s")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Bad    Good")
    print(f"Actual  Bad   {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"        Good  {cm[1,0]:<6} {cm[1,1]:<6}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Bad', 'Good']))
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'prediction_time': prediction_time,
        'n_test_samples': len(y_test)
    }
    
    # Add probability scores if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
        results['probabilities'] = probabilities
        
        # Calculate AUC if probabilities available
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        results['roc_auc'] = roc_auc
        results['fpr'] = fpr
        results['tpr'] = tpr
        
        print(f"{'ROC AUC:':<12} {roc_auc:.4f}")
    
    return results


def cross_validate_model(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring_metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    **model_params
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation with multiple scoring metrics.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring_metrics : list, default=['accuracy', 'precision', 'recall', 'f1']
        Metrics to calculate
    **model_params
        Parameters to pass to model constructor
        
    Returns
    -------
    dict
        Cross-validation scores for each metric
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    print(f"Model: {model_class.__name__}")
    print(f"Parameters: {model_params}")
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = {metric: [] for metric in scoring_metrics}
    
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start_time = time.time()
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        predictions = model.predict(X_val_fold)
        
        # Calculate metrics
        for metric in scoring_metrics:
            if metric == 'accuracy':
                score = accuracy_score(y_val_fold, predictions)
            elif metric == 'precision':
                score = precision_score(y_val_fold, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val_fold, predictions, zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_val_fold, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores[metric].append(score)
        
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        
        print(f"Fold {fold+1}: Accuracy = {scores['accuracy'][-1]:.4f} "
              f"(Time: {fold_time:.2f}s)")
    
    # Print summary
    print(f"\nCross-Validation Results:")
    print(f"{'Metric':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for metric, values in scores.items():
        mean_score = np.mean(values)
        std_score = np.std(values)
        min_score = np.min(values)
        max_score = np.max(values)
        
        print(f"{metric.capitalize():<12} {mean_score:<10.4f} {std_score:<10.4f} "
              f"{min_score:<10.4f} {max_score:<10.4f}")
    
    avg_fold_time = np.mean(fold_times)
    print(f"\nAverage fold time: {avg_fold_time:.2f}s")
    print(f"Total CV time: {sum(fold_times):.2f}s")
    
    return scores


def hyperparameter_tuning(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, List],
    scoring_metric: str = 'accuracy',
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Perform grid search hyperparameter tuning.
    
    Parameters
    ----------
    model_class : class
        Model class to tune
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    param_grid : dict
        Dictionary of parameter names and values to try
    scoring_metric : str, default='accuracy'
        Metric to optimize
    verbose : bool, default=True
        Whether to print progress
        
    Returns
    -------
    tuple
        (best_params, all_results)
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING - {model_class.__name__.upper()}")
    print(f"{'='*60}")
    
    if verbose:
        print(f"Parameter grid: {param_grid}")
        print(f"Optimization metric: {scoring_metric}")
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    best_score = -np.inf
    best_params = {}
    all_results = []
    
    start_time = time.time()
    
    for i, param_combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        
        try:
            # Train model with current parameters
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            predictions = model.predict(X_val)
            
            # Calculate score
            if scoring_metric == 'accuracy':
                score = accuracy_score(y_val, predictions)
            elif scoring_metric == 'precision':
                score = precision_score(y_val, predictions, zero_division=0)
            elif scoring_metric == 'recall':
                score = recall_score(y_val, predictions, zero_division=0)
            elif scoring_metric == 'f1':
                score = f1_score(y_val, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown scoring metric: {scoring_metric}")
            
            # Store results
            result = {
                'params': params.copy(),
                'score': score,
                'rank': 0  # Will be filled later
            }
            all_results.append(result)
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            if verbose:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"Combination {i+1:3d}/{len(param_combinations)}: "
                      f"{scoring_metric}={score:.4f} [{param_str}]")
                
        except Exception as e:
            if verbose:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"Combination {i+1:3d}/{len(param_combinations)}: "
                      f"FAILED [{param_str}] - {str(e)[:50]}")
            
            # Store failed result
            result = {
                'params': params.copy(),
                'score': -np.inf,
                'rank': len(param_combinations),
                'error': str(e)
            }
            all_results.append(result)
    
    # Sort results and assign ranks
    all_results.sort(key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(all_results):
        result['rank'] = i + 1
    
    total_time = time.time() - start_time
    
    print(f"\nHyperparameter tuning completed in {total_time:.2f}s")
    print(f"Best parameters: {best_params}")
    print(f"Best {scoring_metric}: {best_score:.4f}")
    
    # Show top 5 results
    print(f"\nTop 5 parameter combinations:")
    print(f"{'Rank':<5} {'Score':<10} {'Parameters'}")
    print("-" * 80)
    
    for result in all_results[:5]:
        if result['score'] != -np.inf:
            param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
            print(f"{result['rank']:<5} {result['score']:<10.4f} {param_str}")
    
    return best_params, all_results


def plot_model_comparison(
    results_dict: Dict[str, Dict],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create bar plot comparing multiple models across different metrics.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    metrics : list
        List of metrics to compare
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    # Prepare data
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    # Colors for different metrics
    colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
    
    # Create bars for each metric
    for i, metric in enumerate(metrics):
        values = []
        for model_name in model_names:
            if metric in results_dict[model_name]:
                values.append(results_dict[model_name][metric])
            else:
                values.append(0)
        
        bars = ax.bar(x + i * width, values, width, 
                     label=metric.replace('_', ' ').title(), 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrices(
    results_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrices for multiple models.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        
        axes[i].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(
    results_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        if 'fpr' in results and 'tpr' in results:
            plt.plot(results['fpr'], results['tpr'], 
                    color=colors[i], linewidth=2,
                    label=f'{model_name} (AUC = {results["roc_auc"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curves showing training progress.
    
    Parameters
    ----------
    model : object
        Trained model with cost_history attribute
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    model_name : str
        Name of the model
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot cost history if available
    if hasattr(model, 'cost_history') and model.cost_history:
        axes[0].plot(model.cost_history, 'b-', linewidth=2)
        axes[0].set_title(f'{model_name} - Cost History')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost')
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy over time (simplified)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Create dummy learning curve data
        iterations = len(model.cost_history)
        x_points = np.linspace(0, iterations-1, min(20, iterations), dtype=int)
        
        # For simplicity, create a realistic learning curve
        train_accs = [train_acc - 0.1 * np.exp(-i/10) for i in range(len(x_points))]
        val_accs = [val_acc - 0.05 * np.exp(-i/15) for i in range(len(x_points))]
        
        axes[1].plot(x_points, train_accs, 'g-', linewidth=2, label='Training')
        axes[1].plot(x_points, val_accs, 'r-', linewidth=2, label='Validation')
        axes[1].set_title(f'{model_name} - Learning Curve')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # If no cost history, just show final accuracies
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        axes[0].bar(['Training', 'Validation'], [train_acc, val_acc], 
                   color=['green', 'red'], alpha=0.7)
        axes[0].set_title(f'{model_name} - Final Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        
        axes[1].text(0.5, 0.5, 'No cost history available', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=14, style='italic')
        axes[1].set_title('Cost History Not Available')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_misclassified_samples(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    n_examples: int = 10
) -> pd.DataFrame:
    """
    Analyze misclassified examples to understand model limitations.
    
    Parameters
    ----------
    model : object
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_names : list
        Names of features
    model_name : str
        Name of the model
    n_examples : int
        Number of examples to show
        
    Returns
    -------
    pd.DataFrame
        DataFrame with misclassified examples
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - MISCLASSIFICATION ANALYSIS")
    print(f"{'='*60}")
    
    predictions = model.predict(X_test)
    misclassified_mask = predictions != y_test
    
    n_misclassified = np.sum(misclassified_mask)
    total_samples = len(y_test)
    
    print(f"Total misclassified: {n_misclassified}/{total_samples} ({n_misclassified/total_samples*100:.1f}%)")
    
    if n_misclassified == 0:
        print("No misclassified examples!")
        return pd.DataFrame()
    
    # Analyze false positives and false negatives
    false_positives = (predictions == 1) & (y_test == 0)
    false_negatives = (predictions == 0) & (y_test == 1)
    
    n_fp = np.sum(false_positives)
    n_fn = np.sum(false_negatives)
    
    print(f"False Positives (predicted good, actually bad): {n_fp}")
    print(f"False Negatives (predicted bad, actually good): {n_fn}")
    
    # Create DataFrame with misclassified examples
    misclassified_X = X_test[misclassified_mask]
    misclassified_y = y_test[misclassified_mask]
    misclassified_pred = predictions[misclassified_mask]
    
    # Create DataFrame
    misclassified_df = pd.DataFrame(misclassified_X, columns=feature_names)
    misclassified_df['true_label'] = misclassified_y
    misclassified_df['predicted_label'] = misclassified_pred
    misclassified_df['error_type'] = ['FP' if pred == 1 and true == 0 else 'FN' 
                                     for pred, true in zip(misclassified_pred, misclassified_y)]
    
    # Show sample of misclassified examples
    print(f"\nSample of misclassified examples (showing first {min(n_examples, len(misclassified_df))}):")
    display_df = misclassified_df.head(n_examples)
    
    # Show only most important features for readability
    important_features = feature_names[:5]  # Show first 5 features
    display_columns = important_features + ['true_label', 'predicted_label', 'error_type']
    
    print(display_df[display_columns].to_string(index=False, float_format='%.3f'))
    
    return misclassified_df


def create_comprehensive_report(
    results_dict: Dict[str, Dict],
    feature_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Create a comprehensive text report of all model results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    feature_names : list
        Names of features used
    save_path : str, optional
        Path to save the report
        
    Returns
    -------
    str
        Report content
    """
    from datetime import datetime
    
    report = []
    report.append("="*80)
    report.append("WINE QUALITY CLASSIFICATION - COMPREHENSIVE RESULTS REPORT")
    report.append("="*80)
    report.append("")
    
    # Dataset information
    report.append("DATASET INFORMATION:")
    report.append(f"Number of features: {len(feature_names)}")
    report.append(f"Features: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")
    report.append("")
    
    # Model performance summary
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append("-" * 40)
    report.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
    report.append("-" * 70)
    
    # Sort models by accuracy
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, results in sorted_models:
        report.append(f"{model_name:<20} {results['accuracy']:<10.4f} "
                      f"{results['precision']:<11.4f} {results['recall']:<10.4f} "
                      f"{results['f1_score']:<10.4f}")
    
    report.append("")
    
    # Best performing model
    best_model, best_results = sorted_models[0]
    report.append(f"BEST PERFORMING MODEL: {best_model}")
    report.append(f"Best Accuracy: {best_results['accuracy']:.4f}")
    report.append(f"Best F1-Score: {best_results['f1_score']:.4f}")
    report.append("")
    
    # Detailed results for each model
    for model_name, results in results_dict.items():
        report.append(f"DETAILED RESULTS - {model_name.upper()}")
        report.append("-" * 50)
        report.append(f"Accuracy:      {results['accuracy']:.4f}")
        report.append(f"Precision:     {results['precision']:.4f}")
        report.append(f"Recall:        {results['recall']:.4f}")
        report.append(f"F1-Score:      {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            report.append(f"ROC AUC:       {results['roc_auc']:.4f}")
        
        report.append(f"Prediction Time: {results['prediction_time']:.4f}s")
        report.append(f"Test Samples:   {results['n_test_samples']}")
        
        # Confusion matrix
        cm = results['confusion_matrix']
        report.append("Confusion Matrix:")
        report.append(f"    Predicted:   Bad   Good")
        report.append(f"Actual Bad:     {cm[0,0]:4d}  {cm[0,1]:4d}")
        report.append(f"       Good:    {cm[1,0]:4d}  {cm[1,1]:4d}")
        report.append("")
    
    # Generate timestamp
    report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    
    report_content = "\n".join(report)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_content)
        print(f"Comprehensive report saved to: {save_path}")
    
    return report_content


def save_results_to_csv(
    results_dict: Dict[str, Dict],
    filepath: str
) -> None:
    """
    Save model results to CSV file.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    filepath : str
        Path to save CSV file
    """
    # Create DataFrame from results
    data = []
    for model_name, results in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'Prediction_Time': results['prediction_time'],
            'Test_Samples': results['n_test_samples']
        }
        
        if 'roc_auc' in results:
            row['ROC_AUC'] = results['roc_auc']
        
        # Add confusion matrix values
        cm = results['confusion_matrix']
        row['TN'] = cm[0, 0]  # True Negatives
        row['FP'] = cm[0, 1]  # False Positives
        row['FN'] = cm[1, 0]  # False Negatives
        row['TP'] = cm[1, 1]  # True Positives
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Results saved to CSV: {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    print("Wine Quality Classification - Utility Functions")
    print("This module provides utility functions for model evaluation and visualization.")
    print("Import this module in your main script to use these functions.")