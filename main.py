import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


sys.path.append('src')

from preprocess import create_preprocessing_pipeline
from logistic_regression import LogisticRegression, cross_validate_lr
from svm import SVM, cross_validate_svm
from kernel_logistic_regression import KernelLogisticRegression
from utils import (
    evaluate_model, hyperparameter_tuning, plot_model_comparison,
    plot_confusion_matrices, plot_roc_curves, plot_learning_curves,
    analyze_misclassified_samples, create_comprehensive_report, save_results_to_csv,
    cross_validate_model 
)


np.random.seed(42)

def create_directories():
    
    directories = ['results', 'results/plots', 'results/reports']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("project directories done")

def main():
    
    print("WINE QUALITY CLASSIFICATION PROJECT")
    print("SVM vs Logistic Regression") 
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*80)
    

    config = {
        'red_wine_path': 'data/winequalityred.csv',
        'white_wine_path': 'data/winequalitywhite.csv',
        'test_size': 0.2,
        'validation_size': 0.2,
        'scaler_type': 'standard',
        'cv_folds': 5,
        'plots_dir': 'results/plots',
        'reports_dir': 'results/reports'
    }
    
    try:
        
        create_directories()
        
        # STEP 1: Data Preprocessing
        print(f"\n{'='*60}")
        print("STEP 1: DATA PREPROCESSING")
        print(f"{'='*60}")
        
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = create_preprocessing_pipeline(
            red_wine_path=config['red_wine_path'],
            white_wine_path=config['white_wine_path'],
            test_size=config['test_size'],
            validation_size=config['validation_size'],
            scaler_type=config['scaler_type'],
            create_plots=True,
            save_path=config['plots_dir']
        )
        
        print(f"\n Preprocessing done")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")  
        print(f"Test set: {X_test.shape}")
        print(f"Features: {len(feature_names)}")
        
        # STEP 2: Hyperparameter Tuning
        print(f"\n{'='*60}")
        print("STEP 2: HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        
        # Logistic Regression tuning
        print("\n Tuning Logistic Regression...")
        lr_param_grid = {
            'learning_rate': [0.01, 0.1, 1.0],
            'regularization': [None, 'l2'],
            'lambda_reg': [0.01, 0.1],
            'max_iterations': [1000]
        }
        
        best_lr_params, _ = hyperparameter_tuning(
            LogisticRegression, X_train, y_train, X_val, y_val,
            lr_param_grid, scoring_metric='accuracy', verbose=True
        )
        
        # SVM tuning
        print("\n Tuning SVM...")
        svm_param_grid = {
            'C': [1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale'],
            'max_iterations': [1000]
        }
        
        best_svm_params, _ = hyperparameter_tuning(
            SVM, X_train, y_train, X_val, y_val,
            svm_param_grid, scoring_metric='accuracy', verbose=True
        ) 

        # Kernel Logistic Regression (RBF) tuning
        print("\n Tuning Kernel Logistic Regression (RBF)...")
        klr_rbf_grid = {
            'kernel': ['rbf'],
            'gamma': [0.001, 0.01, 0.1, 'scale'],
            'learning_rate': [0.05, 0.1],
            'regularization': ['l2'],
            'lambda_reg': [1e-3, 3e-3, 1e-2],
            'max_iterations': [1000]
        }
        best_klr_rbf_params, _ = hyperparameter_tuning(
            KernelLogisticRegression, X_train, y_train, X_val, y_val,
            klr_rbf_grid, scoring_metric='f1', verbose=True
        )

        # Kernel Logistic Regression (Polynomial) tuning
        print("\n Tuning Kernel Logistic Regression (Polynomial)...")
        klr_poly_grid = {
            'kernel': ['poly'],
            'degree': [2, 3],
            'gamma': ['scale', 0.01],
            'coef0': [1.0],
            'learning_rate': [0.05, 0.1],
            'regularization': ['l2'],
            'lambda_reg': [1e-3, 3e-3, 1e-2],
            'max_iterations': [1000]
        }
        best_klr_poly_params, _ = hyperparameter_tuning(
            KernelLogisticRegression, X_train, y_train, X_val, y_val,
            klr_poly_grid, scoring_metric='f1', verbose=True
        )

        
        # STEP 3: Train Final Models
        print(f"\n{'='*60}")
        print("STEP 3: TRAINING FINAL MODELS")
        print(f"{'='*60}")
        
        models = {}
        
        # Train LR
        print("\n Training Logistic Regression...")
        lr_model = LogisticRegression(**best_lr_params, verbose=True)
        lr_model.fit(X_train, y_train)
        models['Logistic Regression'] = lr_model
        
        # Train SVMs
        print("\n Training SVM (Linear)...")
        svm_linear = SVM(C=best_svm_params['C'], kernel='linear', verbose=True)
        svm_linear.fit(X_train, y_train)
        models['SVM (Linear)'] = svm_linear
        
        print("\n Training SVM (RBF)...")
        svm_rbf = SVM(C=best_svm_params['C'], kernel='rbf', gamma='scale', verbose=True)
        svm_rbf.fit(X_train, y_train)
        models['SVM (RBF)'] = svm_rbf
        
        print("\n Training SVM (Polynomial)...")
        svm_poly = SVM(C=best_svm_params['C'], kernel='poly', degree=3, verbose=True)
        svm_poly.fit(X_train, y_train)
        models['SVM (Polynomial)'] = svm_poly

        print("\n Training Kernel LR (RBF)...")
        klr_rbf = KernelLogisticRegression(**best_klr_rbf_params, verbose=True)
        klr_rbf.fit(X_train, y_train)
        models['Kernel LR (RBF)'] = klr_rbf

        print("\n Training Kernel LR (Polynomial)...")
        klr_poly = KernelLogisticRegression(**best_klr_poly_params, verbose=True)
        klr_poly.fit(X_train, y_train)
        models['Kernel LR (Polynomial)'] = klr_poly

        
        print(f"\n Trained {len(models)} models successfully!")
        
        # STEP 4: Model Evaluation
        print(f"\n{'='*60}")
        print("STEP 4: MODEL EVALUATION")
        print(f"{'='*60}")
        
        results = {}
        for model_name, model in models.items():
            print(f"\n Evaluating {model_name}...")
            result = evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = result
        
        print(f"\n EVALUATION SUMMARY:")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for model_name, result in sorted_results:
            print(f"{model_name:<20} {result['accuracy']:<10.4f} "
                  f"{result['precision']:<11.4f} {result['recall']:<10.4f} "
                  f"{result['f1_score']:<10.4f}")
        
        best_model = sorted_results[0]
        print(f"\n Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        # STEP 5: Cross-Validation
        print(f"\n{'='*60}")
        print("STEP 5: CROSS-VALIDATION")
        print(f"{'='*60}")
        
        print(f"\n Performing {config['cv_folds']}-fold cross-validation for Logistic Regression...")
        lr_cv_scores = cross_validate_lr(X_train, y_train, config['cv_folds'], **best_lr_params)
        
        print(f"\nLogistic Regression CV Results:")
        for metric, scores in lr_cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {metric.capitalize()}: {mean_score:.4f} (±{std_score:.4f})")

        print(f"\n Performing {config['cv_folds']}-fold cross-validation for Kernel LR (RBF)...")
        klr_cv_scores = cross_validate_model(
            KernelLogisticRegression, X_train, y_train, cv_folds=config['cv_folds'],
            kernel='rbf',
            gamma=best_klr_rbf_params.get('gamma', 'scale'),
            learning_rate=best_klr_rbf_params.get('learning_rate', 0.05),
            regularization='l2',
            lambda_reg=best_klr_rbf_params.get('lambda_reg', 1e-2),
            max_iterations=best_klr_rbf_params.get('max_iterations', 1000)
        )
        print("\nKernel LR (RBF) CV Results:")
        for metric, scores in klr_cv_scores.items():
            print(f"  {metric.capitalize()}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        
        # STEP 6: Create Basic Visualizations
        print(f"\n{'='*60}")
        print("STEP 6: VISUALIZATIONS")
        print(f"{'='*60}")
        
        print("\n Creating model comparison plot...")
        plot_model_comparison(
            results, 
            save_path=os.path.join(config['plots_dir'], 'model_comparison.png')
        )
        
        print("\n Creating confusion matrices...")
        plot_confusion_matrices(
            results,
            save_path=os.path.join(config['plots_dir'], 'confusion_matrices.png')
        )
        
        print("\n Creating ROC curves...")
        plot_roc_curves(
            results,
            save_path=os.path.join(config['plots_dir'], 'roc_curves.png')
        )
        
        # Learning curves for LR
        if hasattr(lr_model, 'cost_history') and lr_model.cost_history:
            print(" Creating learning curve for Logistic Regression...")
            plot_learning_curves(
                lr_model, X_train, y_train, X_val, y_val, 
                "Logistic Regression",
                save_path=os.path.join(config['plots_dir'], 'lr_learning_curve.png')
            )
        
        # STEP 7: Error Analysis
        print(f"\n{'='*60}")
        print("STEP 7: ERROR ANALYSIS")
        print(f"{'='*60}")
        
        # Analyze errors for best model
        best_model_name = best_model[0]
        best_model_obj = models[best_model_name]
        
        print(f"\n Analyzing misclassified examples for {best_model_name}...")
        misclassified_df = analyze_misclassified_samples(
            best_model_obj, X_test, y_test, feature_names, 
            best_model_name, n_examples=5
        )
        
        # STEP 8: Generate Reports
        print(f"\n{'='*60}")
        print("STEP 8: GENERATING REPORTS")
        print(f"{'='*60}")
        
        print("\n Creating comprehensive report...")
        report_content = create_comprehensive_report(
            results, feature_names,
            save_path=os.path.join(config['reports_dir'], 'comprehensive_report.txt')
        )
        
        print("\n Saving results to CSV...")
        save_results_to_csv(
            results,
            os.path.join(config['reports_dir'], 'model_results.csv')
        )
        
        
        # FINAL SUMMARY
        print(f"\n{'='*80}")
        print(" PROJECT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print(f"\n BEST PERFORMING MODEL:")
        print(f"   Model:     {best_model[0]}")
        print(f"   Accuracy:  {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.1f}%)")
        print(f"   F1-Score:  {best_model[1]['f1_score']:.4f}")
        
        print(f"\n ALL MODELS RANKED:")
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"   {i}. {model_name:<20} Accuracy: {result['accuracy']:.4f}")
        
        print(f"\n OUTPUT LOCATIONS:")
        print(f"    Plots:   {config['plots_dir']}")
        print(f"    Reports: {config['reports_dir']}")
        

        
    except FileNotFoundError as e:
        print(f"\n DATASET ERROR: {e}")
        print("\n Please ensure these files exist:")
        print(f"   • {config['red_wine_path']}")
        print(f"   • {config['white_wine_path']}")
        print("\n Download from: https://archive.ics.uci.edu/ml/datasets/wine+quality")
        print("   Save as 'winequalityred.csv' and 'winequalitywhite.csv' in data/ folder")
        
    except ImportError as e:
        print(f"\n IMPORT ERROR: {e}")
        print("\n Please ensure all required modules are in src/ folder:")
        print("   • preprocess.py")
        print("   • logistic_regression.py") 
        print("   • svm.py")
        print("   • utils.py")
        
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
