#!/usr/bin/env python3
"""
Support Vector Machine Implementation from Scratch
==================================================

This module implements Support Vector Machine using a simplified Sequential 
Minimal Optimization (SMO) algorithm with support for multiple kernels.

Author: [SW]
Course: Machine Learning
"""

import numpy as np
from typing import Optional, Callable, Tuple
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Abstract base class for kernel functions."""
    
    @abstractmethod
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        pass


class LinearKernel(Kernel):
    """Linear kernel: K(x, y) = x^T * y"""
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.dot(X1, X2.T)


class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    
    Parameters
    ----------
    gamma : float or str, default='scale'
        Kernel coefficient. If 'scale', uses 1 / (n_features * X.var())
    """
    
    def __init__(self, gamma='scale'):
        self.gamma = gamma
        self._computed_gamma = None
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Compute gamma if not set
        if self.gamma == 'scale' and self._computed_gamma is None:
            self._computed_gamma = 1 / (X1.shape[1] * X1.var())
        elif isinstance(self.gamma, (int, float)):
            self._computed_gamma = self.gamma
        
        # Compute squared Euclidean distances
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        
        return np.exp(-self._computed_gamma * sq_dists)


class PolynomialKernel(Kernel):
    """
    Polynomial kernel: K(x, y) = (gamma * x^T * y + coef0)^degree
    
    Parameters
    ----------
    degree : int, default=3
        Polynomial degree
    gamma : float or str, default='scale'
        Kernel coefficient
    coef0 : float, default=1.0
        Independent term in kernel function
    """
    
    def __init__(self, degree=3, gamma='scale', coef0=1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self._computed_gamma = None
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Compute gamma if not set
        if self.gamma == 'scale' and self._computed_gamma is None:
            self._computed_gamma = 1 / X1.shape[1]
        elif isinstance(self.gamma, (int, float)):
            self._computed_gamma = self.gamma
        
        return (self._computed_gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree


class SVM:
    """
    Support Vector Machine classifier implemented from scratch.
    
    This implementation uses a simplified Sequential Minimal Optimization (SMO)
    algorithm and supports multiple kernel functions.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter
    kernel : str or Kernel, default='linear'
        Kernel type ('linear', 'rbf', 'poly') or Kernel instance
    gamma : float or str, default='scale'
        Kernel coefficient for 'rbf' and 'poly'
    degree : int, default=3
        Degree for polynomial kernel
    coef0 : float, default=1.0
        Independent term for polynomial kernel
    max_iterations : int, default=1000
        Maximum iterations for optimization
    tolerance : float, default=1e-3
        Tolerance for convergence
    verbose : bool, default=False
        Whether to print training progress
        
    Attributes
    ----------
    alpha : ndarray of shape (n_support_vectors,)
        Lagrange multipliers for support vectors
    support_vectors : ndarray of shape (n_support_vectors, n_features)
        Support vectors
    support_labels : ndarray of shape (n_support_vectors,)
        Labels of support vectors
    bias : float
        Bias term
    kernel_func : Kernel
        Kernel function object
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'linear',
        gamma: str = 'scale',
        degree: int = 3,
        coef0: float = 1.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-3,
        verbose: bool = False
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Initialize model parameters
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = 0.0
        self.kernel_func = None
        self.n_features = None
        self.is_fitted = False
        
        # Training data (needed for kernel computation)
        self._X_train = None
        self._y_train = None
        
    def _setup_kernel(self, X: np.ndarray) -> None:
        """Setup the kernel function based on the kernel parameter."""
        if isinstance(self.kernel, Kernel):
            self.kernel_func = self.kernel
        elif self.kernel == 'linear':
            self.kernel_func = LinearKernel()
        elif self.kernel == 'rbf':
            self.kernel_func = RBFKernel(gamma=self.gamma)
        elif self.kernel == 'poly':
            self.kernel_func = PolynomialKernel(
                degree=self.degree, 
                gamma=self.gamma, 
                coef0=self.coef0
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _check_kkt_conditions(
        self, 
        i: int, 
        error_i: float, 
        alpha_i: float
    ) -> bool:
        """Check if KKT conditions are violated for sample i."""
        if alpha_i < self.C and self._y_train[i] * error_i < -self.tolerance:
            return True
        if alpha_i > 0 and self._y_train[i] * error_i > self.tolerance:
            return True
        return False
    
    def _compute_error(self, i: int, K: np.ndarray) -> float:
        """Compute prediction error for sample i."""
        prediction = np.sum(self.alpha * self._y_train * K[i]) + self.bias
        return prediction - self._y_train[i]
    
    def _select_second_alpha(self, i: int, error_i: float) -> int:
        """Select second alpha using heuristic (simplified)."""
        # Simple random selection (can be improved with better heuristics)
        candidates = [j for j in range(len(self._y_train)) if j != i]
        return np.random.choice(candidates)
    
    def _compute_bounds(self, i: int, j: int) -> Tuple[float, float]:
        """Compute bounds L and H for alpha_j."""
        if self._y_train[i] != self._y_train[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H
    
    def _update_bias(
        self, 
        i: int, 
        j: int, 
        alpha_i_old: float, 
        alpha_j_old: float,
        error_i: float, 
        error_j: float, 
        K: np.ndarray
    ) -> None:
        """Update bias term."""
        b1 = (
            self.bias - error_i - 
            self._y_train[i] * (self.alpha[i] - alpha_i_old) * K[i, i] -
            self._y_train[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
        )
        
        b2 = (
            self.bias - error_j - 
            self._y_train[i] * (self.alpha[i] - alpha_i_old) * K[i, j] -
            self._y_train[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
        )
        
        if 0 < self.alpha[i] < self.C:
            self.bias = b1
        elif 0 < self.alpha[j] < self.C:
            self.bias = b2
        else:
            self.bias = (b1 + b2) / 2
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train the SVM model using simplified SMO algorithm.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values (0 or 1, will be converted to -1 or 1)
            
        Returns
        -------
        self
            Returns the instance itself
        """
        # Validate input
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Convert labels to -1, 1 format
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM requires exactly 2 classes")
        
        # Map labels to -1, 1
        y_binary = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Store training data
        self._X_train = X.copy()
        self._y_train = y_binary.copy()
        
        # Setup kernel
        self._setup_kernel(X)
        
        # Initialize parameters
        np.random.seed(42)  # For reproducibility
        self.alpha = np.zeros(n_samples)
        self.bias = 0.0
        
        if self.verbose:
            print(f"Training SVM with {self.kernel} kernel...")
            print(f"Samples: {n_samples}, Features: {n_features}")
            print(f"C: {self.C}")
        
        # Compute kernel matrix
        K = self.kernel_func.compute(X, X)
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        
        for iteration in range(self.max_iterations):
            num_changed = 0
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    num_changed += self._examine_example(i, K)
            else:
                # Examine non-bound samples
                for i in range(n_samples):
                    if 0 < self.alpha[i] < self.C:
                        num_changed += self._examine_example(i, K)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Check convergence
            if examine_all and num_changed == 0:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        # Extract support vectors
        sv_indices = self.alpha > self.tolerance
        self.support_vectors = X[sv_indices].copy()
        self.support_labels = y_binary[sv_indices].copy()
        self.alpha = self.alpha[sv_indices].copy()
        
        if self.verbose:
            print(f"Number of support vectors: {len(self.support_vectors)}")
            print(f"Support vector ratio: {len(self.support_vectors)/n_samples:.2%}")
        
        self.is_fitted = True
        return self
    
    def _examine_example(self, i: int, K: np.ndarray) -> int:
        """Examine example i and try to optimize it."""
        alpha_i = self.alpha[i]
        y_i = self._y_train[i]
        error_i = self._compute_error(i, K)
        
        # Check KKT conditions
        if not self._check_kkt_conditions(i, error_i, alpha_i):
            return 0
        
        # Select second alpha
        j = self._select_second_alpha(i, error_i)
        
        return self._take_step(i, j, K)
    
    def _take_step(self, i: int, j: int, K: np.ndarray) -> int:
        """Take optimization step for alpha_i and alpha_j."""
        if i == j:
            return 0
        
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        y_i = self._y_train[i]
        y_j = self._y_train[j]
        
        # Compute errors
        error_i = self._compute_error(i, K)
        error_j = self._compute_error(j, K)
        
        # Compute bounds
        L, H = self._compute_bounds(i, j)
        
        if L == H:
            return 0
        
        # Compute eta
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        
        if eta >= 0:
            return 0  # Skip this pair
        
        # Update alpha_j
        self.alpha[j] = alpha_j_old - y_j * (error_i - error_j) / eta
        
        # Clip alpha_j
        self.alpha[j] = np.clip(self.alpha[j], L, H)
        
        # Check if change is significant
        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            return 0
        
        # Update alpha_i
        self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alpha[j])
        
        # Update bias
        self._update_bias(i, j, alpha_i_old, alpha_j_old, error_i, error_j, K)
        
        return 1
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} features")
        
        # Compute kernel between X and support vectors
        K = self.kernel_func.compute(X, self.support_vectors)
        
        # Compute decision function
        # K has shape (n_samples, n_support_vectors)
        # alpha has shape (n_support_vectors,)
        # support_labels has shape (n_support_vectors,)
        decision = np.dot(K, self.alpha * self.support_labels) + self.bias
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1)
        """
        decision = self.decision_function(X)
        # Convert from -1/1 to 0/1
        return (decision > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Platt scaling approximation.
        
        Note: This is a simplified probability estimation.
        For proper probability calibration, use Platt scaling or isotonic regression.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted probabilities for the positive class
        """
        decision = self.decision_function(X)
        # Simple sigmoid transformation (not proper Platt scaling)
        return 1 / (1 + np.exp(-decision))
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'SVM':
        """
        Set model parameters.
        
        Parameters
        ----------
        **params
            Model parameters to set
            
        Returns
        -------
        self
            Returns the instance itself
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


def cross_validate_svm(
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5,
    **svm_params
) -> dict:
    """
    Perform k-fold cross-validation for SVM.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Target values
    cv_folds : int, default=5
        Number of cross-validation folds
    **svm_params
        Parameters to pass to SVM
        
    Returns
    -------
    dict
        Cross-validation scores
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model = SVM(**svm_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        predictions = model.predict(X_val_fold)
        
        scores['accuracy'].append(accuracy_score(y_val_fold, predictions))
        scores['precision'].append(precision_score(y_val_fold, predictions, zero_division=0))
        scores['recall'].append(recall_score(y_val_fold, predictions, zero_division=0))
        scores['f1'].append(f1_score(y_val_fold, predictions, zero_division=0))
    
    return scores


if __name__ == "__main__":
    # Simple test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    print("Testing SVM implementation...")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, 
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different kernels
    kernels = ['linear', 'rbf']
    
    for kernel in kernels:
        print(f"\n--- Testing {kernel.upper()} kernel ---")
        
        # Train model
        svm = SVM(
            C=1.0,
            kernel=kernel,
            gamma='scale',
            max_iterations=1000,
            verbose=True
        )
        
        svm.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = svm.predict(X_test_scaled)
        
        # Compute accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    print("SVM test completed successfully!")