import numpy as np
from typing import Optional, Callable, Tuple
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Kernel(ABC):

    @abstractmethod
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        
        pass


class LinearKernel(Kernel):
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.dot(X1, X2.T)


class RBFKernel(Kernel):
    
    def __init__(self, gamma='scale'):
        self.gamma = gamma
        self._computed_gamma = None
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Compute gamma 
        if self.gamma == 'scale' and self._computed_gamma is None:
            self._computed_gamma = 1 / (X1.shape[1] * X1.var())
        elif isinstance(self.gamma, (int, float)):
            self._computed_gamma = self.gamma
        
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        
        return np.exp(-self._computed_gamma * sq_dists)


class PolynomialKernel(Kernel):
    
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
        
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = 0.0
        self.kernel_func = None
        self.n_features = None
        self.is_fitted = False
        
        self._X_train = None
        self._y_train = None
        
    def _setup_kernel(self, X: np.ndarray) -> None:

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
       
        if alpha_i < self.C and self._y_train[i] * error_i < -self.tolerance:
            return True
        if alpha_i > 0 and self._y_train[i] * error_i > self.tolerance:
            return True
        return False
    
    def _compute_error(self, i: int, K: np.ndarray) -> float:
        
        prediction = np.sum(self.alpha * self._y_train * K[i]) + self.bias
        return prediction - self._y_train[i]
    
    def _select_second_alpha(self, i: int, error_i: float) -> int:
        
        candidates = [j for j in range(len(self._y_train)) if j != i]
        return np.random.choice(candidates)
    
    def _compute_bounds(self, i: int, j: int) -> Tuple[float, float]:
        
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
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # enforce binary {0,1} then map to {-1,+1}
        labels = np.unique(y)
        if len(labels) != 2:
           raise ValueError("SVM requires exactly 2 classes")
        if not np.all(np.isin(labels, [0, 1])):
 # map smallest to 0, largest to 1 to be safe
         y01 = (y == labels.max()).astype(int)
        else:
             y01 = y.astype(int)
        y_binary = np.where(y01 == 1, 1, -1)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        self._X_train = X.copy()
        self._y_train = y_binary.copy()
        
        self._setup_kernel(X)
        
       
        np.random.seed(42)  
        self.alpha = np.zeros(n_samples)
        self.bias = 0.0
        
        if self.verbose:
            print(f"Training SVM with {self.kernel} kernel...")
            print(f"Samples: {n_samples}, Features: {n_features}")
            print(f"C: {self.C}")
        
        
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
        
        alpha_i = self.alpha[i]
        y_i = self._y_train[i]
        error_i = self._compute_error(i, K)
        
       
        if not self._check_kkt_conditions(i, error_i, alpha_i):
            return 0
        
        
        j = self._select_second_alpha(i, error_i)
        
        return self._take_step(i, j, K)
    
    def _take_step(self, i: int, j: int, K: np.ndarray) -> int:
        
        if i == j:
            return 0
        
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        y_i = self._y_train[i]
        y_j = self._y_train[j]
        
        
        error_i = self._compute_error(i, K)
        error_j = self._compute_error(j, K)
        
       
        L, H = self._compute_bounds(i, j)
        
        if L == H:
            return 0
        
        
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        
        if eta >= 0:
            return 0  
        
        self.alpha[j] = alpha_j_old - y_j * (error_i - error_j) / eta
        
        
        self.alpha[j] = np.clip(self.alpha[j], L, H)
        
        
        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            return 0
        
        
        self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alpha[j])
        
        
        self._update_bias(i, j, alpha_i_old, alpha_j_old, error_i, error_j, K)
        
        return 1
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} features")
        
        # Compute kernel between X and support vectors
        K = self.kernel_func.compute(X, self.support_vectors)
        
        
        decision = np.dot(K, self.alpha * self.support_labels) + self.bias
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for samples in X.
        Predicted binary class labels (0 or 1)
        """
        decision = self.decision_function(X)
        
        return (decision > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        
        decision = self.decision_function(X)
        # Simple sigmoid transformation 
        return 1 / (1 + np.exp(-decision))
    
    def get_params(self) -> dict:
        
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
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

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
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    print("Testing SVM implementation...")
    
    
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, 
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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