import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(
        self, 
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        verbose: bool = False
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        
        # Initialize model parameters
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.n_features = None
        self.is_fitted = False
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        
        # Clip z to prevent overflow in exp(-z)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        # small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # cross-entropy loss
        n_samples = len(y_true)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization term
        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2)
            
        return cost
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
       
        n_samples = X.shape[0]
        
        # base gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_true))
        db = (1 / n_samples) * np.sum(y_pred - y_true)
        
        # Add regularization to weight gradients
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
            
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1 values")
            
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # parameters initialization and reproducibility 
        np.random.seed(42)  
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        self.cost_history = []
        
        if self.verbose:
            print(f"Training Logistic Regression...")
            print(f"Samples: {n_samples}, Features: {n_features}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Regularization: {self.regularization}")
        
        # GDO
        for iteration in range(self.max_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            dw, db = self._compute_gradients(X, y, y_pred)
            
            old_weights = self.weights.copy()
            
            # Update 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            weight_change = np.linalg.norm(self.weights - old_weights)
            if weight_change < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Cost: {cost:.6f}")
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"Training completed. Final cost: {self.cost_history[-1]:.6f}")
            
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        #Predicted probabilities for the positive class
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} features")
            
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
         #Predict binary class labels for samples in X.
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def get_feature_importance(self) -> np.ndarray:
        
        #Get feature importance based on absolute weight values.
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return np.abs(self.weights)
    
    def get_params(self) -> dict:
        
        return {
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'LogisticRegression':
        
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


def cross_validate_lr(
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5,
    **lr_params
) -> dict:
    
    # Cross-validation scores
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model = LogisticRegression(**lr_params)
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
    
    print("Testing Logistic Regression implementation...")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_redundant=0, 
        n_informative=10, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    lr = LogisticRegression(
        learning_rate=0.1,
        max_iterations=1000,
        regularization='l2',
        lambda_reg=0.01,
        verbose=True
    )
    
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = lr.predict(X_test_scaled)
    probabilities = lr.predict_proba(X_test_scaled)
    
    # Compute accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Feature importance: {lr.get_feature_importance()[:5]}")
    
    print("Logistic Regression test completed successfully!")