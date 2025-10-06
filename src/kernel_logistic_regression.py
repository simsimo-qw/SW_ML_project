# src/kernel_logistic_regression.py

import numpy as np
from logistic_regression import LogisticRegression
from svm import RBFKernel, PolynomialKernel

class KernelLogisticRegression(LogisticRegression):
    

    def __init__(self, kernel='rbf', gamma='scale', degree=3, coef0=1.0, **kwargs):
        
        
        super().__init__(**kwargs)

        self.kernel_name = kernel
        if kernel == 'rbf':
            self.kernel_func = RBFKernel(gamma=gamma)
        elif kernel == 'poly':
            self.kernel_func = PolynomialKernel(degree=degree, gamma=gamma, coef0=coef0)
        else:
            raise ValueError("Unsupported kernel type. Use 'rbf' or 'poly'.")

        self.X_train_ = None  

    def fit(self, X, y):
        
        self.X_train_ = np.asarray(X)
       
        K = self.kernel_func.compute(self.X_train_, self.X_train_)
        
        return super().fit(K, y)

    def _check_fitted(self):
        if self.X_train_ is None:
            raise ValueError("Model not fitted. Call fit(X, y) before predict/predict_proba.")

    def predict_proba(self, X):
        self._check_fitted()
        X = np.asarray(X)
        # K(test, train)
        K = self.kernel_func.compute(X, self.X_train_)
        # z = K @ w + b  (LogisticRegression learned weights on Gram space)
        z = K.dot(self.weights) + self.bias
        # use base sigmoid
        return self._sigmoid(z)

    def predict(self, X):
        # threshold the probabilities
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
