"""Linear regression implemented with NumPy only (great for understanding / viva questions).
Two solvers: Normal Equation (closed form) and Batch Gradient Descent."""
import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="gd", lr=0.05, n_iter=2000, l2=0.0):
        self.method, self.lr, self.n_iter, self.l2 = method, lr, n_iter, l2
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.c_[np.ones(len(X)), X]                      # add bias column
        n = len(Xb)
        if self.method == "normal":                         # w = (X'X + λI)^-1 X'y
            reg = self.l2 * np.eye(Xb.shape[1]); reg[0, 0] = 0
            self.w = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ y
        else:                                               # gradient descent on MSE
            self.w = np.zeros(Xb.shape[1])
            for _ in range(self.n_iter):
                err = Xb @ self.w - y
                grad = (2 / n) * Xb.T @ err
                grad[1:] += 2 * self.l2 * self.w[1:] / n
                self.w -= self.lr * grad
                self.loss_history.append((err ** 2).mean())
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), np.asarray(X, float)] @ self.w

    @property
    def intercept_(self): return self.w[0]
    @property
    def coef_(self): return self.w[1:]
