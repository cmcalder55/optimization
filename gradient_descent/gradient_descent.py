#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Gradient Descent
Includes: 
    Batch GD
    Stochastic GD
    Normal Equation
    Newton's Steepest Descent
'''

import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"

FloatArray = NDArray[np.float64]
ErrorDict = Dict[int, float]


class PlotData():
    """A class for handling plotting of regression data and predictions."""
    def __init__(self, figsize: Tuple[int, int] = (8, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.scatter_lim_x = (4, 25)
        self.scatter_lim_y = (-5, 25)
    def plot_line(self, X: FloatArray, y: FloatArray, p: FloatArray, 
                 use_ax: Tuple[int, int, int], title: str = "Training Data and Predicted Values",
                 xlabel: str = "Training Input", ylabel: str = "Target/Predicted") -> plt.Figure:
        ax = self.fig.add_subplot(*use_ax)
        ax.scatter(X, y)
        ax.plot(X, p, 'r')
        ax.grid(True)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return self.fig
    def plot_scatter(self, X: FloatArray, y: FloatArray, 
                    use_ax: Tuple[int, int, int], title: str = "Raw Data") -> plt.Figure:
        ax = self.fig.add_subplot(*use_ax)
        ax.scatter(X, y)
        ax.set(xlim=self.scatter_lim_x, ylim=self.scatter_lim_y)
        ax.grid(True)
        ax.set_title(title)
        return self.fig

class NormalRegCoeffs():
    """Normal equation regression for linear models."""
    def __init__(self, df):
        self.m = df.shape[0]
        self.X = df[0].to_numpy().reshape(self.m,1)
        self.y = df[1].to_numpy().reshape(self.m,1)
        self.plot = PlotData()
    def predict(self, X, w):
        return X.dot(w)
    def get_coeff_norm(self, X, y):
        a,b = X.T.dot(X), X.T.dot(y)
        return np.linalg.inv(a).dot(b)
    def main(self):
        bias_X = np.append(self.X, np.ones((self.m,1)), axis=1)
        w = self.get_coeff_norm(bias_X, self.y)
        p = self.predict(bias_X, w)
        print("Linear model: y = w0 + w1*x\n" \
             f"Predicted regression coefficients: {w[0][0]:.4}, {w[1][0]:.4}\n")
        fig1 = self.plot.plot_scatter(self.X, self.y, (2,1,1))
        fig2 = self.plot.plot_line(self.X, self.y, p, (2,1,2))
        return [fig1, fig2]

class LinGD():
    """Simple linear regression using gradient descent."""
    def __init__(self, learning_rate: float = 1e-10, epochs: int = 10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def yhat(self, X, w):
        return np.dot(w.T, X)
    def loss(self, yhat, y):
        return 1/y.shape[1] * np.sum(np.power(yhat - y, 2))
    def gradient_descent(self, w, X, y, yhat):
        dldw = 2/y.shape[1] * np.dot(X, (yhat-y).T)
        return w - self.learning_rate*dldw
    def main(self, X, y):
        x1 = np.ones((1, X.shape[1]))
        X = np.append(X, x1, axis=0)
        w = np.zeros((X.shape[0], 1))
        for epoch in range(self.epochs+1):
            yhat = self.yhat(X, w)
            loss = self.loss(yhat, y)
            if epoch % 2000 == 0:
                print(f'cost at epoch {epoch} is {loss:.8}')
            w = self.gradient_descent(w, X, y, yhat)
        return w

class GradientDescent():
    """Batch and stochastic gradient descent for linear regression."""
    def __init__(self, learning_rate: float = 1e-3, epochs: int = 1000, tol: float = 1e-4):
        self.epochs = epochs
        self.learn = learning_rate
        self.tol = tol
        
    def prediction(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.dot(w.T, X)
    def MSE(self, error: np.ndarray, m: int) -> float:
        return np.mean(error**2)
    def descent(self, X: np.ndarray, error: np.ndarray, N: int) -> np.ndarray:
        dfdw = 2/N * X @ error.T
        return -self.learn * dfdw
    def _compute_errors(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, 
                       y_test: np.ndarray, w: np.ndarray, m: int, m_test: int) -> Tuple[np.ndarray, np.ndarray]:
        p_test = self.prediction(X_test, w)
        t_error = p_test - y_test
        p = self.prediction(X, w)
        error = p - y
        return error, t_error
    def get_coeff_BGD(self, X: np.ndarray, y: np.ndarray, m: int, m_test: int, 
                      start: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[dict, dict]:
        X = np.vstack([X, np.ones((1, m))])
        X_test = np.vstack([X_test, np.ones((1, m_test))])
        train_error = {}
        test_error = {}
        w = start.copy()
        for i in range(self.epochs + 1):
            error, t_error = self._compute_errors(X, y, X_test, y_test, w, m, m_test)
            train_error[i] = self.MSE(error, m)
            test_error[i] = self.MSE(t_error, m_test)
            delta = self.descent(X, error, m)
            if np.all(np.abs(delta) <= self.tol):
                break
            w += delta
        return test_error, train_error
    def get_coeff_SGD(self, X: np.ndarray, y: np.ndarray, m: int, m_test: int, 
                      start: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[dict, dict]:
        X = np.vstack([X, np.ones((1, m))])
        X_test = np.vstack([X_test, np.ones((1, m_test))])
        train_error = {}
        test_error = {}
        w = start.copy()
        for i in range(self.epochs + 1):
            error, t_error = self._compute_errors(X, y, X_test, y_test, w, m, m_test)
            train_error[i] = self.MSE(error, m)
            test_error[i] = self.MSE(t_error, m_test)
            idx = np.random.randint(m)
            X_sample = X[:, idx:idx+1]
            error_sample = error[:, idx:idx+1]
            delta = self.descent(X_sample, error_sample, 1)
            if np.all(np.abs(delta) <= self.tol):
                break
            w += delta
        return test_error, train_error
    def _plot(self, rate: List[float], train: List[dict], test: List[dict], title: str, n: int) -> plt.Figure:
        rows = n // 2
        fig, axs = plt.subplots(rows, 2, figsize=(10, rows * 3))
        axs = axs.flatten() if rows > 1 else [axs]
        make_plottable = lambda d: (list(d.keys()), list(d.values()))
        for i, ax in enumerate(axs):
            ax.plot(*make_plottable(train[i]), label='train')
            ax.plot(*make_plottable(test[i]), label='test')
            ax.set_title(f"Learning Rate = {rate[i]}")
            ax.legend()
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        return fig

class NewtonsDescent():
    """Newton's method for unconstrained minimization using steepest descent."""
    def __init__(self, k=10, epsilon=1e-5, t_stop=1e-6, alpha=1e-8, print_out=True, plot_out=True):
        self.printout = print_out
        self.plotout = plot_out
        self.k = k
        self.epsilon = epsilon
        self.t_stop = t_stop
        self.alpha = alpha
    def steepest_descent(self, minimizer):
        dim = len(inspect.getfullargspec(minimizer).args)
        start = np.ones(dim)
        costfxn = {}
        x = start
        fx_current = minimizer(*x)
        for i in range(self.k):
            gk = self.gradient(minimizer, x)
            H = self.hessian(minimizer, x, dim)
            alpha = self.get_alpha(gk, H)
            delta = -alpha * gk
            if np.linalg.norm(delta) < self.t_stop:
                break
            x += delta
            fx_current = minimizer(*x)
            costfxn[i] = fx_current
        if self.printout:
            self._print_output(x, minimizer, H)
        if self.plotout:
            self._plot_iter(costfxn)
        return costfxn
    def gradient(self, minimizer, x):
        x_epsilons = x[:, np.newaxis] + self.epsilon * np.eye(len(x)).T
        grad = (minimizer(*x_epsilons) - minimizer(*x)) / self.epsilon
        return grad
    def hessian(self, minimizer, x, n):
        H = np.zeros((n, n))
        fx = minimizer(*x)
        for i in range(n):
            x_plus = np.array(x)
            x_minus = np.array(x)
            x_plus[i] += self.epsilon
            x_minus[i] -= self.epsilon
            f_plus = minimizer(*x_plus)
            f_minus = minimizer(*x_minus)
            H[i, i] = (f_plus - 2 * fx + f_minus) / self.epsilon**2
        for i in range(n):
            for j in range(i + 1, n):
                x_ij = np.array(x)
                x_ij[i] += self.epsilon
                x_ij[j] += self.epsilon
                f_pp = minimizer(*x_ij)
                x_ij[j] -= 2 * self.epsilon
                f_pm = minimizer(*x_ij)
                x_ij[i] -= 2 * self.epsilon
                f_mm = minimizer(*x_ij)
                x_ij[j] += 2 * self.epsilon
                f_mp = minimizer(*x_ij)
                H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4 * self.epsilon**2)
        return H
    def get_alpha(self, gk, H):
        gk_H_gk = np.dot(gk, H @ gk)
        alpha = np.dot(gk, gk) / (gk_H_gk + self.alpha)
        return alpha
    def _plot_iter(self, costfxn):
        iterations = list(costfxn.keys())
        objective_values = list(costfxn.values())
        plt.figure()  
        plt.plot(iterations, objective_values, marker='o', linestyle='-')
        plt.title('Minimization with Steepest Gradient Descent')
        plt.xlabel('Iterations')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.show()       
    def _print_output(self, x_final, minimizer, H_final):
        print('\nMinimizing x vector: \n' + str(x_final))
        print('\nObjective function value: ' + str(minimizer(*x_final)))
        print('\nFinal Hessian matrix: \n' + str(H_final))
