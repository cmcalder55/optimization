#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created On:   2024/09/14
Last Revision: 0000/00/00

<DESCRIPTION>
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from random import sample

__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2020-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"


class PlotData():
    def __init__(self, figsize=(8,10)):
        self.fig = plt.figure(figsize=figsize)
        self.scatter_lim_x, self.scatter_lim_y = (4,25), (-5,25)

    def plot_line(self, X, y, p, use_ax, title="Training Data and Predicted Values, Normal Eq.",
                                    xlabel="Training Input", ylabel="Target/Predicted"):
        """Plot raw data and line prediction in one plot."""
        
        # plot
        ax = self.fig.add_subplot(*use_ax)
        ax.scatter(X, y)
        ax.plot(X, p, 'r')
        # format
        ax.grid(True)
        # add title
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        
    def plot_scatter(self, X, y, use_ax, title="Raw Data, X vs. y"):
        """Plot raw data in a scatter plot."""

        ax = self.fig.add_subplot(*use_ax)
        # plot
        ax.scatter(X,y)

        # format
        ax.set(xlim=self.scatter_lim_x, ylim=self.scatter_lim_y)
        ax.grid(True)
        # add title
        ax.set_title(title)

class NormalRegCoeffs():
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
        
        self.plot.plot_scatter(self.X, self.y, (2,1,1))
        self.plot.plot_line(self.X, self.y, p, (2,1,2))

        plt.show()
        
class LinGD():
    def __init__(self):
        self.learning_rate = 1e-10
        self.epochs = 10000
        
    def yhat(self, X, w):
        return np.dot(w.T, X)
    
    def loss(self, yhat, y):
        L = 1/self.m * np.sum(np.power(yhat - y, 2))
        return L
        
    def gradient_descent(self, w, X, y, yhat):
        dldw = 2/self.m * np.dot(X, (yhat-y).T)
        w = w - self.learning_rate*dldw
        return w
        
    def main(self,X,y):
        x1 = np.ones((1, X.shape[1]))
        X = np.append(X, x1, axis=0)
        
        self.m = X.shape[1]
        self.n = X.shape[0]
        
        w = np.zeros((self.n, 1))
        
        for epoch in range(self.epochs+1):
            yhat = self.yhat(X,w)
            loss = self.loss(yhat, y)
            
            if epoch % 2000 == 0:
                print(f'cost at epoch {epoch} is {loss:.8}')
                
            w = self.gradient_descent(w, X, y, yhat)
            
        return w

class GradientDescent():
    def __init__(self, learn=1e-3, epochs=1000, tol=1e-4):
        self.epochs = epochs
        self.learn = learn
        self.tol = tol

    def prediction(self, X, w):
        return np.dot(w.T, X)

    def MSE(self, error, m):
        return 1/m*(error**2).sum()

    def descent(self, X, error, N):
        dfdw = 2/N*X.dot(error.T)
        # return negative gradient direction scaled by learning rate
        return -self.learn*dfdw

    def get_coeff_BGD(self, X, y, m, m_test, start, X_test, y_test):
        
        X = np.append(X, np.ones((1, m)), axis = 0)
        X_test = np.append(X_test, np.ones((1, m_test)), axis = 0)
        
        train_error = {}
        test_error = {}
        
        # initialize w vector as a random guess
        w = start
        
        for i in range(self.epochs + 1):
            # calculate test/train error and store in dicts
            p_test = self.prediction(X_test, w)
            t_error = p_test - y_test
            
            p = self.prediction(X, w)
            error = p - y
            
            train_error[i] = self.MSE(error, m)
            test_error[i] = self.MSE(t_error, m)
            
            # find step size and adjust w vector
            delta = self.descent(X, error, m)
            
            # if rate of change is less than or eq to tolerance, end
            if np.all(np.abs(delta) <= self.tol):
                break
                
            w = w + delta
            
        return test_error, train_error

    def get_coeff_SGD(self, X, y, m, m_test, start, X_test, y_test):
        X = np.append(X, np.ones((1, m)), axis = 0)
        X_test = np.append(X_test, np.ones((1, m_test)), axis = 0)
        
        train_error = {}
        test_error = {}
        
        # initialize w vector as a random guess
        w = start
        
        for i in range(self.epochs + 1):
            p_test = self.prediction(X_test, w)
            t_error = p_test - y_test
            p = self.prediction(X, w)
            error = p-y
            
            train_error[i] = self.MSE(error,m)
            test_error[i] = self.MSE(t_error,m)
            
            pt = sample(list(range(m)),1)[0]
            dfdw = 2*np.dot(X[0][pt],error.T[pt])
            delta = -self.learn*dfdw
            
            if np.all(np.abs(delta) <= self.tol):
                break
                
            w = w + delta
            
        return test_error, train_error
    
    def _plot(self, rate, train, test, title, n):
        
        rows = n // 2
        
        fig, axs = plt.subplots(rows, 2, figsize=(10, rows * 3))
        # Flatten the axs array if necessary
        if rows > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
                
        make_plottable = lambda d: (list(d.keys()), list(d.values()))
        
        for i, ax in enumerate(axs):
            ax.plot(*make_plottable(train[i]), label='train')
            ax.plot(*make_plottable(test[i]), label='test')
            ax.set_title(f"Learning Rate = {rate[i]}")
            ax.legend()
            
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        # Adjust the subplots to fit the main title
        plt.subplots_adjust(top=0.9)
        plt.show()
    
def make_descent(mode, X, y, start, rate):
    
    gd = GradientDescent() 
    mse = []
    test_data = []    
    train_data = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    m_test, m_train = len(X_test), len(X_train)
    
    X_train, y_train = X_train.reshape(1, m_train), y_train.reshape(1, m_train)
    X_test, y_test = X_test.reshape(1, m_test), y_test.reshape(1, m_test)
    
    if mode == "Stochastic":
        coeff = gd.get_coeff_SGD
    elif mode == "Batch":
        coeff = gd.get_coeff_BGD
        
    start = np.random.randn(2,1)    
    
    for learn in rate:
        gd.learn = learn
        ## stochastic GD
        test, train = coeff(X_train, y_train, m_train, m_test, start, X_test, y_test) 
            
        error = np.mean(list(test.values())) / m_test
        test_data.append(test)
        train_data.append(train)
        mse.append(error)
        
    formatted_start = tuple(map(lambda x: round(x[0], 2), start))
    title = f'{mode} GD: MSE vs. Iteration, start = {formatted_start}'
    gd._plot(rate, train_data, test_data, title, len(rate))
        
    # Set the size of the entire figure
    plt.figure(figsize=(6,6))  # 10 inches wide, 5 inches tall
    plt.title(f"Learning Rate vs. MSE, {mode} Gradient Descent")
    plt.plot(rate, mse)
    plt.xlabel("Learning Rate")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    
    plt.show()
               
if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), "data", "data.txt")
    df = pd.read_csv(filepath, header = None, delimiter = ",")
    
    # estimate using normal coeff regression
    norms = NormalRegCoeffs(df)
    norms.main()
    
    # regular linear regression + GD
    Xg = np.random.rand(1,500)
    yg = 3*Xg + np.random.randn(1,500)*0.1

    reg = LinGD()
    eg = reg.main(Xg, yg)
    
    # batch/stochastic GD
    X, y = df[0].to_numpy(), df[1].to_numpy()
    # intialize arrays for dataset as well as test/train set    
    rate = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    start = np.random.randn(2,1)
    
    modes = ["Stochastic", "Batch"]
    for mode in modes:
        make_descent(mode, X, y, start, rate)
