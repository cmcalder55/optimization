# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:51:56 2022

@author: camer
"""
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

class matrixFactorization:
    def __init__(self, R, K=2, steps=5000, alpha=0.0001, beta=0.001):
        self.R = np.array(R)
        self.m, self.n = self.R.shape
        self.k = K
        self.steps = steps
        self.a = alpha
        self.b = beta
        self.P = np.random.rand(self.m, self.k)
        self.Q = np.random.rand(self.k, self.n)
        self.mse = []
        
    def factorize(self):
        '''Given a test matrix R with dimensions MxN, and the initial decomposed
        prediction matrix P dim. MxK and Q dim. KxN, compares P.Q to R and adjusts 
        using the factors alpha and beta over the given number of steps to converge
        error between the non-zero elements of R and P.Q to zero. Zero elements 
        are replaced with predictions.'''
        

        for _ in range(self.steps):
            # iterate over the rows of R s number of times
            for i, j in product(range(self.m), range(self.n)):   
                # if the current element is non-zero, see similarity to the training matrix
                if self.R[i, j] > 0:
                    # get error between R and P.Q
                    e = self.R[i, j] - self.P[i,:]@self.Q[:,j]
                    # adjust P and Q and predict unrated items
                    for k in range(self.k):
                        self.P[i, k] += self.a*(2*e*self.Q[k, j] - self.b*self.P[i, k])
                        self.Q[k, j] += self.a*(2*e*self.P[i, k] - self.b*self.Q[k, j])
                        
            step_mse = np.mean((self.R - np.dot(self.P, self.Q))**2)
            self.mse.append(step_mse)

def plot_mse(matrix_error):
    
    plt.figure(figsize=(10, 5))
    for i, errors in enumerate(matrix_error):
        plt.plot(errors, label=f'R{i+1} MSE')
    plt.xlabel('Step')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Matrix Factorization Error Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    mf = matrixFactorization

    R = [[
            [5,3,0,1], 
            [4,0,0,1], 
            [1,1,0,5], 
            [1,0,0,4], 
            [0,1,5,4]
        ],
        [
            [4,3,0,1,2], 
            [5,0,0,1,0], 
            [1,2,1,5,4], 
            [1,0,0,4,0], 
            [0,1,5,4,0],
            [5,5,0,0,1]
        ]
    ]
    
    matrix_error = []
    for R_mat in R:
        factorizer = mf(R=R_mat)
        factorizer.factorize()
        matrix_error.append(factorizer.mse)

    plot_mse(matrix_error)