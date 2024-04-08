# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:51:56 2022

@author: camer
"""
from itertools import product
from numpy import random, array, stack
import matplotlib.pyplot as plt

class matrixFactorization:
    def __init__(self, R, K=2, steps=5000, alpha=0.0002, beta=0.02):
        self.r = array(R)
        self.m = len(self.r)
        self.n = len(self.r[0])
        self.k = K
        self.steps = steps
        self.a = alpha
        self.b = beta
        self.result = self.matrixFactorization()
        
    def matrixFactorization(self):
        '''Given a test matrix R with dimensions MxN, and the initial decomposed
        prediction matrix P dim. MxK and Q dim. KxN, compares P.Q to R and adjusts 
        using the factors alpha and beta over the given number of steps to converge
        error between the non-zero elements of R and P.Q to zero. Zero elements 
        are replaced with predictions.'''
        
        P = stack([random.ranf(self.k) for _ in range(self.m)], axis=0)
        Q = stack([random.ranf(self.n) for _ in range(self.k)], axis=0)
        self.errors = []
        for _ in range(self.steps):
            # iterate over the rows of R s number of times
            for m,n in product(range(self.m),range(self.n)):   
                    # if the current element is non-zero, see similarity to the training matrix
                if self.r[m][n] != 0:
                    # get error between R and P.Q
                    e = self.r[m][n] - P[m,:]@Q[:,n]
                    mse = (e**2).mean()
                    self.errors.append(mse)
                    # adjust P and Q and predict unrated items
                    for k in range(self.k):
                        P[m][k] = P[m][k] + self.a*(2*e*Q[k][n] - self.b*P[m][k])
                        Q[k][n] = Q[k][n] + self.a*(2*e*P[m][k] - self.b*Q[k][n])
        result = {
            'nP': P,
            'nQ': Q,
            'nR': P@Q
        }
        return result
    
def visualize_errors(errors):
    """
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='Mean Squared Error over Steps')
    plt.xlabel('Step')
    plt.ylabel('Mean Squared Error')
    plt.title('Matrix Factorization Error Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():

    results = []
    for r_matrix in R:
        mf = matrixFactorization(R=r_matrix)
        res = {
            'R': r_matrix,
            'result': mf.result
        }
        results.append(res)

        if visualize:
            visualize_errors(mf.errors)

        if print_res:
            print(results)
            
    return results

if __name__ == "__main__":
    
    visualize=True
    print_res=False

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
    
    results = main()


    
    