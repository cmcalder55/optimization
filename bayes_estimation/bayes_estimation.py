#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Created On:    2024/09/21
Last Revision: 0000/00/00

<DESCRIPTION>
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# metadata
__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"


class BayesRisk():
    def __init__(self, x, N, P):
        self.x = x
        self.N = N
        self.P = P
        self.likelihood = self.norm_distribution()
        self.likelihood_ratio = self.likelihood[0]/self.likelihood[1]
        self.prior = self.prior_probability()
        self.evidence = self.evidence_distribution()
        self.posterior = self.posterior_probability()

    def plot_data(self, data_vector,  title="", n_classes=2, xlim=(-10, 10), ylim=(0, 0.5), legends=[]):

        _, ax = plt.subplots()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability, P')
        
        if len(data_vector) > n_classes:
            data_vector = [data_vector]

        for idx, d in enumerate(data_vector):
            label = legends[idx] if legends else f"Class {idx + 1}"
            ax.plot(self.x, d, label=label)

        if legends:
            ax.legend()

        plt.show()

    def norm_distribution(self):
        '''Given mean and standard deviation, outputs the normal distribution for x,
        which can be a vector or a specific value.'''

        # return partial dist. vector from N(mu, std. dev.)
        return np.array([stats.norm(mu, np.sqrt(sigma)).pdf(self.x) for mu, sigma in self.N])

    def prior_probability(self):
        return np.array([likelihood*prior for likelihood, prior in zip(self.likelihood, self.P)])

    def evidence_distribution(self):
        return sum(self.prior, 0)

    def posterior_probability(self):
        return np.array([p/ev for p, ev in zip(self.prior, self.evidence)])
        # return self.prior/self.evidence

    def zero_one_loss(self, risk_matrix):
        theta_zero_one = self.P[1]/self.P[0]
        return (risk_matrix[0][1] - risk_matrix[1][1])/(risk_matrix[1][0] - risk_matrix[0][0])*theta_zero_one

    def bayes_risk_df(self, risk_matrix):
        '''Calculate the likelihood ratio threshold for zero-one loss function, i.e. maximum posterior classification.'''

        data = {}
        for idx, risk in enumerate(risk_matrix):
            R = np.sum([a*b for a,b in zip(self.posterior, risk)], 0)
            data[f"Class {idx+1}"] = R

        return pd.DataFrame(data, index=self.x)

    def plot_bayes_risk_threshold(self, risk_matrix, legends=[], title="Bayes Conditional Risk, R(αi|x)"):
        """sketch the Bayes risk (conditional risk R(αi|x) associated with the action αi
        according to observation x) as a function of x"""


        risk_threshold = self.zero_one_loss(risk_matrix)
        df = self.bayes_risk_df(risk_matrix)
        df.plot()

        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('R(αi|x)')

        if not legends:
            legends = df.columns

        plt.fill_between(self.x, df[legends[0]], where=(self.x < risk_threshold))
        plt.fill_between(self.x, df[legends[1]], where=(self.x > risk_threshold))

        plt.legend(legends)
        plt.show()


def plot_decision_boundary(mu, boundary_func, x=np.linspace(-6, 8, 400), y=np.linspace(-6, 8, 400)):

    # Create a grid of points
    X, Y = np.meshgrid(x, y)

    # Compute the function values
    Z = np.array([boundary_func(xi, yi) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    ### Plotting
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=[0], colors='red', linestyles='dashed')
    plt.contourf(X, Y, Z, levels=[0, Z.max()], colors='blue', alpha=0.5)
    plt.contourf(X, Y, Z, levels=[Z.min(), 0], colors='orange', alpha=0.5)
    plt.scatter(*mu[0], color='blue', label='Class 1 Mean')
    plt.scatter(*mu[1], color='orange', label='Class 2 Mean')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


def decision_boundary(mu, sigma, P, plot=False):
    
    sig_inv = [np.linalg.inv(s) for s in sigma]

    W = [-0.5 * s_inv for s_inv in sig_inv]
    w = [np.dot(s_inv, m) for s_inv, m in zip(sig_inv, mu)]
    w0 = [
        -0.5 * np.dot(np.dot(m.T, s_inv), m) - 0.5 * np.log(np.linalg.det(s)) + np.log(p)
        for m, s_inv, s, p in zip(mu, sig_inv, sigma, P)
    ]

    # Define the decision function g(x) = g1(x) - g2(x); 
    # when g1 = g2 --> g1-g2 = 0
    decision_function = lambda x1, x2: (
        np.dot(np.dot([x1, x2], W[0]), [x1, x2]) + 
        np.dot(w[0], [x1, x2]) + w0[0] -
        (np.dot(np.dot([x1, x2], W[1]), [x1, x2]) +
        np.dot(w[1], [x1, x2]) + w0[1])
    )
    if plot:
        plot_decision_boundary(mu, decision_function)

    return decision_function


def bhattacharyya_error_bound(mu, sigma, P):

    avg_sigma = np.sum(sigma, axis=0)/2
    delta_mu = mu[1] - mu[0]
    
    # Calculate Bhattacharyya coefficients
    t1 = 1/8 * np.dot(np.dot(delta_mu.T, np.linalg.inv(avg_sigma)), delta_mu)
    t2 = 0.5 * np.log(np.linalg.det(avg_sigma) /
                      np.sqrt(np.linalg.det(sigma[0]) * np.linalg.det(sigma[1])))
    
    boundary = np.sqrt(P[0] * P[1]) * np.exp(-(t1 + t2))
    print(f'\nBhattacharyya error bound: P(error) <= {boundary:.4f}')
    
    return boundary
