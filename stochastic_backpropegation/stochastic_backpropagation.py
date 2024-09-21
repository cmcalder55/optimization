#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created On:   2020/01/01
Last Revision: 2024/09/15

Stochastic backpropegation.
'''

import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2020-Present, Cameron Calder"
__license__=""
__version__= "1.0.0"


class StochasticBP():
    def __init__(self, data, a, b, theta, eta, nh, ni, no, epochs):
        self.data = data
        self.a = a
        self.b = b
        self.theta = theta              # convergence criterion to stop
        self.eta = eta                  # rate / step size
        self.nh = nh                    # number of hidden nodes; 10 + bias
        self.ni = ni                    # dimension of input vector = number of input nodes; 2 + bias
        self.no = no                    # number of classes / number of output nodes
        self.epochs = epochs
        self.train = None
        self.target = None
        self.features = None
        self.labels = None
        self.w_ih = None
        self.w_ho = None
        self.plots = None


    def plot_init_data(self, data):
        
        self.set_data_vectors(data)
        # Extract features and labels
        features = data.values[:, :2]  # Assuming the first two columns are features
        labels = data.values[:, 2]    # Assuming the third column contains the class labels
        self.features = features
        self.labels = labels       
        
        _, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))
        _, ax = plt.subplots(num=2, clear=True)
        self.plots = axes
        
        self.plots[0] = self.scatter_plot(0, title="Original Data Classes")
        return ax
        

    def init_weights(self):
        '''Initialize weights, including bias.'''
        # Input-to-hidden layer
        d = 1 / np.sqrt(self.ni + 1)  # Adjusting for bias
        w_ih = np.random.uniform(-d, d, size=(self.nh, self.ni + 1))
        # Hidden-to-output layer
        n = 1 / np.sqrt(self.nh + 1)  # Adjusting for bias
        w_ho = np.random.uniform(-n, n, size=(self.no, self.nh + 1))
        return w_ih, w_ho
    
    
    def update_weights(self, x, t, w_ih, w_ho):
        # Forward pass
        net_j = self.calc_net(w_ih, np.append(x, 1))  # Ensure x has bias appended
        y = self.activation(net_j)
        
        # Here, append bias to y for the output layer calculation
        y_with_bias = np.append(y, 1)  
        net_k = self.calc_net(w_ho, y_with_bias)  # w_ho should be expecting this size
        z_out = self.activation(net_k)
        
        # Backward pass: Compute deltas
        error_k = t - z_out
        delta_k = error_k * self.derivative_activation(net_k)
        error_j = np.dot(w_ho[:, :-1].T, delta_k) * self.derivative_activation(net_j)
        
        # Update weights
        w_ho += self.eta * np.outer(delta_k, y_with_bias)
        w_ih += self.eta * np.outer(error_j, np.append(x, 1))
        
        return w_ih, w_ho, z_out
    
    
    def calc_net(self, w, x):
        '''Calculate net input, including proper bias handling.'''
        # Add bias term for the hidden layer calculations
        return np.dot(w, x)       


    def activation(self, net):
        '''Vectorized tanh activation function.'''
        return self.a * np.tanh(self.b * net)


    def derivative_activation(self, net):
        '''Derivative of tanh activation for vectorized inputs.'''
        return self.a * self.b * (1 - np.tanh(self.b * net) ** 2)


    def standardize_data(self, d):
        '''Standardize data to have mean 0 and standard deviation 1.'''
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)
        return (d - mean) / std


    def classify_new_points(self, new_data):
        new_data = self.standardize_data(new_data)
        predictions = np.zeros(new_data.shape[0])  # Initialize predictions array
        for i, x in enumerate(new_data):
            net_j = self.calc_net(self.w_ih, np.append(x, 1))
            y = self.activation(net_j)
            y_with_bias = np.append(y, 1)
            net_k = self.calc_net(self.w_ho, y_with_bias)
            z_out = self.activation(net_k)
            predicted_class = 0 if z_out[0] > z_out[1] else 1
            predictions[i] = predicted_class
            
        self.plot_classes_overlay(new_data, predictions, title='New Data Class Predictions')
        return predictions


    def set_data_vectors(self, data):
        # Assuming the last column is the class label and the rest are features
        train = self.standardize_data(data.iloc[:, :-1].values)
        # labels = data.iloc[:, -1].values
        
        # Get target vectors for samples in using target vector [1, -1] for feature 1 and [-1, 1] for feature 2
        target = np.array([[1, -1] if label == 0 else [-1, 1] for label in data.iloc[:, -1].values])
        self.train = train
        self.target = target


    def plot_classes_overlay(self, new_data, new_predictions, title='Overlayed Class Predictions'):
        
        # Plot original data
        self.plots[2] = self.scatter_plot(2, alpha=0.5, title=title)
        
        # Plot new predictions
        colors = {0: 'blue', 1: 'orange'}
        for class_label, color in colors.items():
            # Filter new data by predicted class
            idx_new = new_predictions == class_label
            self.plots[2].scatter(new_data[idx_new, 0], 
                       new_data[idx_new, 1], 
                       color=color, 
                       label=f'New Class {class_label}', 
                       marker='^', 
                       edgecolor='k')


    def scatter_plot(self, ax, labels=None, alpha=1.0, title="Scatter Plot with Class Labels"):
        
        colors = {0: 'blue', 1: 'orange'} 
        ax = self.plots[ax]
        if labels is None:
            labels = self.labels
        
        for label, color in colors.items():
            idx = labels == label
            ax.scatter(self.features[idx, 0],
                        self.features[idx, 1],
                        label=f'Class {label}',
                        color=color,
                        alpha=alpha)

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        # Position the legend outside the plot area on the right side
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust bbox_to_anchor as needed
        return ax


    def plot_error(self, errors, ax):
        ax.plot(errors, marker='o', linestyle='-', color='blue')
        ax.set_title('Error Evolution During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Squared Error')
        ax.grid(True)


    def classify(self, data):
        
        errors = []        
        error_plot = self.plot_init_data(data)
        
        w_ih, w_ho = self.init_weights()
        predictions = np.zeros(self.train.shape[0])  # Initialize predictions array

        for epoch in range(self.epochs):
            epoch_error = []
            for i, (x, t) in enumerate(zip(self.train, self.target)):
                # update weights
                w_ih, w_ho, z_out = self.update_weights(x, t, w_ih, w_ho)
                # get error
                e = np.mean(0.5 * (t - z_out) ** 2)
                epoch_error.append(e)
                # make prediction
                predicted_class = 0 if z_out[0] > z_out[1] else 1
                predictions[i] = predicted_class  # Store prediction
            
            mean_error = np.mean(epoch_error)
            errors.append(mean_error)

            if mean_error < self.theta:
                print(f'Convergence achieved after {epoch + 1} iterations with mean error {mean_error}.')
                break

        # final results
        print(f"Final weights:\n{w_ih}\n{w_ho}\n")
        self.w_ih = w_ih
        self.w_ho = w_ho
        
        self.scatter_plot(1, labels=predictions, title="Original Data Prediction")
        self.plot_error(errors, error_plot)

        return predictions, errors


def main():
    """Example neural network with:  
        - Number of hidden nodes = 10  
        - Number of input nodes = 2 (input pattern dimension)  
        -  Number of output nodes = 2 (number of categories)   
        - Convergence criterion = 0.1  
        - Convergence rate (i.e. step size) = 0.1  
        - Activation function f(net) = a.tanh (b.net), and a=1.716, b=2/3
        - Target vectors for (ω1, ω2) = [1, -1]t, [-1, 1]t
        - Standardized input patterns and random uniform weights initialization
    """
    
    a = 1.716
    b = 2/3   
    theta=0.01                     
    eta=0.01                         
    nh=10                         
    ni=2                               
    no=2                               
    epochs = 100

    # load data
    filepath = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    data = pd.read_csv(filepath)
    
    sbp = StochasticBP(data, a, b, theta, eta, nh, ni, no, epochs)
    sbp.classify(data)

    # Classify new points D using the trained model
    D = np.array([[2, 2], [-3, -3], [-2, 5], [3, -4]])
    sbp.classify_new_points(D)
    
    plt.show()


if __name__ == "__main__":
    
    sys.exit(main())
    