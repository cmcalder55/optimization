#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created On:   2024/09/14
Last Revision: 0000/00/00

<DESCRIPTION>
'''

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist
from sklearn.datasets import fetch_california_housing

__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"


class LinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.exit_code = None
        self.exit_eval(self.exit_code)

    def feature_scale(self, X):
        """Applies feature scaling on the dataset X."""
        return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    def exit_eval(self, exit_code):
        if exit_code == 1:
            print("Minimum inputs needed: [X, y]")

    def yhat(self, X, w):
        return np.dot(w.T, X)

    def loss(self, yhat, y):
        m = y.shape[1]  # Number of samples
        L = 1 / (2 * m) * np.sum((yhat - y) ** 2)
        return L

    def gradient_descent(self, w, X, y, yhat):
        m = y.shape[1]  # Number of samples
        dldw = 1 / m * np.dot(X, (yhat - y).T)

        # Define a maximum gradient norm
        max_gradient_norm = 1.0  # Adjust as needed

        # Calculate the L2 norm of the gradient
        gradient_norm = np.linalg.norm(dldw)

        # If the gradient norm exceeds the maximum, clip it
        if gradient_norm > max_gradient_norm:
            dldw = dldw * (max_gradient_norm / gradient_norm)

        w -= self.learning_rate * dldw
        return w

    def start(self, scale, X, y):

        # Apply feature scaling to the inputs
        if scale:
            X = self.feature_scale(X)

        x1 = np.ones((1, X.shape[1]))
        X = np.append(X, x1, axis=0)

        self.m = X.shape[1]  # Number of samples
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for epoch in range(self.epochs + 1):
            yhat = self.yhat(X, w)
            loss = self.loss(yhat, y)

            if epoch % 2000 == 0:
                print(f'Cost at epoch {epoch} is {loss:.3f}')

            w = self.gradient_descent(w, X, y, yhat)

        return w
 
    
def lin_reg(input_data):
    m, _ = input_data.data.shape
    bias = np.c_[np.ones((m, 1)), input_data.data]

    X = tf.constant(bias, dtype=tf.float32, name="X")
    y = tf.constant(input_data.target.reshape(-1, 1), dtype=tf.float32, name="y")

    XT = tf.transpose(X)

    theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)

    print(theta)


def plot_greyscale(imgs):
    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(imgs[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(imgs[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(imgs[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(imgs[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()


def normalize_img_data(X_train, y_train, X_test, y_test):
    # flatten 28*28 images to a 784 vector for each image
    n_pix = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], n_pix).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], n_pix).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train /= 255
    X_test /= 255

    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def baseline_model():
    
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=784, kernel_initializer='normal', activation='relu'))
    #  model.add(Dense(num_pixels/2, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def regularization_model(reg = 0.007):
    # create model
    model = Sequential()
    model.add(Dense(512, kernel_regularizer=regularizers.l2(reg), 
                        activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))
    # complile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def dropout_model(rate=0.5):
    # create model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dropout(rate))
    model.add(Dense(10, activation='softmax'))
    # complile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_history(model, title=""):
    history_dict = model.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    epochs = range(1, len(acc_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

    plt.title('Training and validation loss ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def model_and_evaluate(X_train, y_train, X_test, y_test, model, title="", fixed_seed=None):

    if fixed_seed:
        # fix random seed for reproducibility
        np.random.seed(fixed_seed)

    # plot 4 images as gray scale
    plot_greyscale(X_train)

    # normalize data
    (X_train, y_train), (X_test, y_test) = normalize_img_data(X_train, y_train, X_test, y_test)

    # Fit the model
    fit_model = model.fit(X_train, y_train, 
                            validation_data=(X_test, y_test), 
                            epochs=10, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    plot_history(fit_model, title)

    return scores


def run_model(data, m, title, fixed_seed):
    # load (downloaded if needed) the MNIST dataset
    (X_train, y_train), (X_test, y_test) = data

    if m == "base":
        # fit model and evaluate
        model = baseline_model()
    elif m == "dropout":
        model = dropout_model()
    elif m == "reg":
        model = regularization_model()

    # fit model and evaluate
    model_and_evaluate(X_train, y_train, X_test, y_test, model, title, fixed_seed)


def run_all_models(data, title="", fixed_seed=None):
    run_model(data, "base", title, fixed_seed)
    run_model(data, "dropout", title, fixed_seed)
    run_model(data, "reg", title, fixed_seed)


def main(X, y, scale=False):

    reg = LinearRegression()

    np.set_printoptions(precision=3)

    start_time = time.time()  # Start time
    w = reg.start(scale, X, y)

    end_time = time.time()  # End time
    print(f"Final vector: \n{w}")
    print(f"Total computation time: {end_time - start_time:.3f} seconds")

    return w


if __name__ == "__main__":

    n = 500
    X = np.random.rand(1, n)
    y = 3 * X + 0.1 * np.random.randn(1, n)
    main(X, y, scale=True)

    # tensorflow test
    housing = fetch_california_housing()
    lin_reg(housing)

    # tensorflow test with mnist datasets
    run_all_models(mnist.load_data())
    run_all_models(fashion_mnist.load_data())
    