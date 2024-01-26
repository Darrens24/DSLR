from logreg_train import logistic_cost_function, logistic_gradient
from logreg_train import prepare_classes, prepare_features, sigmoid
from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

import pandas as pd
# import os
import sys
from sklearn.metrics import accuracy_score
# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', '..')))


def logistic_stochastic_gradient_descent(X, y, alpha, iter):
    """
    Performs stochastic gradient descent on the dataset (X, y).
    It uses random permutations of the dataset to update Theta,
    usefull when the dataset is too big to fit in memory, or
    to avoid local minima.

    Parameters
    ----------
    X(m, n) : numpy.ndarray
        Matrix with m training examples and n features.
    y(m, 1) : numpy.ndarray
        Vector with m labels.
    Theta(n, 1) : numpy.ndarray
        Matrix with n parameters.
    alpha : float
        Learning rate.
    iter : int
        Number of iterations.

    Returns
    -------
    Theta : numpy.ndarray
        Matrix with updated parameters.
    """
    m, n = X.shape
    num_classes = len(np.unique(y))
    all_theta = np.zeros((num_classes, n))
    g_J_history = []
    for i in range(num_classes):
        Theta = np.random.randn(n)
        y_c = np.where(y == i, 1, 0)
        J_history = []

        for iteration in range(iter):
            random_index = np.random.randint(m)
            X_shuffle = X[random_index]
            y_shuffle = y_c[random_index]
            grad = logistic_gradient(X_shuffle, y_shuffle, Theta)
            Theta = Theta - alpha * grad
            J_history.append(logistic_cost_function(X, y, Theta))
        all_theta[i] = Theta.T   
        g_J_history.extend(J_history)
    print(all_theta)
    accuracy = accuracy_score(y, np.argmax(sigmoid(X.dot(all_theta.T)), axis=1))
    print(f"Accuracy: {accuracy}")
    return all_theta, g_J_history


def logistic_minibatch_gradient_descent(X, y, alpha, iter, batch_size):
    """
    Performs minibatch gradient descent on the dataset (X, y).
    Like stochastic gradient descent, it uses random permutations
    of the dataset to update Theta, but it uses batches of size
    batch_size instead of single examples.

    Parameters
    ----------
    X(m, n) : numpy.ndarray
        Matrix with m training examples and n features.
    y(m, 1) : numpy.ndarray
        Vector with m labels.
    Theta(n, 1) : numpy.ndarray
        Matrix with n parameters.
    alpha : float
        Learning rate.
    iter : int
        Number of iterations.
    batch_size : int
        Size of the batches.

    Returns
    -------
    Theta : numpy.ndarray
        Matrix with updated parameters.
    """
    m, n = X.shape
    num_classes = len(np.unique(y))
    J_history = []
    all_theta = np.zeros((num_classes, n))
    for i in range(num_classes):
        Theta = np.random.randn(n + 1, 1)
        y_c = (y == i).astype(int)
        for iteration in range(iter):
            indices = np.random.choice(m, batch_size, replace=False)
            X_shuffle = X[indices]
            y_shuffle = y_c[indices]
            grad = logistic_gradient(X_shuffle, y_shuffle, Theta)
            Theta = Theta - alpha * grad
            if i == 3:
                J_history.append(logistic_cost_function(X, y, Theta))
        all_theta[i] = Theta.T
    print(all_theta)
    accuracy = accuracy_score(y, np.argmax(sigmoid(X.dot(all_theta.T)), axis=1))
    print(f"Accuracy: {accuracy}")
    return all_theta, J_history


def logistic_momentum_gradient_descent(X, y, Theta, alpha, beta, iter):
    """
    Performs momentum gradient descent on the dataset (X, y).
    Momentum is a method that helps accelerate SGD in the
    relevant direction and dampens oscillations.
    We can compare it to a ball rolling down a hill, Beta being
    the friction coefficient, v being the velocity of the ball
    and Theta being the position of the ball.

    Parameters
    ----------
    X(m, n) : numpy.ndarray
        Matrix with m training examples and n features.
    y(m, 1) : numpy.ndarray
        Vector with m labels.
    Theta(n, 1) : numpy.ndarray
        Matrix with n parameters.
    alpha : float
        Learning rate.
    beta : float
        Momentum hyperparameter.
    iter : int
        Number of iterations.

    Returns
    -------
    Theta : numpy.ndarray
        Matrix with updated parameters.
    """
    # we added a column of 1s to X in logreg_train.py
    # so the bias is already included in Theta
    J_history = []
    v = np.zeros(Theta.shape)

    for i in range(iter):
        grad = logistic_gradient(X, y, Theta)
        v = beta * v + (1 - beta) * grad
        Theta = Theta - alpha * v
        J_history.append(logistic_cost_function(X, y, Theta))

    return Theta, J_history


def main():
    # pickle allows to use objects in Python
    # it's disabled by default for security reasons

    try:
        data = np.load("./programs/logistic_regression/cleaned_data.npz", allow_pickle=True)
    except FileNotFoundError:
        print(Fore.RED + "[ERROR]" + Fore.RESET +
              " File 'cleaned_data.npy' not found.")
        print(Fore.BLUE + "[INFO]" + Fore.RESET +
              " Please run 'python logreg_train.py' first.")
        return
    cleaned_data = {key: data[key] for key in data.files}
    houses_array = data['houses']
    houses_df = pd.Series(houses_array)
    X = prepare_features(cleaned_data)
    y = prepare_classes(houses_df)
    all_theta, history = logistic_stochastic_gradient_descent(X, y, alpha=0.1, iter=1000)
    # all_thetaMB, historyMB = logistic_minibatch_gradient_descent(X, y, alpha=0.1, iter=1000, batch_size=10)

    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.show()
    return


if __name__ == "__main__":
    assert len(sys.argv) == 1, "Usage: python stochastic_grad.py"
    main()
