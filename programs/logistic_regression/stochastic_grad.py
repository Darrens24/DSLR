from logreg_train import logistic_cost_function, logistic_gradient
from colorama import Fore
import numpy as np
# import os
import sys
# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', '..')))


def logistic_stochastic_gradient_descent(X, y, Theta, alpha, iter):
    m = len(y)
    J_history = []
    for i in range(iter):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for j in range(m):
            # Maybe we need to reshape X and Y here
            grad = logistic_gradient(X[j], y[j], Theta)
            Theta = Theta - alpha * grad
        J_history.append(logistic_cost_function(X, y, Theta))
    return Theta, J_history


def logistic_minibatch_gradient_descent(X, y, Theta, alpha, iter, batch_size):
    m = len(y)
    J_history = []
    for i in range(iter):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for j in range(0, m, batch_size):
            # Maybe we need to reshape X and Y here
            end = j + batch_size if j + batch_size < m else m
            grad = logistic_gradient(X[j:end], y[j:end], Theta)
            Theta = Theta - alpha * grad
        J_history.append(logistic_cost_function(X, y, Theta))
    return Theta, J_history


def logistic_momentum_gradient_descent(X, y, Theta, alpha, beta, iter):
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
        data = np.load("cleaned_data.npy", allow_pickle=True)
    except FileNotFoundError:
        print(Fore.RED + "[ERROR]" + Fore.RESET +
              " File 'cleaned_data.npy' not found.")
        print(Fore.BLUE + "[INFO]" + Fore.RESET +
              " Please run 'python logreg_train.py' first.")
        return
    print("type of data:", type(data))
    print("data:", data)
    print("-*-"*50)
    dict_data = data.item()
    print("type of dict_data:", type(dict_data))
    print("dict_data:", dict_data)
    np_data = np.array(list(dict_data.values()))
    print("type of np_data:", type(np_data))
    print("np_data:", np_data)
    return


if __name__ == "__main__":
    assert len(sys.argv) == 1, "Usage: python stochastic_grad.py"
    main()
