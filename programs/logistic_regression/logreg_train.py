import numpy as np
from programs.analysis.describe import Describe
from programs.logistic_regression.utils import load_dataset
import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', '..')))
# import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Computes the sigmoid of z.
    We use the sigmoid function to map values between 0 and 1,
    to use them as probabilities for our logistic regression model
    which is a binary classifier.

    Parameters
    ----------
    z : numpy.ndarray
        Matrix of our model.

    Returns
    -------
    numpy.ndarray
        Matrix with values between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def logistic_cost_function(X, y, Theta):
    """
    Computes the cost function of our logistic regression model,
    using sigmoid as activation function.
    We transpose the log of the sigmoid of our model, and multiply
    it by the labels and their opposite, then we divide by the number
    of training examples.

    Parameters
    ----------
    X(m, n) : numpy.ndarray
        Matrix with m training examples and n features.
    y(m, 1) : numpy.ndarray
        Vector with m labels.
    Theta(n, 1) : numpy.ndarray
        Matrix with n parameters.

    Returns
    -------
    numpy.ndarray
        Matrix with the cost function.
    """
    m = len(y)
    h = sigmoid(X.dot(Theta))
    J = (-1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    return J


def logistic_gradient(X, y, Theta):
    """
    Computes the gradient of our logistic regression model,
    using sigmoid as activation function.
    We multiply the transpose of our model by the difference
    between the sigmoid of our model and the labels, then we
    divide by the number of training examples.

    Parameters
    ----------
    X(m, n) : numpy.ndarray
        Matrix with m training examples and n features.
    y(m, 1) : numpy.ndarray
        Vector with m labels.
    Theta(n, 1) : numpy.ndarray
        Matrix with n parameters.

    Returns
    -------
    numpy.ndarray
        Matrix with the gradient.
    """
    m = len(y)
    h = sigmoid(X.dot(Theta))
    grad = (1/m) * X.T.dot(h-y)
    return grad


def logistic_gradient_descent(X, y, Theta, alpha, iter):
    """
    Performs gradient descent on the dataset (X, y).
    It uses the gradient of our logistic regression model
    to update Theta.

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
    J_history : list
        List with the cost function for each iteration.
    """
    J_history = []
    for i in range(iter):
        grad = logistic_gradient(X, y, Theta)
        Theta = Theta - alpha * grad
        J_history.append(logistic_cost_function(X, y, Theta))
    return Theta, J_history


def logistic_regression(data):
    return


def clean_normalize_data(data):
    """
    Cleans and normalizes the data.
    We use the Describe class to get the mean and standard deviation
    of each subject, then we subtract the mean and divide by the
    standard deviation to normalize the data.
    We save the cleaned data in a numpy file.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with the data.

    Returns
    -------
    cleaned_data : dict
        Dictionary with the cleaned data.
    houses : pandas.Series
        Series with the houses.
    """
    houses = data["Hogwarts House"]
    print("houses:", houses)
    marks = data[["Astronomy", "Herbology", "Divination", "Muggle Studies",
                  "Ancient Runes", "History of Magic", "Transfiguration",
                  "Potions", "Charms", "Flying"]]
    print("marks:", marks)
    myDescribe = Describe(marks)
    # myDescribe.print_stats()
    print("-*-"*50)
    print("type of myDescribe.stats:", type(myDescribe.stats))
    print("marks.columns:", marks.columns)
    # print("myDescribe.mean:", myDescribe.stats["Astronomy"]["mean"])
    cleaned_data = {}
    for subject in marks.columns:
        mean = myDescribe.stats[subject]["mean"]
        std = myDescribe.stats[subject]["std"]
        cleaned_data[subject] = (marks[subject] - mean) / std
    print("-*-"*50)
    print("type of cleaned_data:", type(cleaned_data))
    print("Astronomy:", cleaned_data["Astronomy"])
    print("-*-"*50)
    print("cleaned_data:", cleaned_data)
    np.save("cleaned_data.npy", cleaned_data)
    return cleaned_data


def main(arg):
    data = load_dataset(arg)
    print("data:", data)
    normalized_data = clean_normalize_data(data)
    result = logistic_regression(normalized_data)
    return


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python logreg_train.py <dataset_train.csv>"
    main(sys.argv[1])
