import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from programs.analysis.describe import Describe
from programs.logistic_regression.utils import load_dataset
from sklearn.metrics import accuracy_score

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
    m = X.shape[0]
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

def prepare_features(normalized_data):
    features = pd.DataFrame(normalized_data)
    numeric_features = features.select_dtypes(include=[np.number])
    return numeric_features.values

def prepare_classes(classes):
    unique_classes = np.unique(classes)
    classes_num = {classe: i for i, classe in enumerate(unique_classes)}
    return classes.map(classes_num)

def logistic_regression(X, y, alpha, iter):
    m, n = X.shape
    num_labels = len(np.unique(y))
    X = np.insert(X, 0, 1, axis=1)
    all_theta = np.zeros((num_labels, n+1 ))
    for i in range(num_labels):
        init_theta = np.zeros(n + 1)
        y_c = np.where(y == i, 1, 0)
        theta, J_history = logistic_gradient_descent(X, y_c, init_theta, alpha, iter)
        all_theta[i] = theta
    accuracy = accuracy_score(y, np.argmax(sigmoid(X.dot(all_theta.T)), axis=1))
    print(f"Accuracy: {accuracy}")
    return all_theta, J_history
    
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
    marks = data[["Astronomy", "Herbology", "Divination", "Muggle Studies",
                  "Ancient Runes", "History of Magic", "Transfiguration",
                  "Potions", "Charms", "Flying"]]
    myDescribe = Describe(marks)
    cleaned_data = {}
    for subject in marks.columns:
        marks[subject] = marks[subject].fillna(myDescribe.stats[subject]["mean"])
        mean = myDescribe.stats[subject]["mean"]
        std = myDescribe.stats[subject]["std"]
        cleaned_data[subject] = (marks[subject] - mean) / std
    np.savez("./programs/logistic_regression/cleaned_data.npz", **cleaned_data, houses=houses.values)
    return cleaned_data, houses


def main(arg):
    np.set_printoptions(suppress=True)
    data = load_dataset(arg)
    normalized_data, classes = clean_normalize_data(data)
    X = prepare_features(normalized_data)
    y = prepare_classes(classes)
    result, cost_history = logistic_regression(X, y, alpha=0.1, iter=1000)
    print(result)
    result = np.array(result)
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.show()
    np.save("./programs/logistic_regression/theta.npy", result) 
    return


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python logreg_train.py <dataset_train.csv>"
    main(sys.argv[1])
