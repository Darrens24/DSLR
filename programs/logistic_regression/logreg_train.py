import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from programs.analysis.describe import Describe
from programs.logistic_regression.utils import load_dataset
from sklearn.metrics import accuracy_score

# import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_cost_function(X, y, Theta):
    m = len(y)
    h = sigmoid(X.dot(Theta))

    J = (-1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    return J


def logistic_gradient(X, y, Theta):
    m = len(y)
    h = sigmoid(X.dot(Theta))

    grad = (1/m) * X.T.dot(h-y)
    return grad


def logistic_gradient_descent(X, y, Theta, alpha, iter):
    J_history = []
    for i in range(iter):

        grad = logistic_gradient(X, y, Theta)
        Theta = Theta - alpha * grad

        J_history.append(logistic_cost_function(X, y, Theta))
    return Theta, J_history

def prepare_features(normalized_data):
    features = pd.DataFrame(normalized_data)
    return features.values

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
    return all_theta, J_history
    
def clean_normalize_data(data):
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
    return cleaned_data, houses


def main(arg):
    np.set_printoptions(suppress=True)
    data = load_dataset(arg)
    normalized_data, classes = clean_normalize_data(data)
    X = prepare_features(normalized_data)
    y = prepare_classes(classes)
    result, cost_history = logistic_regression(X, y, alpha=0.1, iter=1000)
    result = np.array(result)
    np.save("./programs/logistic_regression/theta.npy", result)    
    return


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python logreg_train.py <dataset_train.csv>"
    main(sys.argv[1])
