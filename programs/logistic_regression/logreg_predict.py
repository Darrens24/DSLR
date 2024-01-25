import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from programs.analysis.describe import Describe
from programs.logistic_regression.utils import load_dataset
from programs.logistic_regression.logreg_train import sigmoid
from programs.logistic_regression.logreg_train import prepare_classes, prepare_features
from programs.logistic_regression.logreg_train import clean_normalize_data
# import matplotlib.pyplot as plt
def load_dataset(path):
    assert path.endswith(
        "dataset_test.csv"), "Dataset must be dataset_test.csv"
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Dataset not found")
        exit(1)
    return data

def predict(data, all_theta):
    normalize_data, houses = clean_normalize_data(data)
    X = prepare_features(normalize_data)
    X = np.insert(X, 0, 1, axis=1)
    predict = sigmoid(X.dot(all_theta.T))
    for i in range(len(predict)):
        housevalue = predict[i].argmax()
        if housevalue == 0:
            houses[i] = "Gryffindor"
        elif housevalue == 1:
            houses[i] = "Hufflepuff"
        elif housevalue == 2:
            houses[i] = "Ravenclaw"
        elif housevalue == 3:
            houses[i] = "Slytherin"
    houses = pd.DataFrame(houses)
    houses.columns = ["Hogwarts House"]
    index = pd.DataFrame(data["Index"])
    houses = pd.concat([index, houses], axis=1)
    houses.to_csv("./datasets/houses.csv", index=False)

def main(dataset_test, theta):
    data = load_dataset(dataset_test)
    try:
        all_theta = np.load(theta)
    except FileNotFoundError:
        print("Theta file not found, train the model first")
        return 1
    predict(data, all_theta)
if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python logreg_predict.py <dataset_test.csv> <theta.npy>"
    main(sys.argv[1], sys.argv[2])
