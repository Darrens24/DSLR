from programs.logistic_regression.logreg_train import prepare_classes, prepare_features
from programs.logistic_regression.logreg_train import sigmoid
from programs.analysis.describe import Describe
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


def load_dataset(path):
    """
    Loads the dataset from a csv file.
    We use pandas to load the csv file into a DataFrame.
    It must be the test file.

    Parameters
    ----------
    path : str
        Path to the csv file.
    """
    assert path.endswith(
        "dataset_test.csv"), "Dataset must be dataset_test.csv"
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Dataset not found")
        exit(1)
    return data


def clean_normalize_data(data):
    """
    Cleans and normalizes the data.
    We use the Describe class to get the mean and standard deviation
    of each subject, then we subtract the mean and divide by the
    standard deviation to normalize the data.

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
        mean_value = myDescribe.stats[subject]["mean"]
        marks.loc[:, subject] = marks.loc[:, subject].fillna(mean_value)
        mean = myDescribe.stats[subject]["mean"]
        std = myDescribe.stats[subject]["std"]
        cleaned_data[subject] = (marks[subject] - mean) / std
    return cleaned_data, houses


def predict(data, all_theta):
    """
    Predicts the Hogwarts House of a student using the trained model.
    We load the cleaned data and the parameters from the numpy files,
    then we use the sigmoid function to predict the house of each student.
    We save the predictions in a csv file.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with the data.
    all_theta : numpy.ndarray
        Matrix with the parameters.
    """
    normalize_data, houses = clean_normalize_data(data)
    X = prepare_features(normalize_data)
    predict = sigmoid(X.dot(all_theta.T))
    predicted_houses = []
    for i in range(len(predict)):
        housevalue = predict[i].argmax()
        if housevalue == 0:
            predicted_houses.append("Gryffindor")
        elif housevalue == 1:
            predicted_houses.append("Hufflepuff")
        elif housevalue == 2:
            predicted_houses.append("Ravenclaw")
        elif housevalue == 3:
            predicted_houses.append("Slytherin")

    predicted_houses_df = pd.DataFrame(
        predicted_houses, columns=["Hogwarts House"])
    index = pd.DataFrame(data["Index"])
    final_df = pd.concat([index, predicted_houses_df], axis=1)
    final_df.to_csv("./datasets/houses.csv", index=False)


def main(dataset_test, theta):
    data = load_dataset(dataset_test)
    try:
        all_theta = np.load(theta)
    except FileNotFoundError:
        print("Theta file not found, train the model first")
        return 1
    predict(data, all_theta)


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(
        sys.argv) == 3, "Usage: python logreg_predict.py <dataset_test.csv> <theta.npy>"
    main(sys.argv[1], sys.argv[2])
