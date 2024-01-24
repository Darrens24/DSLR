from programs.analysis.describe import Describe
from programs.logistic_regression.utils import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
# import numpy as np
# import matplotlib.pyplot as plt


def logistic_regression(data):
    return


def clean_normalize_data(data):
    houses = data["Hogwarts House"]
    print("houses:", houses)
    marks = data[["Astronomy", "Herbology", "Divination", "Muggle Studies",
                  "Ancient Runes", "History of Magic", "Transfiguration",
                  "Potions", "Charms", "Flying"]]
    print("marks:", marks)
    myDescribe = Describe(marks)
    myDescribe.print_stats()
    print("-*-"*10)
    print("type of myDescribe.stats:", type(myDescribe.stats))
    print("marks.columns:", marks.columns)
    # print("myDescribe.mean:", myDescribe.stats["Astronomy"]["mean"])
    cleaned_data = {}
    for subject in marks.columns:
        mean = myDescribe.stats[subject]["mean"]
        std = myDescribe.stats[subject]["std"]
        cleaned_data[subject] = (marks[subject] - mean) / std
    print("cleaned_data:", cleaned_data)
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
