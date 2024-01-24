import sys
# import numpy as np
from utils import load_dataset
from describe import Describe
# import matplotlib.pyplot as plt


def clean_data(data):
    houses = data["Hogwarts House"]
    print("houses:", houses)
    marks = data[["Astronomy", "Herbology", "Divination", "Muggle Studies",
                  "Ancient Runes", "History of Magic", "Transfiguration",
                  "Potions", "Charms", "Flying"]]
    print("marks:", marks)
    myDescribe = Describe(marks)
    myDescribe.print_stats()


def main(arg):
    data = load_dataset(arg)
    print("data:", data)
    cleaned_data = clean_data(data)
    return


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python logreg_train.py <dataset_train.csv>"
    main(sys.argv[1])
