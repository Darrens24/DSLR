import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import colorama
import sys


def load_dataset(path):
    return pd.read_csv(path)


def main(arg):
    data = load_dataset(arg[1])
    print("Head of dataset is:", data.head())
    print("--------------------")
    print("Shape of dataset is:", data.shape)
    print("--------------------")
    print("Info of dataset is:")
    data.info()
    # print("--------------------")
    # print("Type of data is:", type(data))
    # print("--------------------")
    # print("Describe of dataset is:", data.describe())


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python describe.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
