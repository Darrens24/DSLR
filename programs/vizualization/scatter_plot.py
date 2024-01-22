import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
# import colorama
from describe import Describe
import sys

def check_feature(data, feature1, feature2):
    if feature1 not in data.columns or feature2 not in data.columns:
        print("Feature not found")
        return 1
    if feature1 == feature2:
        print("Feature must be different")
        return 1
    if data[feature1].dtype == object or data[feature2].dtype == object:
        print("Feature must be numeric")
        return 1
    if feature1 == "Index" or feature2 == "Index":
        print("Feature must not")
        return 1
    return 0

def check_valid_feature(data, feature):
    if feature not in data.columns:
        print("Feature not found")
        return 1
    if data[feature].dtype == object:
        print("Feature must be numeric")
        return 1
    if feature == "Index":
        print("Feature must not")
        return 1
    return 0

def scatter_2_features(data, feature1, feature2):
    groups = data.groupby('Hogwarts House')
    if check_feature(data, feature1, feature2) == 1:
        return
    plt.figure(figsize=(8,8))
    for name, group in groups:
        plt.scatter(group[feature1], group[feature2], label=name, s=10)
    title = feature1 + " vs " + feature2
    plt.title(title , fontsize=14)
    plt.xlabel(feature1 , fontsize=14, color='red')
    plt.ylabel(feature2, fontsize=14, color='blue')
    plt.grid(True)
    plt.legend()
    plt.show()


def scatter_all_features(data, feature):
    groups = data.groupby('Hogwarts House')
    if check_valid_feature(data, feature) == 1:
        return

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    cols_to_plot = [col for col in numeric_cols if col != 'Index' and col != feature]

    n_cols = 2
    n_rows = 2
    n_plots_per_page = n_rows * n_cols

    for i in range(0, len(cols_to_plot), n_plots_per_page):
        fig = plt.figure(figsize=(16, 16))

        for j in range(n_plots_per_page):
            if i + j >= len(cols_to_plot):
                break
            ax = fig.add_subplot(n_rows, n_cols, j + 1)
            col = cols_to_plot[i + j]
            for name, group in groups:
                ax.scatter(group[feature], group[col], label=name, s=10)
            ax.set_title(feature + " vs " + col, fontsize=14)
            ax.set_xlabel(feature, fontsize=10, color='red')
            ax.set_ylabel(col, fontsize=10, color='blue')
            ax.grid(True)
            ax.legend()

    plt.show()

def load_dataset(path):
    return pd.read_csv(path)

def main(arg):
    data = load_dataset(arg[1])
    if len(arg) == 4:
        feature1 = arg[2]
        feature2 = arg[3]
        scatter_2_features(data, feature1, feature2)
    elif len(arg) == 3:
        feature = arg[2]
        scatter_all_features(data, feature)
    elif len(arg) == 2:
        scatter_2_features(data, "Astronomy", "Defense Against the Dark Arts")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 4 or len(sys.argv) == 3 or len(sys.argv) == 2, "Usage: python scatter_plot.py <your_dataset.csv> <first_feature> (optionnal : <second_feature>)"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
