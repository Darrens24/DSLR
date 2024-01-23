from utils import load_dataset, check_feature, check_valid_feature
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def scatter_2_features(data, feature1, feature2):
    """
    Plot a scatter plot of 2 features
    """
    groups = data.groupby('Hogwarts House')
    if check_feature(data, feature1, feature2) == 1:
        return
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    plt.figure(figsize=(8, 8))
    for name, group in groups:
        plt.scatter(group[feature1], group[feature2],
                    label=name, s=10, color=colors[name])
    title = feature1 + " vs " + feature2
    plt.title(title, fontsize=14, color='black')
    plt.xlabel(feature1, fontsize=14, color='black')
    plt.ylabel(feature2, fontsize=14, color='black')
    plt.grid(True)
    plt.legend()
    plt.show()


def scatter_all_features(data, feature):
    """
    Plot a scatter plot of all features
    """
    groups = data.groupby('Hogwarts House')
    if check_valid_feature(data, feature) == 1:
        return

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    cols_to_plot = [col for col in numeric_cols if col !=
                    'Index' and col != feature]

    n_cols = 2
    n_rows = 2
    n_plots_per_page = n_rows * n_cols

    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    for i in range(0, len(cols_to_plot), n_plots_per_page):
        fig = plt.figure(figsize=(16, 16))

        for j in range(n_plots_per_page):
            if i + j >= len(cols_to_plot):
                break
            ax = fig.add_subplot(n_rows, n_cols, j + 1)
            col = cols_to_plot[i + j]
            for name, group in groups:
                ax.scatter(group[feature], group[col],
                           label=name, s=10, color=colors[name])
            ax.set_title(feature + " vs " + col, fontsize=14, color='black')
            ax.set_xlabel(feature, fontsize=10, color='black')
            ax.set_ylabel(col, fontsize=10, color='black')
            ax.grid(True)
            ax.legend()

    plt.show()


def main(arg):
    """
    Main function
    If 2 features are given, plot a scatter plot of these 2 features
    If 1 feature is given, plot every plots of this feature with all the others
    If no feature is given, plot a scatter plot of the answer of the exercise
    """
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


if __name__ == "__main__":
    """
    Check if the number of arguments is correct
    Check if the dataset is a .csv file
    """
    sys.tracebacklimit = 0
    assert len(sys.argv) == 4 or len(sys.argv) == 3 or len(
        sys.argv) == 2, "Usage: python scatter_plot.py <your_dataset.csv> <first_feature> (optionnal : <second_feature>)"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
