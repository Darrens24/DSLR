import sys
from scatter_plot import check_valid_feature, load_dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def histogram_feature(data, feature):
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    groups = data.groupby('Hogwarts House')
    house_data = {house: group for house, group in groups}

    plt.figure(figsize=(12, 10))
    for house, group in house_data.items():
        plt.hist(group[feature].dropna(), bins=20,
                 color=colors[house], alpha=0.3, label=house)
    plt.title(f"{feature} Distribution")
    plt.xlabel('Marks')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def histogram_all_features(data):
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    numeric_data_columns = [col for col in data.select_dtypes(
        include=np.number).columns if col != 'Index']

    groups = data.groupby('Hogwarts House')
    house_data = {house: group for house, group in groups}
    num_columns = len(numeric_data_columns)

    for i in range(0, num_columns, 4):
        plt.figure(figsize=(12, 10))

        for j in range(4):
            if i + j < num_columns:
                ax = plt.subplot(2, 2, j+1)
                for house, group in house_data.items():
                    ax.hist(group[numeric_data_columns[i+j]].dropna(),
                            bins=20, color=colors[house], alpha=0.3, label=house)
                ax.set_title(f"{numeric_data_columns[i+j]} Distribution")
                ax.set_xlabel('Marks')
                ax.set_ylabel('Frequency')
                ax.legend()
        plt.tight_layout()
        if i < num_columns - 4:
            plt.show(block=False)
        else:
            plt.show()


def histogram_answer(data):
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    groups = data.groupby('Hogwarts House')
    house_data = {house: group for house, group in groups}
    feature_1 = 'Arithmancy'
    feature_2 = 'Care of Magical Creatures'

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    for house, group in house_data.items():
        plt.hist(group[feature_1].dropna(), bins=20,
                 color=colors[house], alpha=0.3, label=house)
    plt.title(f"{feature_1} Distribution")
    plt.xlabel('Marks')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 2, 2)
    for house, group in house_data.items():
        plt.hist(group[feature_2].dropna(), bins=20,
                 color=colors[house], alpha=0.3, label=house)
    plt.title(f"{feature_2} Distribution")
    plt.xlabel('Marks')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(arg):
    data = load_dataset(arg[1])
    if len(arg) == 2:
        histogram_answer(data)
        return
    elif len(arg) == 3:
        feature = arg[2]
        if feature == 'all':
            histogram_all_features(data)
            return
        elif check_valid_feature(data, feature) == 0:
            histogram_feature(data, feature)
            return


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 3 or len(
        sys.argv) == 2, "Usage: python scatter_plot.py <your_dataset.csv> (optionnal : <feature> or 'all')"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
