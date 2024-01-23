import sys
import seaborn as sns
import numpy as np
from utils import load_dataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['toolbar'] = 'None'


def main(arg):
    """
    Plot a pair plot of the dataset
    """
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    data = load_dataset(arg[1])
    numeric_data = data.select_dtypes(include=np.number)
    numeric_data['Hogwarts House'] = data['Hogwarts House']
    numeric_data_clean = numeric_data.dropna().drop(
        columns=['Index'])
    pair_plot = sns.pairplot(
        numeric_data_clean, hue='Hogwarts House', palette=colors, height=1.5)
    plt.subplots_adjust(hspace=0.1, wspace=0.5)
    for ax in pair_plot.axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xlabel(ax.get_xlabel(), fontsize='x-small')

        ax.set_ylabel(ax.get_ylabel(), fontsize='x-small',
                      rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Call the main function
    Check if the number of arguments is correct
    Check if the dataset is a .csv file
    """
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python pair_plot.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
