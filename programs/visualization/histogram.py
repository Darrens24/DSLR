import sys
import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt


def load_dataset(path):
    return pd.read_csv(path)


def main(arg):
    data = load_dataset(arg[1])
    print(data)


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 3 or len(
        sys.argv) == 2, "Usage: python scatter_plot.py <your_dataset.csv> (optionnal : <feature> or 'all')"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
