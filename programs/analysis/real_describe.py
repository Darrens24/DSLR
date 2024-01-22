import sys
import pandas as pd
from describe import load_dataset


def main(arg):
    """
    Main function:
    - Loads the dataset, displaying all rows and columns
    - Uses the pandas describe method to compute the statistics
    """
    data = load_dataset(arg[1])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(data.describe())


if __name__ == "__main__":
    """
    Main entry point of the program.
    - Checks that the user has provided a dataset
    - Checks that the dataset is a .csv file
    - Calls the main function
    """
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python describe.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
