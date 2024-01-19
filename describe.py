import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import colorama
import sys

class Describe:
    def __init__(self, data):
        self.data = data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.stats = {col: {} for col in numeric_cols}
        # print(self.stats)

    def get_count(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            self.stats[col]["count"] = data_col.shape[0]

    def get_mean(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            self.stats[col]["mean"] = sum(data_col) / self.stats[col]["count"]

    def get_std(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            variance = np.var(data_col, ddof=1)
            self.stats[col]["std"] = np.sqrt(variance)

def load_dataset(path):
    return pd.read_csv(path)

def main(arg):
    data = load_dataset(arg[1])
    # print("Head of dataset is:", data.head())
    # print("--------------------")
    # print("Shape of dataset is:", data.shape)
    # print("--------------------")
    # print("Info of dataset is:")
    # data.info()
    # print("--------------------")
    # print("Type of data is:", type(data))
    # print("--------------------")
    # print("Describe of dataset is:", data.describe())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(data.describe())
    myDescribe = Describe(data)
    myDescribe.get_count()
    myDescribe.get_mean()
    myDescribe.get_std()
    print(myDescribe.stats)


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python describe.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
