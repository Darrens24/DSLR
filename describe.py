import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import colorama
import sys


class Describe:
    """
    Class that computes descriptive statistics for a given dataset.
    It includes the following statistics, calculated on object initialization:
    - count
    - mean
    - variance
    - standard deviation
    - minimum
    - 25th percentile
    - 50th percentile (median)
    - 75th percentile
    - maximum
    """

    def __init__(self, data):
        """
        Constructor:
        - Initializes the data
        - Initializes the stats dictionary
        - Calls all the functions to calculate the statistics
        """
        self.data = data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.stats = {col: {} for col in numeric_cols}
        self.get_count()
        self.get_mean()
        self.get_var()
        self.get_std()
        self.get_min()
        self.get_25()
        self.get_50()
        self.get_75()
        self.get_max()

    def get_count(self):
        """
        Finds the number of non-NaN values in every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            self.stats[col]["count"] = data_col.shape[0]

    def get_mean(self):
        """
        Finds the mean of every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            mean = sum(data_col) / self.stats[col]["count"]
            self.stats[col]["mean"] = round(mean, 6)

    def get_var(self):
        """
        Finds the variance of every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            mean = sum(data_col) / self.stats[col]["count"]
            variance = sum([(i - mean)**2 for i in data_col]) / \
                (self.stats[col]["count"] - 1)
            self.stats[col]["var"] = round(variance, 6)

    def get_std(self):
        """
        Finds the standard deviation of every column.
        """
        for col in self.stats:
            variance = self.stats[col]["var"]
            self.stats[col]["std"] = np.sqrt(variance).round(6)

    def get_min(self):
        """
        Finds the minimum value of every column.
        """
        for col in self.stats:
            self.stats[col]["min"] = self.data[col][0]
            for i in self.data[col]:
                if not np.isnan(i):
                    if i < self.stats[col]["min"]:
                        self.stats[col]["min"] = round(i, 6)

    def get_25(self):
        """
        Finds the 25th percentile of every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            data_col = data_col.sort_values().reset_index(drop=True)
            value = ((self.stats[col]["count"]) - 1) * 0.25
            if value % 1 == 0:
                value = int(value)
                quartile = data_col[value]
            else:
                lower_value = int(value)
                upper_value = lower_value + 1
                interpolation = value - lower_value
                quartile = (data_col[lower_value] * (1 - interpolation)
                            ) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["25%"] = quartile

    def get_50(self):
        """
        Finds the 50th percentile of every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            data_col = data_col.sort_values().reset_index(drop=True)
            value = ((self.stats[col]["count"]) - 1) * 0.50
            if value % 1 == 0:
                value = int(value)
                quartile = data_col[value]
            else:
                lower_value = int(value)
                upper_value = lower_value + 1
                interpolation = value - lower_value
                quartile = (data_col[lower_value] * (1 - interpolation)
                            ) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["50%"] = quartile

    def get_75(self):
        """
        Finds the 75th percentile of every column.
        """
        for col in self.stats:
            data_col = self.data[col].dropna()
            data_col = data_col.sort_values().reset_index(drop=True)
            value = ((self.stats[col]["count"]) - 1) * 0.75
            if value % 1 == 0:
                value = int(value)
                quartile = data_col[value]
            else:
                lower_value = int(value)
                upper_value = lower_value + 1
                interpolation = value - lower_value
                quartile = (data_col[lower_value] * (1 - interpolation)
                            ) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["75%"] = quartile

    def get_max(self):
        """
        Finds the maximum value of every column.
        """
        for col in self.stats:
            self.stats[col]["max"] = self.data[col][0]
            for i in self.data[col]:
                if not np.isnan(i):
                    if i > self.stats[col]["max"]:
                        self.stats[col]["max"] = round(i, 6)

    def find_max_width(self, items):
        """
        Finds the maximum width of a list of strings.
        """
        max_width = 0
        for item in items:
            if len(item) > max_width:
                max_width = len(item)
        return max_width

    def print_stats(self):
        stats_headers = ["count", "mean", "var", "std",
                         "min", "25%", "50%", "75%", "max"]
        stats_data = {}
        for col in self.stats:
            stats_data[col] = [str(self.stats[col].get(stat, 'NaN'))
                               for stat in stats_headers]

        columns = list(self.stats.keys())
        cols_per_row = 3

        for i in range(0, len(columns), cols_per_row):
            selected_columns = columns[i:i + cols_per_row]
            col_widths = []
            for col in selected_columns:
                max_width = self.find_max_width([col] + stats_data[col])
                col_widths.append(max_width)
            column_headers = [""] + selected_columns
            header_row = "".join(
                [f"{column_headers[j]:<{col_widths[j-1] + 2}}"
                    for j in range(1, len(column_headers))])
            print(f"{'':<15}{header_row}")
            for stat in stats_headers:
                row_data = [f"{stat:<15}"]
                for j, col in enumerate(selected_columns):
                    formatted_stat = f"{stats_data[col][stats_headers.index(stat)]:<{col_widths[j] + 2}}"
                    row_data.append(formatted_stat)
                print("".join(row_data))
            print("-" * 60)


def load_dataset(path):
    """
    Loads a dataset from a given path and returns it as a pandas DataFrame.
    """
    return pd.read_csv(path)


def main(arg):
    """
    Main function:
    - Loads the dataset, displaying all rows and columns
    - Creates a Describe object for the dataset
    - Prints the statistics
    """
    data = load_dataset(arg[1])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(data.describe())
    myDescribe = Describe(data)
    myDescribe.print_stats()


if __name__ == "__main__":
    """
    Main entry point of the program:
    - Checks that the user has provided a dataset
    - Checks that the dataset is a .csv file
    - Calls the main function
    """
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python describe.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
