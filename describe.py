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
        self.get_count()
        self.get_mean()
        self.get_std()
        self.get_min()
        self.get_25()
        self.get_50()
        self.get_75()
        self.get_max()

    def get_count(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            self.stats[col]["count"] = data_col.shape[0]

    def get_mean(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            mean = sum(data_col) / self.stats[col]["count"]
            self.stats[col]["mean"] = round(mean, 6)

    def get_std(self):
        for col in self.stats:
            data_col = self.data[col].dropna()
            variance = np.var(data_col, ddof=1)
            self.stats[col]["std"] = np.sqrt(variance).round(6)

    def get_min(self):
        for col in self.stats:
            self.stats[col]["min"] = self.data[col][0]
            for i in self.data[col]:
                if np.isnan(i) == False:
                    if i < self.stats[col]["min"]:
                        self.stats[col]["min"] = round(i, 6)

    def get_25(self):
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
                quartile = (data_col[lower_value] * (1 - interpolation)) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["25%"] = quartile

    def get_50(self):
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
                quartile = (data_col[lower_value] * (1 - interpolation)) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["50%"] = quartile

    def get_75(self):
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
                quartile = (data_col[lower_value] * (1 - interpolation)) + (data_col[upper_value] * interpolation)
            quartile = round(quartile, 6)
            self.stats[col]["75%"] = quartile

    def get_max(self):
        for col in self.stats:
            self.stats[col]["max"] = self.data[col][0]
            for i in self.data[col]:
                if np.isnan(i) == False:
                    if i > self.stats[col]["max"]:
                        self.stats[col]["max"] = round(i, 6)
 
    def find_max_width(self, items):
        max_width = 0
        for item in items:
            if len(item) > max_width:
                max_width = len(item)
        return max_width

    def print_stats(self):
        stats_headers = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        stats_data = {}
        for col in self.stats:
            stats_data[col] = [str(self.stats[col].get(stat, 'NaN')) for stat in stats_headers]

        columns = list(self.stats.keys())
        cols_per_row = 3

        for i in range(0, len(columns), cols_per_row):
            selected_columns = columns[i:i + cols_per_row]
            col_widths = []
            for col in selected_columns:
                max_width = self.find_max_width([col] + stats_data[col])
                col_widths.append(max_width)
            column_headers = [""] + selected_columns
            header_row = "".join([f"{column_headers[j]:<{col_widths[j-1] + 2}}" for j in range(1, len(column_headers))])
            print(f"{'':<15}{header_row}")
            for stat in stats_headers:
                row_data = [f"{stat:<15}"]
                for j, col in enumerate(selected_columns):
                    formatted_stat = f"{stats_data[col][stats_headers.index(stat)]:<{col_widths[j] + 2}}"
                    row_data.append(formatted_stat)
                print("".join(row_data))
            print("-" * 60)

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
    myDescribe.print_stats()


if __name__ == "__main__":
    sys.tracebacklimit = 0
    assert len(sys.argv) == 2, "Usage: python describe.py <your_dataset.csv>"
    assert sys.argv[1].endswith(".csv"), "Dataset must be a .csv file"
    main(sys.argv)
