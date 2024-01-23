import pandas as pd


def load_dataset(path):
    return pd.read_csv(path)


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
        print("Feature can't be Index")
        return 1
    return 0
