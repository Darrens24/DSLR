import pandas as pd


def load_dataset(path):
    """
    We use pandas to load the csv file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the csv file.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the data.
    """
    return pd.read_csv(path)


def check_feature(data, feature1, feature2):
    """
    Checks if the features are valid.
    The features must be numeric and different.
    The features must be in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with the data.
    feature1 : str
        Name of the first feature.
    feature2 : str
        Name of the second feature.

    Returns
    -------
    int
        0 if the features are valid, 1 otherwise.
    """
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
    """
    Checks if the feature is valid.
    The feature must be numeric.
    The feature must be in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with the data.
    feature : str
        Name of the feature.

    Returns
    -------
    int
        0 if the feature is valid, 1 otherwise.
    """
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
