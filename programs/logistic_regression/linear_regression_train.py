import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def load_data(file_name):
    """
    This function is used to load the data from a csv file
    file_name: the name of the csv file
    return: the data in a pandas dataframe
    """
    assert isinstance(file_name, str), "file_name must be a string"
    assert file_name.endswith('.csv'), "file_name must be a csv file"
    data = pd.read_csv('data.csv')
    return data


def model(X, theta):
    """
    This function is used to compute the model of the linear regression
    """
    return X.dot(theta)


def cost_function(X, y, theta):
    """
    This function is used to compute the cost function of the linear regression
    """
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)


def grad_function(X, y, theta):
    """
    This function is used to compute the gradient of the linear regression
    """
    m = len(y)
    return 1 / m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, alpha, iter):
    """
    This function is used to compute the gradient descent of the linear regression
    """
    cost_history = np.zeros(iter)
    for i in range(0, iter):
        theta = theta - alpha * grad_function(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history


def main():
    """
    Data is normalized to avoid overflow
    Results are saved in result.json and cost_history.json
    Data is plotted with matplotlib
    It will also show the linear regression model
    """
    data = load_data('data.csv')

    y = np.array(data['price']).reshape(data.shape[0], 1)

    x = np.array(data['km']).reshape(data.shape[0], 1)
    mean_x = np.mean(x)
    std_x = np.std(x)
    x_normalized = (x - mean_x) / std_x
    X = np.hstack((x_normalized, np.ones(x_normalized.shape)))

    theta = np.random.randn(2, 1)

    result, cost_history = gradient_descent(X, y, theta, alpha=0.01, iter=1000)

    with open('result.json', 'w') as file:
        json.dump(result.tolist(), file)
    with open('cost_history.json', 'w') as file:
        json.dump(cost_history.tolist(), file)

    prediction = model(X, result)
    plt.scatter(x, y, color='purple', marker='*', s=12, alpha=1)
    plt.plot(x, prediction, color='red')
    plt.title('Scatter distribution of price vs km',
              fontsize=14, color='purple')
    plt.xlabel('Mileage(Km)', fontsize=12, color='purple')
    plt.ylabel('Price(€)', fontsize=12, color='purple')
    plt.show()


if __name__ == '__main__':
    main()
