# Library to read csv file effectively
# To plot the data
import matplotlib.pyplot as plt
import numpy as np
import pandas


# Method to read the csv file
def load_data(file_name):
    # print(file_name)
    column_names = ['area', 'rooms', 'price']
    # To read columns
    io = pandas.read_csv(file_name, names=column_names, header=None)
    x_val = (io.values[1:, 0])
    y_val = (io.values[1:, 2])
    size_array = len(y_val)
    for i in range(size_array):
        x_val[i] = float(x_val[i])
        y_val[i] = float(y_val[i])
    return x_val, y_val


# Call the method for a specific file

# NORMALIZE THE INPUT
def feature_normalize(train_X):
    global mean, std
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    return (train_X - mean) / std


x_raw, y_raw = load_data('area_price.csv')
X = feature_normalize(x_raw)
# print(x_raw, X)
y = y_raw

# Modeling
w, b = 0.1, 0.1
num_epoch = 1000
learning_rate = 1e-3
for e in range(num_epoch):
    # Calculate the gradient of the loss function with respect to arguments (model parameters) manually.
    y_predicted = w * X + b
    grad_w, grad_b = (y_predicted - y).dot(X), (y_predicted - y).sum()
    # Update parameters.
    w, b = w - learning_rate * grad_w, b - learning_rate * grad_b
print(w, b)
y_estimated = w * X + b
# Plot the data
plt.scatter(X, y, color='red')
plt.xlabel('Apartment area (square meters)')
plt.ylabel('Apartment price (1000 Euros)')
plt.title('Apartment price vs area in Berlin')
plt.plot(X, y_estimated, linewidth=4.0)
plt.savefig('numpy_regression.png', transparent=True)
plt.show()
