# Linear Regression is represented by the formula y = mx + b, where:
# y is the dependent variable (target or label),
# m is the slope of the line (coefficient),
# x is the independent variable (feature),
# b is the y-intercept (bias).

import numpy as np
from sklearn import linear_model
import sklearn.metrics as metrics
import matplotlib.pyplot as plt # provides functions for plotting graphs - a simple interface for quick and easy plotting

input = 'linear.txt' # Path to the dataset file -- where the dataset is stored

# The dataset is expected to be in a text file with two columns: features and labels
input_data = np.loadtxt(input, delimiter= ",") # loadtxt loads the data from a text file
x, y = input_data[:, :-1], input_data[:, -1] # split the data into features (x) and labels (y)

training_samples = int(0.8 * len(x)) # 80% of the data for training
testing_samples = len(x) - training_samples # remaining 20% for testing

X_train, y_train = x[:training_samples].reshape(-1, 1), y[:training_samples] # training data
X_test, y_test = x[training_samples:].reshape(-1, 1), y[training_samples:] # testing data
# Create a linear regression model
# linear regression assumes the change in the output (dependent variable - target or label) is proportional to the change in the input (independent variable - feature)
reg_linear = linear_model.LinearRegression() # create an instance of the LinearRegression class

# Train the model using the training data
reg_linear.fit(X_train, y_train) # fit method trains the model using the training data

# Predict the labels for the test set
predictions = reg_linear.predict(X_test) # predict method uses the trained model to predict labels for the test set

# plot and visualize the results
plt.scatter(X_test, y_test, color='blue', label = "Actual") # scatter plot for actual data points
plt.plot(X_test, predictions, color='red', linewidth = 2, label = "Predicted") # line plot for predicted values
plt.xticks(()) # remove x-axis ticks
plt.yticks(()) # remove y-axis ticks
plt.show() # display the plot

# Evaluate the model
print("Performance of Linear regressor:")

# Print the regression line equation

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predictions), 2)) # mean absolute error
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predictions), 2)) # mean squared error
print("Median Absolute Error:", round(metrics.median_absolute_error(y_test, predictions), 2)) # median absolute error
print("Explained Variance Score:", round(metrics.explained_variance_score(y_test, predictions), 2)) # explained variance score
print("R2 Score:", round(metrics.r2_score(y_test, predictions), 2))
#Print the performance metrics of the linear regression model
