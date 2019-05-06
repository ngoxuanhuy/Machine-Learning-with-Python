# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# Recall what is linear regression
# Create and plot a simple linear equation: y = 2*(x) + 3
x = np.arange(-5.0, 5.0, 0.1)
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
y_data = y + y_noise
plt.scatter(x, y_data, color='purple')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Create a cubic function's graph - non linear regression graph
# y = x^3 + x^2 + x + 3
x = np.arange(-5.0, 5.0, 0.1)
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
y_data = y + y_noise
plt.scatter(x, y_data, color='pink')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Create a quadratic function's graph
# y = x^2
x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
y_data = y + y_noise
plt.scatter(x, y_data, color='green')
plt.plot(x, y, 'r')
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()

# Create an exponential function with base c is defined by
# Y = a + b*c^x
# where b != 0, c > 0, c != 1
# the base c is consonant and the exponent x is a variable
x = np.arange(-5.0, 5.0, 0.1)
y = 3 + 4 * np.power(5,x)
plt.plot(x, y)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()

# Sigmoidal/Logistic
# y = a + (b/(1+c^(x-d)))
x = np.arange(-5.0, 5.0, 0.1)
y = 1-4/(1+np.power(3,x-2))
plt.plot(x, y, 'brown')
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()

# ===================================================================
#   Non-Linear Regression example
# ===================================================================
df = pd.read_csv("china_gdp.csv")

# Plotting the Dataset
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel("GDP")
plt.xlabel("Year")
plt.show()

# Building the model
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (x + np.exp(-Beta_1*(x-Beta_2)))
    return y

# We need to find the best parameters Beta_1 and Beta_2 for our model.
# Let's normalize our x and y
x_data = x_data/max(x_data)
y_data = y_data/max(y_data)

# "curve_fit" function uses non-linear least squares to fit our sigmoid function
# It optimizes values for the parameters so that the sum of the squared residuals of sigmoid() - y_data is minimum
popt, pcov = curve_fit(sigmoid, x_data, y_data)
# Show the final optimized parameters
print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Plot our resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(x_data, y_data, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Calculate the accuracy of the model
mask = np.random.rand(len(df)) < 0.8
x_train = x_data[mask]
y_train = y_data[mask]
x_test = x_data[~mask]
y_test = y_data[~mask]

# Build the model using train set
popt, pcov = curve_fit(sigmoid, x_train, y_train)

# Predict using test set
y_hat = sigmoid(x_test, *popt)

# Evaluation
print("Mean absolute error: %.3f" % np.mean(np.absolute(y_test-y_hat)))
print("Mean squared error: %.3f" % mean_squared_error(y_test, y_hat))
print('R2-score: %.3f' % r2_score(y, y_data))

