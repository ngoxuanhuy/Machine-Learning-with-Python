# Import needed packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import data
data = pd.read_csv("FuelConsumption.csv")
# Take a look at the dataset
# print (data.head())

# Data exploration
# print(data.describe()

# Select some features to explore more
sub_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#sub_data.head(10)

# Plot each of these features above
sub_data.hist()
plt.show()

# Create train and test dataset
mask = np.random.rand(len(data)) < 0.8
train = sub_data[mask]
test = sub_data[mask]

# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.title("Relation between Engine size and CO2 Emissions")
plt.show()

# Create model
linear_regression = LinearRegression()
X_train = train[['ENGINESIZE']]
y_train = train[['CO2EMISSIONS']]
linear_regression.fit(X_train, y_train)

# The coefficients
# Coefficient and Intercept in the simple linear regression are the parameters of the fit line
print("Coefficients: ", linear_regression.coef_)
print("Intercept: ", linear_regression.intercept_)

# Plot the fit line over the data
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, linear_regression.coef_[0][0]*X_train + linear_regression.intercept_[0], 'r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.title("Linear relation between the train data of Engine Size and CO2 Emissions")
plt.show()

# Evaluation
X_test = test[['ENGINESIZE']]
y_test = test[['CO2EMISSIONS']]
y_test_predicted = linear_regression.predict(X_test)

print("Mean absolute error:  %.2f" % np.mean(np.absolute(y_test - y_test_predicted)))
print ("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_test_predicted)) # equivalent to np.mean((y_test - y_test_predicted) ** 2)
print("R2-score: %.2f" % r2_score(y_test, y_test_predicted))

# Plot the fit line on test data
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, linear_regression.coef_[0][0]*X_test + linear_regression.intercept_[0], 'r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear relation between the test data of Engine Size and CO2 Emissions")
plt.show()