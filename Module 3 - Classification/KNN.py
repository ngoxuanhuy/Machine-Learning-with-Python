# Import required library
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import data
df = pd.read_csv('teleCust1000t.csv')
# print(df.head())

# Check how many of each class in our desired output
# Four possible values that correspond to the four customer groups as follow
# 1 - Basic Service
# 2 - Service
# 3 - Plus service
# 4 - Total service
print(df['custcat'].value_counts())

# Visualize our data
df.hist(column='income', bins=50)
# plt.show()

# Convert Pandas data frame to a Numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
y = df['custcat'].values

# Standardization give data zero mean and unit variance
# It is good practice, especially for algorithms such as KNN which is based on distance of cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Create a KNN model with k = 4
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train)
y_hat = neigh.predict(X_test)

# Evaluation
print("Train set Accuracy: ", accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, y_hat))

# Calculate the accuracy of KNN for different Ks
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
confustion_matrix = []

for n in range(1, Ks):
    # Train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[n-1] = accuracy_score(y_test, y_hat)
    std_acc[n-1] = np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])
print(mean_acc)

# Plot model accuracy for different number of Neighbors
plt.plot(range(1,Ks), mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend('Accuracy ', '+/- 3xstd')
plt.ylabel('Accuracy')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)