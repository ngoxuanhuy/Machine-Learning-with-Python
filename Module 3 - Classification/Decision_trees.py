# Import libraries
import numpy as np
import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.externals.six import StringIO
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import needed data
drug_data = pd.read_csv("drug200.csv", delimiter=",")

# Create feature matrix and label vector
X = drug_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = drug_data['Drug'].values

# Sklearn Decision Trees do not handle categorical variables
# We need to convert some categorical features to numerical values
encode_sex = LabelEncoder()
encode_sex.fit(['F','M'])
X[:,1] = encode_sex.transform(X[:,1])

encode_BP = LabelEncoder()
encode_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = encode_BP.transform(X[:,2])

encode_Chol = LabelEncoder()
encode_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = encode_Chol.transform(X[:,3])

# Split the data set into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Create an instance of the DecisionTreeClassifier called drugTree
# Specify criterion='entropy' so we can see the information gain of each node
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train the model
drugTree.fit(X_train, y_train)

# Prediction
y_pred = drugTree.predict(X_test)

# Evaluation
print("DecisionTree's accuracy: ", accuracy_score(y_test, y_pred))

# Visualize the tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = drug_data.columns[0:5]
targetNames = drug_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')