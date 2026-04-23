import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

import mlflow
from mlflow.models import infer_signature



df = pd.read_csv('../data/iris.csv')

X = pd.concat([df['sepal_length'], df['sepal_width'], df['petal_length'], df['petal_width']], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, stratify = y)


mlflow.set_tracking_uri(uri="http://localhost:9080")
loaded_model = mlflow.pyfunc.load_model('models:/m-4c3aa25a2558499c95d13e9e57173636')
y_test_pred = loaded_model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_test_pred)

print(accuracy)




# View model parameters
loaded_model = mlflow.sklearn.load_model('models:/m-4c3aa25a2558499c95d13e9e57173636')

print("Model Parameters:")

for param_name, param_value in loaded_model.get_params().items():

    print('{} : {}'.format(param_name, param_value))

# Access trainable parameters via the `tree_` attribute
n_nodes = loaded_model.tree_.node_count  # Total number of nodes
n_features = loaded_model.n_features_in_  # Number of input features
thresholds = loaded_model.tree_.threshold  # Threshold values for each split
features = loaded_model.tree_.feature  # Feature indices used for each split

print(n_nodes)
print(n_features)
print(thresholds)
print(features)
