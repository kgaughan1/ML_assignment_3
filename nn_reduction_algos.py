# Get Data
from sklearn.datasets import load_iris

# Standard Libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler

# Create Pipeline
from sklearn.pipeline import make_pipeline

# Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Neural Network Lib
from sklearn.neural_network import MLPClassifier
import time
from sklearn.neural_network import MLPClassifier

# Get Data

# l = ['FA', 'ICA', 'PCA', 'RP']
#
# o = []
#
# for al in l:
#     print(al)
#
#     data = pd.read_csv('../datasets/reduced_clustered_dataset_{}.csv'.format(al))
#
#     y = data.default
#     X = data.drop('default', axis=1)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                         test_size=0.2,
#                                                         random_state=1,
#                                                         stratify=y)
#
#     pipeline = make_pipeline(StandardScaler(), MLPClassifier())
#
#     alphas = [0.0001]
#     solvers = ['adam']
#     max_iterations = [500]
#
#     hyperparameters = {'mlpclassifier__alpha': alphas,
#                        'mlpclassifier__solver': solvers,
#                        'mlpclassifier__hidden_layer_sizes': [(5, 2)],
#                        'mlpclassifier__random_state': [1],
#                        'mlpclassifier__max_iter': max_iterations
#                        }
#
#     clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#     start = time.time()
#     clf.fit(X, y)
#     end = time.time()
#
#     diff = end - start
#
#     score = clf.score(X=X_test, y=y_test)
#
#     o.append((al, score, diff))
#
# df = pd.DataFrame(o)
# df.columns = ['algorithm', 'score', 'ex_time']
# df.to_csv('nn_algos_reduced_clustered_score.csv')

data = pd.read_csv('../datasets/credit.csv')

X = data.drop('default', axis = 1)
y = data.default

pipeline = make_pipeline(StandardScaler(), MLPClassifier())

alphas = [0.0001]
solvers = ['adam']
max_iterations = [500]

hyperparameters = {'mlpclassifier__alpha': alphas,
                   'mlpclassifier__solver': solvers,
                   'mlpclassifier__hidden_layer_sizes': [(5, 2)],
                   'mlpclassifier__random_state': [1],
                   'mlpclassifier__max_iter': max_iterations
                   }

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
start = time.time()
clf.fit(X, y)
end = time.time()

print(end - start)