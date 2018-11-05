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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

# Get Data

l = ['FA', 'ICA', 'PCA', 'RP']

o = []

for al in l:
    print(al)

    data = pd.read_csv('../datasets/{}_credit.csv'.format(al))

    y = data.default
    X = data.drop('default', axis=1)

    model = GMM(n_components=2).fit(X)

    o = model.predict(X)

    data['cluster_labels'] = o

    data.to_csv('../datasets/reduced_clustered_dataset_gmm_{}.csv'.format(al))
