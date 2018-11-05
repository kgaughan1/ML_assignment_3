import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
from sklearn import datasets, cluster
import time
from sklearn import metrics


##### CREDIT DATASET ######

data = pd.read_csv("../datasets/credit.csv")

X = data.drop('default', axis = 1)
print(type(X))
y = data.default

numOfFeatures = 25

X = StandardScaler().fit_transform(X)

# o = []

# for k in range(1, numOfFeatures):
#     model = cluster.FeatureAgglomeration(n_clusters=k)
#
#     model.fit(X)
#     reducedDataSet = model.transform(X)
#
#     print(reducedDataSet.shape)
#
#     labels = model.labels_
#
#     print('labels')
#     print(labels)
#
#     # print('labels')
#     # print(labels)
#
#     # sil = metrics.silhouette_score(X, labels, metric='euclidian', sample_size=5000)

from sklearn.decomposition import PCA
import numpy as np


pca = cluster.FeatureAgglomeration(n_clusters=2)
pca.fit(X)

U, S, VT = np.linalg.svd(X - X.mean(0))

X_train_pca = pca.transform(X)

X_train_pca2 = (X - pca.mean_).dot(pca.components_.T)

X_projected = pca.inverse_transform(X_train_pca)
X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_

loss = ((X - X_projected) ** 2).mean()

print(loss)