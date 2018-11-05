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

# data = pd.read_csv("../datasets/credit.csv")
#
# X = data.drop('default', axis = 1)
# y = data.default

data = pd.read_csv("../datasets/poker.csv")

X = data.drop('hand', axis = 1)
y = data.hand

print(type(X))

# numOfFeatures = 16

# X = StandardScaler().fit_transform(X)
#
# model = cluster.FeatureAgglomeration(n_clusters=9)
#
# FAData = model.fit_transform(X)
#
# df = pd.DataFrame(data=FAData, columns=['IC1','IC2','IC3','IC4','IC5','IC6','IC7','IC8','IC9']) #,'IC10','IC11','IC12','IC13','IC14','IC15', 'IC16'] ) # ,'IC17','IC18','IC19','IC20','IC21','IC22','IC23','IC24'])
#
# df.to_csv('FA_poker.csv')



#
# o = []
#
# for k in range(1, numOfFeatures):
#     model = cluster.FeatureAgglomeration(n_clusters=k)
#     model.fit(X)
#
#     X_train_pca = model.transform(X)
#     X_projected = model.inverse_transform(X_train_pca)
#
#     loss = ((X - X_projected) ** 2).mean()
#
#     o.append((k, loss))


# df = pd.DataFrame(o)
# df.columns = ['k', 'loss']
# df.to_csv('FA_credit_reconstruction_loss.csv')






##### POKER DATASET ######

# data = pd.read_csv("../datasets/poker.csv")
#
# X = data.drop('hand', axis = 1)
# y = data.hand
#
# numOfFeatures = 11
#
# X = StandardScaler().fit_transform(X)
#
# o = []
#
# for k in range(1, numOfFeatures):
#     model = cluster.FeatureAgglomeration(n_clusters=k)
#     model.fit(X)
#
#     X_train_pca = model.transform(X)
#     X_projected = model.inverse_transform(X_train_pca)
#
#     loss = ((X - X_projected) ** 2).mean()
#
#     o.append((k, loss))
#
#
# df = pd.DataFrame(o)
# df.columns = ['k', 'loss']
# df.to_csv('FA_poker_reconstruction_loss.csv')
