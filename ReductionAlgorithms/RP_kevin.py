import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import random

data = pd.read_csv("../datasets/poker.csv")

X = data.drop('hand', axis = 1)
y = data.hand

# data = pd.read_csv("../datasets/poker.csv")
#
# X = data.drop('hand', axis = 1)
# y = data.hand

X = StandardScaler().fit_transform(X)

# model = GRP(n_components=2)
# RPData = model.fit_transform(X)
#
# df = pd.DataFrame(data=RPData, columns=['IC1','IC2'] ) #,'IC3','IC4','IC5','IC6','IC7']) #,'IC8','IC9','IC10','IC11','IC12']) #,'IC13','IC14','IC15', 'IC16','IC17','IC18','IC19','IC20','IC21','IC22','IC23','IC24'])
#
# df.to_csv('RP_poker.csv')

numOfFeatures = 11

o = []

RPData = None

for k in range(1, numOfFeatures):
    print('k', k)

    ave = []
    lValues = []
    for i in range(5):
        print('i', i)
        model = GRP(n_components = k)
        RPData = model.fit_transform(X)

        v = np.mean(cdist(X[:, 0:k], RPData, metric='euclidean'))

        ave.append(v)

        lValues.append(v)

    a = sum(ave)/float(len(ave))

    o.append((k, a, lValues))


df = pd.DataFrame(o)
df.columns = ['k', 'dist_ave', 'dist_values']
df.to_csv('rp_poker_ave_value_values.csv')

# df = pd.DataFrame(RPData)
# df.to_csv('rp_credit_RPData.csv')
#
#
#


######## POKER DATASET #############
#
# data = pd.read_csv("../datasets/poker.csv")

# X = data.drop('hand', axis = 1)
# y = data.hand
#
# numOfFeatures = 11
#
# X = StandardScaler().fit_transform(X)
#
# o = []
#
# RPData = None
#
# for k in range(1, numOfFeatures):
#     print('k', k)
#
#     ave = []
#     # for i in range(5):
#     #     print('i', i)
#     model = GRP(n_components=k)
#     RPData = model.fit_transform(X)
#
#     v = np.mean(cdist(X[:, 0:k], RPData, metric='euclidean'))
#
#     # ave.append(v)
#
#     # a = sum(ave)/float(len(ave))
#
#     o.append((k, v))
#
# df = pd.DataFrame(o)
# df.columns = ['k', 'value']
# df.to_csv('rp_poker_value.csv')
#
# df = pd.DataFrame(RPData)
# df.to_csv('rp_poker_RPData.csv')

