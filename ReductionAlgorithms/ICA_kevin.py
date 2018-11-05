import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import time


##### CREDIT DATASET ######

data = pd.read_csv("../datasets/poker.csv")

X = data.drop('hand', axis = 1)
y = data.hand

# data = pd.read_csv("../datasets/poker.csv")
#
# X = data.drop('hand', axis = 1)
# y = data.hand

numOfFeatures = 11

X = StandardScaler().fit_transform(X)

o = []

for k in range(1, numOfFeatures):

    model = ICA(n_components=k)

    start = time.time()
    ICAData = model.fit_transform(X)
    end = time.time()

    kurt = np.mean(kurtosis(ICAData))

    diff = end - start

    X_train_pca = model.transform(X)
    X_projected = model.inverse_transform(X_train_pca)

    loss = ((X - X_projected) ** 2).mean()



    o.append((k, kurt, diff, loss))

df = pd.DataFrame(o)
df.columns = ['k', 'kurtosis', 'execution_time', 'loss']
df.to_csv('ica_poker_kurtosis_reproduction_loss.csv')

#
# model = ICA(n_components=numOfFeatures - 1)
# ICAData = model.fit_transform(X)
#
# df = pd.DataFrame(data=ICAData, columns=['IC1','IC2','IC3','IC4','IC5','IC6','IC7']) #,'IC8','IC9','IC10','IC11','IC12']) #,'IC13','IC14','IC15', 'IC16','IC17','IC18','IC19','IC20','IC21','IC22','IC23','IC24'])
#
# # df.to_csv('df.csv')
#
# df['default'] = y
#
# df.to_csv('ICA_poker.csv')