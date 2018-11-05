from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ### CHANGE THE FILEPATH TO YOUR FILE ###
# data = pd.read_csv('../datasets/credit.csv')
#
# ### CHANGE 'hand' TO YOUR TARGET FEATURE
# X = data.drop('default', axis = 1)
# y_actual = data.default

data = pd.read_csv("../datasets/credit.csv")

X = data.drop('default', axis = 1)
y = data.default

numOfFeatures = 25

X = StandardScaler().fit_transform(X)

o = []

for k in range(1,numOfFeatures):
    model = PCA(n_components=k)
    model.fit(X)

    # outputData = pca.fit_transform(X)

    X_train_pca = model.transform(X)
    X_projected = model.inverse_transform(X_train_pca)

    loss = ((X - X_projected) ** 2).mean()

    o.append((k, loss))

    # data = pca.explained_variance_ratio_

df = pd.DataFrame(o)
df.columns = ['k', 'loss']
df.to_csv('PCA_credit_reconstruction_loss.csv')

#
#
# data = data / sum(data) * float(100)
#
# df = pd.DataFrame(data=outputData)
#
# ### CHANGE SAVE FILENAME ###
# df.to_csv('PCA_credit.csv')