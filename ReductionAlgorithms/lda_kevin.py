import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


##### CREDIT DATASET ######

### CHANGE THE FILEPATH TO YOUR FILE ###
data = pd.read_csv('../datasets/credit.csv')

### CHANGE 'hand' TO YOUR TARGET FEATURE
X = data.drop('default', axis = 1)
y = data.default

numOfFeatures = 25

model = LDA(n_components=numOfFeatures, store_covariance=True)

model.fit(X, y)

LDAComponents = model.transform(X)

# var = np.cumsum(np.round(model.explained_variance_ratio_, decimals=3) * 100)

cov = model.covariance_

eigvals, eigvecs = np.linalg.eig(cov)

o = eigvals / float(sum(eigvals)) * 100

o2 = []

for each in o:
    each = round(each, 2)

    o2.append(each)

print(o2)

print(LDAComponents)



#
# o = []
#
# for k in range(1, numOfFeatures):
#     print(k)
#
#     model = LDA(n_components=k)
#
#     model.fit(X, y)
#
#     # eigen = model.explained_variance_ratio_
#
#     var = np.cumsum(np.round(model.explained_variance_ratio_, decimals=3) * 100)
#
#     o.append((k, var))
#
# df = pd.DataFrame(data=o, columns= ['k', 'eigen'])
#
# ### CHANGE SAVE FILENAME ###
# df.to_csv('LDA_credit_eigenvalues.csv')