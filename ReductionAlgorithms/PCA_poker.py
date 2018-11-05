from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA


### CHANGE THE FILEPATH TO YOUR FILE ###
data = pd.read_csv('../datasets/poker.csv')

### CHANGE 'hand' TO YOUR TARGET FEATURE
X = data.drop('hand', axis = 1)
y_actual = data.hand

pca = PCA(n_components=4)

# pca.fit(X)

outputData = pca.fit_transform(X)

print(outputData.shape)
print(type(outputData))

data = pca.explained_variance_ratio_

data = data / sum(data) * float(100)

df = pd.DataFrame(data=outputData)

### CHANGE SAVE FILENAME ###
df.to_csv('PCA_poker.csv')