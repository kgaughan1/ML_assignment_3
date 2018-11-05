from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.mixture import GaussianMixture

data = pd.read_csv('../datasets/poker.csv')

X = data.drop('hand', axis = 1)
y_actual = data.hand

numOfClusters = 11

o = []

metricList = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis'] # 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']

for m in metricList:
    for k in range(2, numOfClusters):
        print(m, k)

        start = time.time()
        model = GaussianMixture(n_components=k).fit(X)
        end = time.time()

        diff = end - start

        sil = metrics.silhouette_score(X,  metric=m, sample_size=5000)
        o.append((m, k, sil, diff))

df = pd.DataFrame(o)

df.columns = ['metric', 'k', 'silhouette_score', 'execution_time']

df.to_csv('EM_silhouette_poker.csv')