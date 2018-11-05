from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score

data = pd.read_csv('../datasets/credit.csv')

X = data.drop('default', axis = 1)
y = data.default
#
# data = pd.read_csv('../datasets/poker.csv')
#
# X = data.drop('hand', axis = 1)
# y = data.hand

numOfClusters = 25

o = []

metricList = ['l1'] #['l2', 'l1', 'manhattan', 'cityblock', 'braycurtis'] # 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']

for m in metricList:
    for k in range(2, numOfClusters):

        print(m, k)

        start = time.time()
        model = GaussianMixture(n_components=k, reg_covar=1e-3).fit(X)
        end   = time.time()

        diff  = end - start

        labels = model.predict(X)
        print(labels)

        sil = metrics.silhouette_score(X, labels, metric=m, sample_size=5000)

        aic = model.aic(X)
        bic = model.bic(X)

        score = metrics.accuracy_score(y, labels)

        o.append((m, k, sil, aic, bic, diff, score))

df = pd.DataFrame(o)
df.columns = ['metric', 'k', 'silhouette_score', 'aic', 'bic', 'execution_time', 'score']
df.to_csv('EM_silhouette_credit_more2.csv')

