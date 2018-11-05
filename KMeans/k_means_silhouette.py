from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import numpy as np
import pandas as pd
import time
from sklearn.metrics.cluster import adjusted_rand_score

# data = pd.read_csv('../datasets/credit.csv')
#
# X = data.drop('default', axis = 1)
# y = data.default
#
# # data = pd.read_csv('../datasets/poker.csv')
# #
# # X = data.drop('hand', axis = 1)
# # y = data.hand
#
# numOfClusters = 25
#
# o = []
#
# metricList = ['manhattan'] # '['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis'] # 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']
#
# for m in metricList:
#     for k in range(2, numOfClusters):
#         print(m, k)
#         start = time.time()
#         model = KMeans(n_clusters=k).fit(X)
#         end = time.time()
#
#         diff = end - start
#
#         labels = model.labels_
#         sil = metrics.silhouette_score(X, labels, metric=m, sample_size=5000)
#
#         # score = adjusted_rand_score(y, labels)
#
#         score = metrics.accuracy_score(y, labels)
#
#         o.append((m, k, sil, diff, score))
#
# df = pd.DataFrame(o)
#
# df.columns = ['metric', 'k', 'silhouette_score', 'execution_time', 'score']
#
# df.to_csv('kmeans_silhouette_credit_score2.csv')

data = pd.read_csv('../datasets/poker.csv')

X = data.drop('hand', axis = 1)
y = data.hand




o = []

metricList = ['manhattan'] # '['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis'] # 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']

# for m in metricList:
#     for k in range(2, numOfClusters):

model = KMeans(n_clusters=2).fit(X)

labels = model.labels_


data = model.fit_transform(X)
print(type(data))

# score = adjusted_rand_score(y, labels)

# score = metrics.accuracy_score(y, labels)

# o.append((m, k, sil, diff, score))

df = pd.DataFrame(data)
df['default'] = labels

#
df.to_csv('kmeans_k_24_poker_data.csv')
