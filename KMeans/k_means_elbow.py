import pandas as pd
from sklearn.cluster import KMeans
import time

data = pd.read_csv('../datasets/poker.csv')

X = data.drop('hand', axis = 1)
y = data.hand

numOfClusters = 11

o = []

for k in range(1, numOfClusters):

    print(k)

    start = time.time()
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    end = time.time()

    diff = end - start

    X["clusters"] = kmeans.labels_

    sse = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    o.append((k, sse, diff))

df = pd.DataFrame(o)
df.columns = ['k', 'SSE', 'execution_time']
df.to_csv('elbow_poker.csv')