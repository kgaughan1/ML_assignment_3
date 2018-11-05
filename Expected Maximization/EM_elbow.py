import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
import time

data = pd.read_csv('../datasets/poker.csv')

X = data.drop('hand', axis = 1)
y = data.hand

numOfClusters = 11

o = []

for k in range(1, numOfClusters):

    print(k)

    start = time.time()
    model = GMM(n_components=k).fit(X)
    end = time.time()

    diff = end - start

    o.append((k, sse, diff))

df = pd.DataFrame(o)
df.columns = ['k', 'SSE', 'execution_time']
df.to_csv('elbow_poker.csv')