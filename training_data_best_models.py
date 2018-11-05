import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

def getData():
    data = pd.read_csv('data/poker.csv')
    y = data.hand
    X = data.drop('hand', axis=1)

    return X, y

def getScaledData(X):
    min_max_scaler = MinMaxScaler()

    return min_max_scaler.fit_transform(X)

def getFeatures(k, X, y):

    best = SelectKBest(chi2, k=k).fit_transform(X, y)

    return best

def main():

    output = []

    X, y = getData()
    X    = getScaledData(X)

    funcs = [(DecisionTreeClassifier, 'DT'), (GradientBoostingClassifier, 'Boosting'), (svm.SVC, 'SVM'),
             (KNeighborsClassifier, 'KNN'), (MLPClassifier, 'Neural Network')]
    kList = [2,4,6,8,10]

    count = 0

    # X_new = getFeatures(k, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=y)

# Decision Tree

    clf = DecisionTreeClassifier(criterion = 'gini', max_depth= None)
    clf.fit(X_train, y_train)

    score = clf.score(X=X_train, y=y_train)
    output.append(('DT', score))
    count +=1
    print(count)

# Boost

    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=10, max_features=None)
    clf.fit(X_train, y_train)

    score = clf.score(X=X_train, y=y_train)
    output.append(('Boost', score))
    count += 1
    print(count)

# SVM

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    score = clf.score(X=X_train, y=y_train)
    output.append(('SVM', score))
    count += 1
    print(count)

# KNN

    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', n_jobs = 1, n_neighbors=5, p = 2)
    clf.fit(X_train, y_train)

    score = clf.score(X=X_train, y=y_train)
    output.append(('KNN', score))
    count += 1
    print(count)

# MLPClassifier

    clf = MLPClassifier(learning_rate_init=0.001, max_iter=500, solver='lbfgs', hidden_layer_sizes=(20,8))
    clf.fit(X_train, y_train)

    score = clf.score(X=X_train, y=y_train)
    output.append(('Neural Network', score))
    count += 1
    print(count)

    df = pd.DataFrame(output, columns=['Algorithm','Score'])

    df.to_csv('poker_training_data_model_test.csv')



main()