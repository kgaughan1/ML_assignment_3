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
    data = pd.read_csv('credit.csv')
    y = data.default
    X = data.drop('default', axis=1)

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

    for func in funcs:
        for k in kList:

            X_new = getFeatures(k, X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_new, y,
                                                                test_size=0.2,
                                                                random_state=1,
                                                                stratify=y)
            start = time.time()
            clf = func[0]()
            clf.fit(X_train, y_train)
            score = clf.score(X=X_test, y=y_test)
            end = time.time()
            diff = end - start
            score = clf.score(X=X_test, y=y_test)
            output.append((func[1], k, score, diff))
            count +=1
            print(count)


    df = pd.DataFrame(output, columns=['Algorithm', 'k Value', 'Score', 'Time Difference'])

    df.to_csv('credit_num_of_features.csv')



main()