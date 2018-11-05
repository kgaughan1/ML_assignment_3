import os
import pandas as pd
from sklearn.model_selection import train_test_split

directory = '../datasets'

for filename in os.listdir(directory):
    try:

        path = directory + '/' + filename

        data = pd.read_csv(path)

        # Format the data
        y = data.default
        X = data.drop('default', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            random_state=1,
                                                            stratify=y)

        df = pd.DataFrame(X_test)

        df['default'] = y_test

        df.to_csv(path + 'reduced.csv')

    except:
        print('Not a credit file.')


