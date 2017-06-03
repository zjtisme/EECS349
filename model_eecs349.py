import pandas as pd
import numpy as np
import json
from sklearn import preprocessing, cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pickle

##df = pd.read_csv('DelayedFlights.txt',sep = ",")
##drop_list = ['ID','Year','ActualElapsedTime','CRSElapsedTime','AirTime','DepDelay',
##             'TaxiIn','TaxiOut','Cancelled','CancellationCode','Diverted','CarrierDelay',
##             'WeatherDelay', 'NASDelay','SecurityDelay','LateAircraftDelay']
##df.drop(drop_list, 1, inplace = True)
##df.to_csv('newData.csv', index=False)

df = pd.read_csv('newData.csv')
def add_nominal(x):
    if x >= 15:
        return 1
    else:
        return -1
df['res'] = df['ArrDelay'].apply(add_nominal)
df.drop(['ArrTime','ArrDelay','TailNum'],1,inplace=True)
df.dropna(axis=0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    map_list = []
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0;
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            map_list.append({column:text_digit_vals})
            df[column] = list(map(convert_to_int, df[column]))

    return df, map_list
# df = df.iloc[::2, :]
df, map_list = handle_non_numerical_data(df)
print(df.head())
print(df.shape)

## store the dictionary of mapping text to number
# with open('dict3.txt', 'w') as f:
#    json.dump(map_list,f)

X = np.array(df.drop(['res'],1))
X = preprocessing.scale(X)
y = np.array(df['res'])

## Basic zeroR baseline
# count = 0.0
# for i in range(np.size(y)):
#     if y[i] == 1:
#         count += 1

# print count / np.size(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,test_size=0.1)

# clf = RandomForestClassifier()
# clf = MLPClassifier(hidden_layer_sizes=500, activation='logistic')
# clf = AdaBoostClassifier()
# clf = QuadraticDiscriminantAnalysis()
# clf = GaussianNB()
clf = DecisionTreeClassifier()
# clf = KNeighborsClassifier(n_neighbors = 50)
clf.fit(X_train, y_train)
# print clf.feature_importances_
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# with open('MLP.pickle','wb') as f:
#     pickle.dump(clf,f)

##clf = KNeighborsClassifier(n_neighbors = 5)
##clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
