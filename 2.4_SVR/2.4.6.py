
# Importing the libraries
import numpy as np
import pandas as pd


dataset = pd.read_csv('houses.csv')

dataset.head(20)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)


X = dataset[:, :-1]
y = dataset[:, -1]

X
y

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train
X_test
y_train
y_test

from sklearn.svm import SVR
clf = SVR(kernel = 'linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test) 
y_pred

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
