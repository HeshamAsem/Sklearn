
# Importing the libraries
import numpy as np
import pandas as pd


dataset = pd.read_csv('satf.csv')

dataset.head(20)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)


X = dataset[:, :-1]
y = dataset[:, -1]

X
y

from sklearn.svm import SVR
clf = SVR(kernel = 'linear')
clf.fit(X, y)

y_pred = clf.predict([[3.48,684,649,3.61]]) 
y_pred

