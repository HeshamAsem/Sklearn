
# Importing the libraries
import numpy as np
import pandas as pd


dataset = pd.read_csv('Earthquakes.csv')

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

y_pred = clf.predict([[90,12,12,-5,54,0.3,0.9,3.5,16.2,10]]) 
y_pred

