
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


 
from sklearn.linear_model import SGDRegressor
 #clf = linear_model.SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'huber')
sgd = SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'huber')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')


sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test) 
y_pred

sgd.score(X_train,y_train)
sgd.score(X_test,y_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


