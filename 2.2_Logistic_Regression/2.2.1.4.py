import pandas as pd

dataset = pd.read_csv('heart.csv')

dataset.head(20)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X
y


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train
X_test
y_train
y_test 


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train
X_test


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
clss = LogisticRegression(random_state = 0)
clss.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clss.predict(X_test)
y_pred 


clss.n_iter_
clss.classes_


#probability of all values
pr = clss.predict_proba(X_test)[0:10,:]
pr

#probability of zeros
pr = clss.predict_proba(X_test)[0:10,0]
pr

#probability of ones
pr = clss.predict_proba(X_test)[0:10,1]
pr


#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
 
 

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred)

 

 