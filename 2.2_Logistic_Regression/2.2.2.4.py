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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train
X_test


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)
sgd.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sgd.predict(X_test)
y_pred 


sgd.n_iter_


#probability of all values
pr = sgd.predict_proba(X_test)[0:10,:]
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

 
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='micro')
 