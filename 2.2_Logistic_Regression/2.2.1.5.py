import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score , confusion_matrix
 

iris = load_iris()
X = iris.data
y = iris.target

X
y

skf = StratifiedKFold(n_splits = 5)

 
predict = np.zeros(y.shape[0])

for train,test in skf.split(X,y):
    x_train = X[train]
    x_test = X[test]
    y_train = y[train]
    y_test = y[test]
    logreg = LogisticRegression()
    logreg.fit(x_train , y_train)
    result= logreg.predict(x_test)
    predict[test] = result
    print('train data \n' , train)
    print('test data \n ', test)
    print('result \n',result)
    print('this accuracy \n' , accuracy_score(y_test , result))
    print('===============================')
    print()

print('total accuracy \n',accuracy_score(y , predict))

conf = confusion_matrix(y , predict)

print('confusion matrix \n',  conf)
 
 