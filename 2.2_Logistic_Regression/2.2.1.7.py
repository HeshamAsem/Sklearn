import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

def function1(z):
#    return np.log1p(z)
#    return np.sqrt(z)
    return np.power(z,4)


f = FunctionTransformer(func = function1)
f.fit(X)
x_f = f.transform(X)


x_train, x_test, y_train, y_test = train_test_split(x_f, y, test_size = 0.2)


logreg = LogisticRegression()
logreg.fit(x_train , y_train)
result= logreg.predict(x_test)
print(accuracy_score(y_test , result))
 
conf = confusion_matrix(y_test , result)
print('confusion matrix \n',  conf)

