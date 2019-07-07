

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target


xtrain , xtest , ytrain , ytest = train_test_split(X_digits , y_digits ,  test_size = 0.2, )

sgd = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5, random_state=0)

sgd.fit(xtrain, ytrain)


sgd.score(xtrain, ytrain)

y_pred = sgd.predict(xtest)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(ytest, y_pred)



 
