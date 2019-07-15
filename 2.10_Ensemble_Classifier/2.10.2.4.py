import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

data = load_breast_cancer()
X = data.data
y = data.target

skf = StratifiedKFold(n_splits = 5  )
p_rf = np.zeros(y.shape[0])


for n in range(10,100,10):
    for train,test in skf.split(X,y):
        x_train = X[train]
        x_test = X[test]
        y_train = y[train]
        y_test = y[test]
        
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    print('score For' ,n , '   '  ,  accuracy_score(y_test , y_pred))



