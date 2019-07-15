import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
        
        rf = RandomForestClassifier(n_estimators = n , n_jobs = -1)
        rf.fit(x_train , y_train)
        res_rf = rf.predict(x_test)
        p_rf[test] = res_rf
    print('Rand For' ,n , '   '  ,  accuracy_score(y , p_rf ))
 

