import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
y = data.target

skf = StratifiedKFold(n_splits = 5  )
predict = np.zeros(y.shape[0])
p_rf = np.zeros(y.shape[0])

for train,test in skf.split(X,y):
    x_train = X[train]
    x_test = X[test]
    y_train = y[train]
    y_test = y[test]
    logreg = LogisticRegression()
    logreg.fit(x_train , y_train)
    result= logreg.predict(x_test)
    predict[test] = result
    
    rf = RandomForestClassifier(n_estimators = 100 , n_jobs = -1)
    rf.fit(x_train , y_train)
    res_rf = rf.predict(x_test)
    p_rf[test] = res_rf
    

print('Log Reg ' , accuracy_score(y , predict))
print('Rand For' , accuracy_score(y , p_rf ))
