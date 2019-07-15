import matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
X, y = make_hastie_10_2(random_state=0)
X_train = X[:2000]
X_test  = X[2000:]
y_train = y[:2000]
y_test  = y[2000:]


clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()

###########################################################################

for g in range(100,1100 , 100):
    clf = GradientBoostingClassifier(n_estimators=g, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('CM for ' , g , ' estimators is \n' , cm)
    print('Score for ' , g , ' estimators is ' , clf.score(X_test, y_test))    
    print('======================================')
