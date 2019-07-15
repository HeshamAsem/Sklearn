from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train = X[:200]
X_test  = X[200:]
y_train = y[:200]
y_test  = y[200:]


est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))

est.score(X_train,y_train)

y_pred = est.predict(X_test)


###########################################################################

for g in range(100,2000 , 100):
    est = GradientBoostingRegressor(n_estimators=g, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
    score = est.score(X_test, y_test)
    y_pred = est.predict(X_test)
    print('MSE for ' , g , ' estimators is ' , mean_squared_error(y_test, y_pred))
    print('Score for ',g,' estimators is ',score)
    print('======================================')



