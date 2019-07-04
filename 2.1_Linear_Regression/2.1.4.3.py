import numpy as np
from sklearn import linear_model
X = np.random.randn(10, 5)
y = np.random.randn(10)
X
y

#clf = linear_model.SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'huber')
clf = linear_model.SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'huber')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')

clf.fit(X, y)
clf.score(X,y)

z =  np.random.randn(5)
print('Predict for ',z , ' is ' , clf.predict(z.reshape(1,-1)))
