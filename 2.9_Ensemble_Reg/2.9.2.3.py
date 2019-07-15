
from sklearn.metrics import mean_squared_error
from sklearn.datasets import  make_friedman1
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split


x , y = make_friedman1(n_samples = 1000 , noise = 1)

x.shape
y.shape


xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = 0.3 )

model = GradientBoostingRegressor(n_estimators = 100 , learning_rate = 1.5 , max_depth = 1)
model.fit(xtrain , ytrain)

model.predict(xtest)

mean_squared_error(ytest , model.predict(xtest))

model.score(xtest , ytest)
