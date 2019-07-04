import pandas as pd

# Importing the dataset
dataset = pd.read_csv('satf.csv')
dataset.head(10)


X = dataset.iloc[:,:1] 
y = dataset.iloc[:, -1]

X
y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_train
X_test
y_train
y_test 

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_train2 = poly_reg.fit_transform(X_train)
X_test2 = poly_reg.fit_transform(X_test)

# No Polynomial for y

X_train2.shape 
X_test2.shape 

from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train2, y_train )

y_pred2 = lin_reg_2.predict(X_test2) 

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred2)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred2)

from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred2)

 

 
 
