import matplotlib.pyplot as plt
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


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred 

 

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred)
 

# Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('SAT degrees')
plt.xlabel('high_GPA')
plt.ylabel('univ_GPA')
plt.show()
