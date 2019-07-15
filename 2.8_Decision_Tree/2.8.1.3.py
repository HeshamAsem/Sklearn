import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(np.array(4.8).reshape(-1,1))
y_pred 


X2 = np.arange(min(X), max(X), 0.01)
X2 = X2.reshape((len(X2), 1))
plt.scatter(X, y, color = 'r')
plt.plot(X2, regressor.predict(X2), color = 'g')
plt.show()
