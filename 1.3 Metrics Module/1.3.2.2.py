
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]


mean_squared_error(y_true, y_pred)
mean_squared_error(y_true, y_pred, multioutput='uniform_average') 


mean_squared_error(y_true, y_pred, multioutput='raw_values')
