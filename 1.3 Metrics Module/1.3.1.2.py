from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]


mean_absolute_error(y_true, y_pred)


y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]


mean_absolute_error(y_true, y_pred) # 0.75
mean_absolute_error(y_true, y_pred, multioutput='uniform_average') # 0.75


mean_absolute_error(y_true, y_pred, multioutput='raw_values') # array([0.5, 1. ])
