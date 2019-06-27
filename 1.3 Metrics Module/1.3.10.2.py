import numpy as np
from sklearn.metrics import precision_recall_curve
y_pred =  np.array([0, 0, 1, 1])
y_true =   np.array([0.1, 0.4, 0.35, 0.8])

precision, recall, thresholds = precision_recall_curve(y_pred,y_true)

print(precision)
print(recall)
print(thresholds)
