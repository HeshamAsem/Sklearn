
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
y_true =  np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
 
precision_recall_fscore_support(y_true, y_pred, average=None)

