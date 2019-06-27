
import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

