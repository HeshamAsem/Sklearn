import numpy as np
from sklearn.model_selection import LeavePOut
X = np.ones(10)
lpo = LeavePOut(p=3)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))
