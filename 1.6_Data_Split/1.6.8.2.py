import numpy as np
from sklearn.model_selection import ShuffleSplit
X = np.arange(10)

n=0.1
#n=0.3
#n=0.5
#n=0.7
#n=0.9

ss = ShuffleSplit(n_splits=5, test_size=n,random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))
