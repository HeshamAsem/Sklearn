import numpy as np
from sklearn.preprocessing import FunctionTransformer

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]

def function1(z):
    return np.sqrt(z)

FT = FunctionTransformer(func = function1)
FT.fit(X)
newdata = FT.transform(X)
newdata
