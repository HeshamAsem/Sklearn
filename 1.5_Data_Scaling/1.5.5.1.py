#Import Libraries
from sklearn.preprocessing import FunctionTransformer
#----------------------------------------------------

#Function Transforming Data
'''
FunctionTransformer(func=None, inverse_func=None, validate= None,
                    accept_sparse=False,pass_y='deprecated', check_inverse=True,
                    kw_args=None,inv_kw_args=None)
'''

scaler = FunctionTransformer(func = lambda x: x**2,validate = True) # or func = function1
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])