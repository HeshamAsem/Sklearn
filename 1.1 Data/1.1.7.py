#Import Libraries
from sklearn.datasets import make_regression
#----------------------------------------------------

#load regression data

'''
X ,y = make_regression(n_samples=100, n_features=100, n_informative=10,
                       n_targets=1, bias=0.0, effective_rank=None,
                       tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                       random_state=None)
'''

X ,y = make_regression(n_samples=10000, n_features=500,shuffle=True)

#X Data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)