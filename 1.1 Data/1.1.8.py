#Import Libraries
from sklearn.datasets import make_classification
#----------------------------------------------------

#load classification data

'''
X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2,
                           n_repeated = 0, n_classes = 2, n_clusters_per_class = 2, weights = None,
                           flip_y = 0.01, class_sep = 1.0, hypercube = True, shift = 0.0,
                           Scale() = 1.0, shuffle = True, random_state = None)
'''

X, y = make_classification(n_samples = 100, n_features = 20, shuffle = True)

#X Data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)