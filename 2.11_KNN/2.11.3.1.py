#Import Libraries
from sklearn.neighbors import NearestNeighbors
#----------------------------------------------------

#Applying NearestNeighborsModel Model 

'''
#sklearn.neighbors.NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='auto’,leaf_size=30, 
#                                   metric='minkowski’, p=2, metric_params=None, n_jobs=None)
'''

NearestNeighborsModel = NearestNeighbors(n_neighbors=4,radius=1.0,algorithm='auto')#it can be:ball_tree,kd_tree,brute
NearestNeighborsModel.fit(X_train)

#Calculating Details
print('NearestNeighborsModel Train kneighbors are : ' , NearestNeighborsModel.kneighbors(X_train[: 5]))
print('NearestNeighborsModel Train radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_train[:  1]))

print('NearestNeighborsModel Test kneighbors are : ' , NearestNeighborsModel.kneighbors(X_test[: 5]))
print('NearestNeighborsModel Test  radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_test[:  1]))
print('----------------------------------------------------')