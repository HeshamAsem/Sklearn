#Import Libraries
from sklearn.cluster import KMeans
#----------------------------------------------------

#Applying KMeans Model 

'''
sklearn.cluster.KMeans(n_clusters=8, init='k-means++’, n_init=10, max_iter=300,tol=0.0001,
                       precompute_distances='auto’, verbose=0, random_state=None, copy_x=True,
                       n_jobs=None, algorithm='auto’)
'''

KMeansModel = KMeans(n_clusters=5,init='k-means++', #also can be random
                     random_state=33,algorithm= 'auto') # also can be full or elkan
KMeansModel.fit(X_train)

#Calculating Details
print('KMeansModel Train Score is : ' , KMeansModel.score(X_train))
print('KMeansModel Test Score is : ' , KMeansModel.score(X_test))
print('KMeansModel centers are : ' , KMeansModel.cluster_centers_)
print('KMeansModel labels are : ' , KMeansModel.labels_)
print('KMeansModel intertia is : ' , KMeansModel.inertia_)
print('KMeansModel No. of iteration is : ' , KMeansModel.n_iter_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = KMeansModel.predict(X_test)
print('Predicted Value for KMeansModel is : ' , y_pred[:10])