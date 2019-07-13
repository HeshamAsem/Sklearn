#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import numpy as np
#----------------------------------------------------
# creating data
X = np.random.rand(10000,2)
y = np.random.rand(10000,0)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying MiniBatchKMeans Model 

'''
#sklearn.cluster.MiniBatchKMeans(n_clusters=8, init='k-means++’, max_iter=100,batch_size=100, 
#                                verbose=0, compute_labels=True,random_state=None, tol=0.0,
#                                max_no_improvement=10,init_size=None, n_init=3, reassignment_ratio=0.01)
'''

MiniBatchKMeansModel = MiniBatchKMeans(n_clusters=5,batch_size=50,init='k-means++') #also can be random
MiniBatchKMeansModel.fit(X_train)

#Calculating Details
print('MiniBatchKMeansModel Train Score is : ' , MiniBatchKMeansModel.score(X_train))
print('MiniBatchKMeansModel Test Score is : ' , MiniBatchKMeansModel.score(X_test))
print('MiniBatchKMeansModel centers are : ' , MiniBatchKMeansModel.cluster_centers_)
print('MiniBatchKMeansModel labels are : ' , MiniBatchKMeansModel.labels_)
print('MiniBatchKMeansModel intertia is : ' , MiniBatchKMeansModel.inertia_)
print('MiniBatchKMeansModel No. of iteration is : ' , MiniBatchKMeansModel.n_iter_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = MiniBatchKMeansModel.predict(X_test)
print('Predicted Value for MiniBatchKMeansModel is : ' , y_pred[:10])