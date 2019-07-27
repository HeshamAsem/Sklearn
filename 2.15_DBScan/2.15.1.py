#Import Libraries
from sklearn.cluster import DBSCAN
#----------------------------------------------------

#Applying DBSCANModel Model 

'''
sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean’, metric_params=None,
                       algorithm='auto’, leaf_size=30, p=None, n_jobs=None)
'''

DBSCANModel = DBSCAN(metric='euclidean',eps=0.3,min_samples=10,algorithm='auto')#it can be ball_tree, kd_tree, brute
y_pred_train = DBSCANModel.fit_predict(X_train)
y_pred_test = DBSCANModel.fit_predict(X_test)

#Calculating Details
print('DBSCANModel labels are : ' ,DBSCANModel.labels_)
print('DBSCANModel Train data are : ' ,y_pred_train)
print('DBSCANModel Test data are : ' ,y_pred_test)
print('----------------------------------------------------')