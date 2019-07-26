#Import Libraries
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
#----------------------------------------------------

#Applying AggClusteringModel Model 

'''
sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean’, memory=None, connectivity=None, 
                                        compute_full_tree='auto’, linkage=’ward’,pooling_func=’deprecated’)
'''

AggClusteringModel = AgglomerativeClustering(n_clusters=5,affinity='euclidean',# it can be l1,l2,manhattan,cosine,precomputed
                                             linkage='ward')# it can be complete,average,single

y_pred_train = AggClusteringModel.fit_predict(X_train)
y_pred_test = AggClusteringModel.fit_predict(X_test)

#draw the Hierarchical graph for Training set
dendrogram = sch.dendrogram(sch.linkage(X_train[: 30,:], method = 'ward'))# it can be complete,average,single
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show()

#draw the Hierarchical graph for Test set
dendrogram = sch.dendrogram(sch.linkage(X_test[: 30,:], method = 'ward'))# it can be complete,average,single
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distances')
plt.show()

#draw the Scatter for Train set
plt.scatter(X_train[y_pred_train == 0, 0], X_train[y_pred_train == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[y_pred_train == 1, 0], X_train[y_pred_train == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[y_pred_train == 2, 0], X_train[y_pred_train == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(X_train[y_pred_train == 3, 0], X_train[y_pred_train == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_train[y_pred_train == 4, 0], X_train[y_pred_train == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
plt.title('Training Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show()

#draw the Scatter for Test set
plt.scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X_test[y_pred_test == 2, 0], X_test[y_pred_test == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(X_test[y_pred_test == 3, 0], X_test[y_pred_test == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_test[y_pred_test == 4, 0], X_test[y_pred_test == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
plt.title('Testing Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show()