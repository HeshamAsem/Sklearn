#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
#----------------------------------------------------

#load breast cancer data

BreastData = load_breast_cancer()

#X Data
X = BreastData.data
#print('X Data is \n' , X[:10])
#print('X shape is ' , X.shape)
#print('X Features are \n' , BreastData.feature_names)

#y Data
y = BreastData.target
#print('y Data is \n' , y[:10])
#print('y shape is ' , y.shape)
#print('y Columns are \n' , BreastData.target_names)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying AggClusteringModel Model 

'''
#sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean’, memory=None, connectivity=None, 
#                                        compute_full_tree='auto’, linkage=’ward’,pooling_func=’deprecated’)
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