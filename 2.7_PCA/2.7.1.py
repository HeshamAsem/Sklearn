#Import Libraries
from sklearn.decomposition import PCA
#----------------------------------------------------

#Applying PCAModel Model 

'''
#sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False,svd_solver='auto’, tol=0.0,
#                          iterated_power='auto’, random_state=None)
'''

PCAModel = PCA(n_components=2, svd_solver='auto')#it can be full,arpack,randomized
PCAModel.fit(X_train)

#Calculating Details
print('PCAModel Train Score is : ' , PCAModel.score(X_train))
print('PCAModel Test Score is : ' , PCAModel.score(X_test))
print('PCAModel Score Samples is : ' , PCAModel.score_samples(X_test))
print('PCAModel No. of components is : ' , PCAModel.components_)
print('PCAModel Explained Variance is : ' , PCAModel.explained_variance_)
print('PCAModel Explained Variance ratio is : ' , PCAModel.explained_variance_ratio_)
print('PCAModel singular value is : ' , PCAModel.singular_values_)
print('PCAModel mean is : ' , PCAModel.mean_)
print('PCAModel noise variance is : ' , PCAModel.noise_variance_)
print('----------------------------------------------------')