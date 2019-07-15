#Import Libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#----------------------------------------------------

#Applying LDA Model 

'''
sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd’,shrinkage=None,priors=None,
                                                         n_components=None,store_covariance=False,tol=0.0001)
'''

LDAModel = LinearDiscriminantAnalysis(n_components=3,solver='svd',tol=0.0001)
LDAModel.fit(X_train, y_train)

#Calculating Details
print('LDAModel Train Score is : ' , LDAModel.score(X_train, y_train))
print('LDAModel Test Score is : ' , LDAModel.score(X_test, y_test))
print('LDAModel means are : ' , LDAModel.means_)
print('LDAModel classea are : ' , LDAModel.classes_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = LDAModel.predict(X_test)
y_pred_prob = LDAModel.predict_proba(X_test)
print('Predicted Value for LDAModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for LDAModel is : ' , y_pred_prob[:10])