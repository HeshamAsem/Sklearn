#Import Libraries
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#----------------------------------------------------

#Applying QDA Model 

'''
sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0,store_covariance=False,
                                                            tol=0.0001,store_covariances=None)
'''

QDAModel = QuadraticDiscriminantAnalysis(tol=0.0001)
QDAModel.fit(X_train, y_train)

#Calculating Details
print('QDAModel Train Score is : ' , QDAModel.score(X_train, y_train))
print('QDAModel Test Score is : ' , QDAModel.score(X_test, y_test))
print('QDAModel means are : ' , QDAModel.means_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = QDAModel.predict(X_test)
y_pred_prob = QDAModel.predict_proba(X_test)
print('Predicted Value for QDAModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for QDAModel is : ' , y_pred_prob[:10])