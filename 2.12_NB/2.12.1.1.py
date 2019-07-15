#Import Libraries
from sklearn.naive_bayes import GaussianNB
#----------------------------------------------------

#Applying GaussianNB Model 

'''
#sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
'''

GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)

#Calculating Details
print('GaussianNBModel Train Score is : ' , GaussianNBModel.score(X_train, y_train))
print('GaussianNBModel Test Score is : ' , GaussianNBModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)
print('Predicted Value for GaussianNBModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for GaussianNBModel is : ' , y_pred_prob[:10])