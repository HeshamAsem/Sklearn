#Import Libraries
from sklearn.naive_bayes import BernoulliNB
#----------------------------------------------------

#Applying BernoulliNB Model 

'''
#sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)
'''

BernoulliNBModel = BernoulliNB(alpha=1.0,binarize=1)
BernoulliNBModel.fit(X_train, y_train)

#Calculating Details
print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)
print('Predicted Value for BernoulliNBModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for BernoulliNBModel is : ' , y_pred_prob[:10])