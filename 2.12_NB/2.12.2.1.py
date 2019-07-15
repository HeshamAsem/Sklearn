#Import Libraries
from sklearn.naive_bayes import MultinomialNB
#----------------------------------------------------

#Applying MultinomialNB Model 

'''
#naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
'''

MultinomialNBModel = MultinomialNB(alpha=1.0)
MultinomialNBModel.fit(X_train, y_train)

#Calculating Details
print('MultinomialNBModel Train Score is : ' , MultinomialNBModel.score(X_train, y_train))
print('MultinomialNBModel Test Score is : ' , MultinomialNBModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = MultinomialNBModel.predict(X_test)
y_pred_prob = MultinomialNBModel.predict_proba(X_test)
print('Predicted Value for MultinomialNBModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for MultinomialNBModel is : ' , y_pred_prob[:10])