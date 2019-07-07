#Import Libraries
from sklearn.linear_model import SGDClassifier
#----------------------------------------------------

#Applying SGDClassifier Model 

'''
#sklearn.linear_model.SGDClassifier(loss='hinge’, penalty=’l2’, alpha=0.0001,l1_ratio=0.15, fit_intercept=True,
#                                   max_iter=None,tol=None, shuffle=True, verbose=0, epsilon=0.1,n_jobs=None,
#                                   random_state=None, learning_rate='optimal’, eta0=0.0, power_t=0.5,
#                                   early_stopping=False, validation_fraction=0.1,n_iter_no_change=5,
#                                   class_weight=None,warm_start=False, average=False, n_iter=None)
'''

SGDClassifierModel = SGDClassifier(penalty='l2',loss='squared_loss',learning_rate='optimal',random_state=33)
SGDClassifierModel.fit(X_train, y_train)

#Calculating Details
print('SGDClassifierModel Train Score is : ' , SGDClassifierModel.score(X_train, y_train))
print('SGDClassifierModel Test Score is : ' , SGDClassifierModel.score(X_test, y_test))
print('SGDClassifierModel loss function is : ' , SGDClassifierModel.loss_function_)
print('SGDClassifierModel No. of iteratios is : ' , SGDClassifierModel.n_iter_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = SGDClassifierModel.predict(X_test)
print('Predicted Value for SGDClassifierModel is : ' , y_pred[:10])