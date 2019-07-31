#Import Libraries
from sklearn.model_selection import cross_val_predict
#----------------------------------------------------

#Applying Cross Validate Predict :  
'''
model_selection.cross_val_predict(estimator, X, y=None, groups=None,cv=’warn’, n_jobs=None,verbose=0,
                                  fit_params=None, pre_dispatch=‘2*n_jobs’,method=’predict’)
'''

#  don't forget to define the model first !!!
CrossValidatePredictionTrain = cross_val_predict(SelectedModel, X_train, y_train, cv=3)
CrossValidatePredictionTest = cross_val_predict(SelectedModel, X_test, y_test, cv=3)

# Showing Results
print('Cross Validate Prediction for Training Set: \n', CrossValidatePredictionTrain[:10])
print('Cross Validate Prediction for Testing Set: \n', CrossValidatePredictionTest[:10])