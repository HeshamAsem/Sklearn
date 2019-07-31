#Import Libraries
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

#----------------------------------------------------

#load boston data

BostonData = load_boston()

#X Data
X = BostonData.data
#y Data
y = BostonData.target


#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

 
#----------------------------------------------------
#Applying Ridge Regression Model 

RidgeRegressionModel = Ridge(alpha=1.0,random_state=33)

#----------------------------------------------------
#Applying Cross Validate Predict :  
'''
model_selection.cross_val_predict(estimator, X, y=None, groups=None,cv=’warn’, n_jobs=None,verbose=0,
                                  fit_params=None, pre_dispatch=‘2*n_jobs’,method=’predict’)
'''

#  don't forget to define the model first !!!
CrossValidatePredictionTrain = cross_val_predict(RidgeRegressionModel, X_train, y_train, cv=3)
CrossValidatePredictionTest = cross_val_predict(RidgeRegressionModel, X_test, y_test, cv=3)

# Showing Results
print('Cross Validate Prediction for Training Set: \n', CrossValidatePredictionTrain[:10])
print('Cross Validate Prediction for Testing Set: \n', CrossValidatePredictionTest[:10])

