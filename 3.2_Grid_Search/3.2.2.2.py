#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
#----------------------------------------------------

#load boston data

BostonData = load_boston()

#X Data
X = BostonData.data
#print('X Data is \n' , X[:10])
#print('X shape is ' , X.shape)
#print('X Features are \n' , BostonData.feature_names)

#y Data
y = BostonData.target
#print('y Data is \n' , y[:10])
#print('y shape is ' , y.shape)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
 
#----------------------------------------------------
#Applying Randomized Grid Searching :  
'''
model_selection.RandomizedSearchCV(estimator, param_distributions,n_iter=10, scoring=None,fit_params=None,n_jobs=
                                   None,iid=’warn’, refit=True, cv=’warn’, verbose=0, pre_dispatch=‘2*n_jobs’,
                                   random_state=None,error_score=’raise-deprecating’, return_train_score=’warn’)

'''

#=======================================================================
#Example : 
from sklearn.svm import SVR
SelectedModel = SVR(epsilon=1,gamma='auto')
SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2]}

RandomizedSearchModel = RandomizedSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)
RandomizedSearchModel.fit(X_train, y_train)
sorted(RandomizedSearchModel.cv_results_.keys())
RandomizedSearchResults = pd.DataFrame(RandomizedSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', RandomizedSearchResults )
print('Best Score is :', RandomizedSearchModel.best_score_)
print('Best Parameters are :', RandomizedSearchModel.best_params_)
print('Best Estimator is :', RandomizedSearchModel.best_estimator_)