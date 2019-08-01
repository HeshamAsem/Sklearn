#Import Libraries
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
#----------------------------------------------------

#Applying Randomized Grid Searching :  
'''
model_selection.RandomizedSearchCV(estimator, param_distributions,n_iter=10, scoring=None,fit_params=None,n_jobs=
                                   None,iid=’warn’, refit=True, cv=’warn’, verbose=0, pre_dispatch=‘2*n_jobs’,
                                   random_state=None,error_score=’raise-deprecating’, return_train_score=’warn’)

'''

#=======================================================================
#Example : 
#from sklearn.svm import SVR
#SelectedModel = SVR(epsilon=0.1,gamma='auto')
#SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}
#=======================================================================
RandomizedSearchModel = RandomizedSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)
RandomizedSearchModel.fit(X_train, y_train)
sorted(RandomizedSearchModel.cv_results_.keys())
RandomizedSearchResults = pd.DataFrame(RandomizedSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', RandomizedSearchResults )
print('Best Score is :', RandomizedSearchModel.best_score_)
print('Best Parameters are :', RandomizedSearchModel.best_params_)
print('Best Estimator is :', RandomizedSearchModel.best_estimator_)