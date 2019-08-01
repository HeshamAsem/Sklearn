#Import Libraries
from sklearn.model_selection import GridSearchCV
import pandas as pd
#----------------------------------------------------

#Applying Grid Searching :  
'''
model_selection.GridSearchCV(estimator, param_grid, scoring=None,fit_params=None, n_jobs=None, iid=’warn’,
                             refit=True, cv=’warn’, verbose=0,pre_dispatch=‘2*n_jobs’, error_score=
                             ’raisedeprecating’,return_train_score=’warn’)

'''

#=======================================================================
#Example : 
#from sklearn.svm import SVR
#SelectedModel = SVR(epsilon=0.1,gamma='auto')
#SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}
#=======================================================================
GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)
GridSearchModel.fit(X_train, y_train)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)