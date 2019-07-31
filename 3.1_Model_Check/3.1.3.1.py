#Import Libraries
from sklearn.model_selection import cross_val_score
#----------------------------------------------------

#Applying Cross Validate Score :  
'''
model_selection.cross_val_score(estimator,X,y=None,groups=None,scoring=None,cv=’warn’,n_jobs=None,verbose=0,
                                fit_params=None,pre_dispatch=‘2*n_jobs’,error_score=’raise-deprecating’)
'''

#  don't forget to define the model first !!!
CrossValidateScoreTrain = cross_val_score(SelectedModel, X_train, y_train, cv=3)
CrossValidateScoreTest = cross_val_score(SelectedModel, X_test, y_test, cv=3)

# Showing Results
print('Cross Validate Score for Training Set: \n', CrossValidateScoreTrain)
print('Cross Validate Score for Testing Set: \n', CrossValidateScoreTest)