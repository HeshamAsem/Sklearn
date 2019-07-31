#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
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
#Applying DecisionTreeRegressor Model 


DecisionTreeRegressorModel = DecisionTreeRegressor( max_depth=3,random_state=33)

#----------------------------------------------------
#Applying Cross Validate Score :  
'''
model_selection.cross_val_score(estimator,X,y=None,groups=None,scoring=None,cv=’warn’,n_jobs=None,verbose=0,
                                fit_params=None,pre_dispatch=‘2*n_jobs’,error_score=’raise-deprecating’)
'''

#  don't forget to define the model first !!!
CrossValidateScoreTrain = cross_val_score(DecisionTreeRegressorModel, X_train, y_train, cv=3)
CrossValidateScoreTest = cross_val_score(DecisionTreeRegressorModel, X_test, y_test, cv=3)

# Showing Results
print('Cross Validate Score for Training Set: \n', CrossValidateScoreTrain)
print('Cross Validate Score for Testing Set: \n', CrossValidateScoreTest)