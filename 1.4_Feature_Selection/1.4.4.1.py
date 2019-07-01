#Import Libraries
from sklearn.feature_selection import SelectFromModel
#----------------------------------------------------

#Feature Selection by KBest 
#print('Original X Shape is ' , X.shape)

'''
from sklearn.linear_model import LinearRegression
thismodel = LinearRegression()
'''

FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())