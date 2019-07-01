#Import Libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , f_classif 
#----------------------------------------------------

#----------------------------------------------------
#Feature Selection by KBest 
#print('Original X Shape is ' , X.shape)
FeatureSelection = SelectKBest(score_func= chi2 ,k=3) # score_func can = f_classif 
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())