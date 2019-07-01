#Import Libraries
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2 , f_classif 
#----------------------------------------------------

#----------------------------------------------------
#Feature Selection by Generic
#print('Original X Shape is ' , X.shape)
FeatureSelection = GenericUnivariateSelect(score_func= chi2, mode= 'k_best', param=3) # score_func can = f_classif : mode can = percentile,fpr,fdr,fwe 
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())