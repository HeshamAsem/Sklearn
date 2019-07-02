#Import Libraries
from sklearn.preprocessing import PolynomialFeatures
#----------------------------------------------------

#Polynomial the Data

scaler = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])