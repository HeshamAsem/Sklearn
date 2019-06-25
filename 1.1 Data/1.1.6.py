#Import Libraries
from sklearn.datasets import load_diabetes
#----------------------------------------------------

#load diabetes data

DiabetesData= load_diabetes()

#X Data
X = DiabetesData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , DiabetesData.feature_names)

#y Data
y = DiabetesData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)