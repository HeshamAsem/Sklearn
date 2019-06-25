#Import Libraries
from sklearn.datasets import load_breast_cancer
#----------------------------------------------------

#load breast cancer data

BreastData = load_breast_cancer()

#X Data
X = BreastData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , BreastData.feature_names)

#y Data
y = BreastData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are \n' , BreastData.target_names)