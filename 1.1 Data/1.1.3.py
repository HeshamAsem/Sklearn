#Import Libraries
from sklearn.datasets import load_boston
#----------------------------------------------------

#load boston data

BostonData = load_boston()

#X Data
X = BostonData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , BostonData.feature_names)

#y Data
y = BostonData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)