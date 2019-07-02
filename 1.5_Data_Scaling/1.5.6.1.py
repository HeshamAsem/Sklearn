#Import Libraries
from sklearn.preprocessing import Binarizer
#----------------------------------------------------

#Binarizing Data

scaler = Binarizer(threshold = 1.0)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])