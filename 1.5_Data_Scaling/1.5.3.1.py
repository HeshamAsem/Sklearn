#Import Libraries
from sklearn.preprocessing import Normalizer
#----------------------------------------------------

#Normalizing Data

scaler = Normalizer(copy=True, norm='l2') # you can change the norm to 'l1' or 'max' 
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])