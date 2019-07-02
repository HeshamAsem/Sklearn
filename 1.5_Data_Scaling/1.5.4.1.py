#Import Libraries
from sklearn.preprocessing import MaxAbsScaler
#----------------------------------------------------

#MaxAbsScaler Data

scaler = MaxAbsScaler(copy=True)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])