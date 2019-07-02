#Import Libraries
from sklearn.preprocessing import MinMaxScaler
#----------------------------------------------------

#MinMaxScaler for Data

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])