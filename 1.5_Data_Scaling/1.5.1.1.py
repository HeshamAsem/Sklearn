#Import Libraries
from sklearn.preprocessing import StandardScaler
#----------------------------------------------------

#Standard Scaler for Data

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])