from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean_)
newdata = scaler.transform(data)
print(newdata)

newdata = scaler.fit_transform(data) 
print(newdata)


