from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.data_range_)
print(scaler.data_min_)
print(scaler.data_max_)
newdata = scaler.transform(data)
print(newdata)



newdata = scaler.fit_transform(data)
print(newdata)


scaler = MinMaxScaler(feature_range = (1,5))
