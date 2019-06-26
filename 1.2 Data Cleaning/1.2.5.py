from sklearn.impute import SimpleImputer


data = [[1,2,0],
        [3,0,1],
        [5,0,0],
        [0,4,6],
        [5,0,0],
        [4,5,5]]


imp = SimpleImputer(missing_values=0, strategy='mean')
imp = imp.fit(data)


modifieddata = imp.transform(data)
print(modifieddata)
