from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]


#transformer = Normalizer(norm='l1' )

#transformer = Normalizer(norm='l2' )

transformer = Normalizer(norm='max' )

transformer.fit(X)
transformer.transform(X)
