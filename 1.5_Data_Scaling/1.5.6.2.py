from sklearn.preprocessing import Binarizer
X = [[ 1., -1., -2.],[ 2., 0., -1.], [ 0., 1., -1.]]

transformer = Binarizer(threshold=1.5 ) 
transformer.fit(X)

transformer

transformer.transform(X)
