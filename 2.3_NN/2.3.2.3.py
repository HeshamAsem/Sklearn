
from sklearn.neural_network import MLPClassifier
 
X = [[3,6,8],
     [4,5,6],
     [1,5,6],
     [4,7,4],
     [0,5,3],
     [5,6,9],
     [2,4,8],
     [0,6,8]]

y = [0,1,1,1,0,0,0,1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 3),
                   random_state=1,learning_rate='constant',max_iter=100,activation='tanh')

clf.fit(X, y)

print('Coef = \n',  clf.coefs_)
print('============================')

print('Prediction  = ',clf.predict([[10,3,10]]))
print('Prediction  = ',clf.predict([[3,7,9]]))

