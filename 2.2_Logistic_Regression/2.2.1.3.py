from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=10, solver='lbfgs' , max_iter= 1000 , C = 0.5 , tol = 0.01)
#clf = LogisticRegression(random_state=10, solver='liblinear')
#clf = LogisticRegression(random_state=10, solver='saga')

clf.fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])

score = clf.score(X, y)

print('score = ' , score)
print('No of iterations = ' , clf.n_iter_)
print('Classes = ' , clf.classes_)

 