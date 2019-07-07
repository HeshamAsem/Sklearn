from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X = data.data
y = data.target


poly = PolynomialFeatures( degree = 2 , include_bias = False)
poly.fit(X)
x_poly = poly.transform(X)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2)

logreg = LogisticRegression()
logreg.fit(x_train , y_train)
result= logreg.predict(x_test)
print('accuracy =',accuracy_score(y_test , result))
 
conf = confusion_matrix(y_test , result)
print('confusion matrix \n',  conf)

names = poly.get_feature_names()

print('fatures names \n' , names)
 

 