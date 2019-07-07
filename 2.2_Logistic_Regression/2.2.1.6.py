from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import MinMaxScaler ,Normalizer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X = data.data
y = data.target

#
#scale  = MinMaxScaler()
#scale.fit(X)
#newx = scale.transform(X)

nor = Normalizer(norm = 'max')
nor.fit(X)
newx = nor.transform(X)

x_train, x_test, y_train, y_test = train_test_split(newx, y, test_size = 0.2)


logreg = LogisticRegression()
logreg.fit(x_train , y_train)
result= logreg.predict(x_test)
print(accuracy_score(y_test , result))
 
conf = confusion_matrix(y_test , result)
print('confusion matrix \n',  conf)
 
  