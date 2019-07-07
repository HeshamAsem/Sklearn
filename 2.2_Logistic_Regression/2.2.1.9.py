from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile ,f_classif , chi2 


data = load_breast_cancer()
X = data.data
y = data.target

#sel = SelectPercentile(score_func = chi2 , percentile = 20)
#sel = SelectPercentile(score_func = chi2 , percentile = 40)
#sel = SelectPercentile(score_func = chi2 , percentile = 60)
#sel = SelectPercentile(score_func = chi2, percentile = 80)

sel = SelectPercentile(score_func = f_classif , percentile = 20)

sel.fit(X,y)
selected_features = sel.transform(X)
sfeatures = sel.get_support()
print('Selected features = \n' , sfeatures)

x_train, x_test, y_train, y_test = train_test_split(selected_features, y, test_size = 0.2)

logreg = LogisticRegression()
logreg.fit(x_train , y_train)
result= logreg.predict(x_test)
print('accuarcy = ',accuracy_score(y_test , result))

conf = confusion_matrix(y_test , result)
print('confusion matrix \n',  conf)
 