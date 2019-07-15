import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

balance_data = pd.read_csv('data2.csv',sep= ',', header= None)
print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)



print ("Dataset:: ")
balance_data.head()

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

clf_gini.predict([[4, 4, 3, 3]])
clf_entropy.predict([[4, 4, 3, 3]])

y_pred = clf_gini.predict(X_test)
y_pred


y_pred_en = clf_entropy.predict(X_test)
y_pred_en

print ("Accuracy for gini is ", accuracy_score(y_test,y_pred)*100)

print ("Accuracy for entropy is ", accuracy_score(y_test,y_pred_en)*100)


    